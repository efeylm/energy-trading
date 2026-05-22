"""
Energy agents: Producers and Consumers.

Combines:
- Qiu et al. POMG observation/action structure (Section 3.3)
- ADD Bitirme MB/MC curve-based pricing logic (Section 3.1.2)

Each agent:
1. Observes: inflexible load, battery state, market signals, hour
2. Decides: battery action (charge/discharge/store) + market order (price, quantity)
3. Uses MB/MC curves to determine willingness to pay / minimum ask
4. Has emergency bidding when battery drops below starvation threshold

Faz 2 Reward (satıcı için):
    reward = matched_qty × clearing_price + unmatched_qty × FiT
Faz 2 Reward (alıcı için):
    reward = -net_cost  (alıcı maliyet minimize eder)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import os
import json

from src.config import AgentMBParams, AgentMCParams, SimConfig
from src.market import Order


@dataclass
class Observation:
    """Agent's private observation at time step t.
    
    o_n,t = [P^inf_n,t, λ^b_t, λ^s_t, λ^avg_t, hour]
    """
    inflexible_load: float     # P^inf_n,t (kW)
    best_bid: float            # λ^b_t ($/kWh)
    best_ask: float            # λ^s_t ($/kWh)
    last_avg_price: float      # λ^avg_t ($/kWh)
    hour: int                  # t


@dataclass
class Action:
    """Agent's action at time step t."""
    order: Optional[Order]     # Market order (None if agent doesn't trade)


class EnergyAgent:
    """Base class for energy trading agents.
    
    Agents trade based on their instantaneous net energy position (inflexible load):
    - Surplus (inflexible_load < 0) -> Seller
    - Deficit (inflexible_load > 0) -> Buyer
    """
    
    def __init__(
        self,
        agent_id: int,
        agent_type: str,       # "producer" or "consumer"
        mb_params: AgentMBParams,
        mc_params: AgentMCParams,
        config: SimConfig,
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.mb_params = mb_params
        self.mc_params = mc_params
        self.config = config
        
        # Tracking
        self.total_cost = 0.0           # Cumulative energy cost ($)
        self.total_reward = 0.0         # Cumulative reward
        self.total_traded_kwh = 0.0     # Cumulative traded energy
        self.unmet_demand = 0.0         # Cumulative unmet demand (kWh)
        self.curtailed_energy = 0.0     # Cumulative curtailed surplus (kWh)
        self.hourly_log = []            # Detailed per-hour records
        
        # --- Hourly MB Data (from Ausgrid extraction) ---
        self.hourly_mb_data = {}
        # Try agent-specific data first, then global config path
        specific_path = f"src/data/mb_agent_{agent_id}.json"
        path_to_load = specific_path if os.path.exists(specific_path) else self.config.mb_data_path
        
        if os.path.exists(path_to_load):
            try:
                with open(path_to_load, 'r') as f:
                    self.hourly_mb_data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load MB data for agent {agent_id}: {e}")

        # --- Iterative auction state ---
        self._auction_role: Optional[str] = None
        self._auction_unit_size: float = 0.5
        self._auction_total_qty: float = 0.0
        self._auction_units_to_trade: int = 0
        self._auction_units_traded: int = 0
        self._auction_current_offer: Optional[float] = None
        self._current_hour: int = 0
    
    def compute_mb(self, quantity: float, hour: Optional[int] = None) -> float:
        """Marginal Benefit: MB(q) = alpha * exp(-beta * q) (Exponential)"""
        q = max(0.0, quantity)
        if hour is not None:
            total_minutes = int((hour + 1) * 30)
            h = (total_minutes // 60) % 24
            m = total_minutes % 60
            time_key = f"{h}:{m:02d}"
            if time_key in self.hourly_mb_data:
                params = self.hourly_mb_data[time_key]
                alpha = params.get("alpha", self.mb_params.alpha)
                beta = params.get("beta", self.mb_params.beta)
                return alpha * math.exp(-beta * q)
        return self.mb_params.alpha * math.exp(-self.mb_params.beta * q)
    
    def compute_mc(self, quantity: float) -> float:
        """Marginal Cost: MC(q) = gamma * q + delta"""
        q = max(0.0, quantity)
        return self.mc_params.gamma * q + self.mc_params.delta

    def compute_uili_price(self, quantity: float) -> float:
        """Use-It-or-Lose-It (UILI) seller pricing.

        Ters exponential fiyat modeli — üretim miktarıyla fiyat düşer:

            P(Q) = FiT + (ToU - FiT) * exp(-lambda * Q)

        - Q = 0  → P = ToU  (hiç üretim baskısı yok, maksimum fiyat)
        - Q → ∞  → P → FiT  (yüksek üretim baskısı, taban fiyata yaklaşır)

        Satıcı çok ürettiğinde fiyatını düşürmek zorunda kalır;
        aksi hâlde eşleşemeyip FiT'ten bile daha az kazanır.
        """
        q = max(0.0, quantity)
        fit = getattr(self.config, "fit_price", 0.06)
        tou = getattr(self.config, "tou_price", 0.28)
        lam = getattr(self.config, "lambda_uili", 0.4)
        return fit + (tou - fit) * math.exp(-lam * q)
    
    def decide_action(self, obs: Observation) -> Action:
        """Heuristic decision-making based on net energy position."""
        self._current_hour = obs.hour
        market_quantity = obs.inflexible_load  # positive=deficit, negative=surplus
        
        order = None
        if market_quantity > 0.01:
            # BUYER - Agent needs energy
            if self.agent_type != "producer":
                bid_price = self.compute_mb(market_quantity, hour=self._current_hour)
                order = Order(
                    agent_id=self.agent_id,
                    price=bid_price,
                    quantity=market_quantity,
                    is_buy=True,
                    is_emergency=False,
                )
        elif market_quantity < -0.01:
            # SELLER - Agent has surplus → Use-It-or-Lose-It pricing
            # Fiyat üretim miktarıyla ters orantılı: çok üretim → ucuz fiyat
            sell_quantity = abs(market_quantity)
            ask_price = self.compute_uili_price(sell_quantity)
            order = Order(
                agent_id=self.agent_id,
                price=ask_price,
                quantity=sell_quantity,
                is_buy=False,
                is_emergency=False,
            )

        return Action(order=order)

    def process_trade_results(
        self,
        trades_as_buyer: list,
        trades_as_seller: list,
        unmet_demand: float,
        curtailed_surplus: float,
        hour: int,
    ):
        """Update internal state and tracking."""
        buy_cost = sum(t.price * t.quantity for t in trades_as_buyer)
        sell_income = sum(t.price * t.quantity for t in trades_as_seller)
        bought_kwh = sum(t.quantity for t in trades_as_buyer)
        sold_kwh = sum(t.quantity for t in trades_as_seller)
        
        net_cost = buy_cost - sell_income
        self.total_cost += net_cost
        self.total_traded_kwh += bought_kwh + sold_kwh
        self.unmet_demand += unmet_demand
        self.curtailed_energy += curtailed_surplus
        
        self.hourly_log.append({
            "hour": hour,
            "buy_cost": buy_cost,
            "sell_income": sell_income,
            "net_cost": net_cost,
            "bought_kwh": bought_kwh,
            "sold_kwh": sold_kwh,
            "unmet_demand": unmet_demand,
            "curtailed": curtailed_surplus,
        })
    
    # --- Iterative Auction logic (simplified) ---

    def setup_auction(self, role: str, market_quantity: float, unit_size: float) -> int:
        self._auction_role = role
        self._auction_unit_size = unit_size
        self._auction_total_qty = abs(market_quantity)
        self._auction_units_to_trade = max(1, math.ceil(self._auction_total_qty / unit_size))
        self._auction_units_traded = 0
        self._auction_current_offer = None
        return self._auction_units_to_trade

    def auction_valuation(self) -> Optional[float]:
        if self._auction_units_traded >= self._auction_units_to_trade:
            return None
        q = self._auction_units_traded * self._auction_unit_size
        if self._auction_role == 'buyer':
            return self.compute_mb(q, hour=self._current_hour)
        # UILI: seller reservation price = UILI curve at cumulative qty traded so far
        return self.compute_uili_price(q)

    def auction_initial_offer(self, margin: float) -> Optional[float]:
        v = self.auction_valuation()
        if v is None: return None
        if self._auction_role == 'buyer':
            self._auction_current_offer = v * (1.0 - margin)
        else:
            self._auction_current_offer = v * (1.0 + margin)
        return self._auction_current_offer

    def auction_update_offer(self, best_bid: float, best_ask: float, alpha: float):
        if self._auction_current_offer is None: return
        v = self.auction_valuation()
        if v is None: return
        midpoint = (best_bid + best_ask) / 2.0
        if self._auction_role == 'buyer':
            new = self._auction_current_offer + alpha * (midpoint - self._auction_current_offer)
            self._auction_current_offer = min(v, max(self._auction_current_offer, new))
        else:
            new = self._auction_current_offer - alpha * (self._auction_current_offer - midpoint)
            self._auction_current_offer = max(v, min(self._auction_current_offer, new))

    def auction_record_trade(self):
        self._auction_units_traded += 1
        self._auction_current_offer = None

    def auction_units_remaining(self) -> int:
        return max(0, self._auction_units_to_trade - self._auction_units_traded)

    def get_reward(self, hour: int) -> float:
        """
        Faz 2 Reward Fonksiyonu:

        Satıcı (inflexible_load < 0 → net_load negatif):
            reward = matched_qty × clearing_price + unmatched_qty × FiT
            - Eşleşen enerji → piyasa fiyatından gelir
            - Eşleşemeyen enerji → FiT taban fiyatından gelir (şebekeye ihracat)

        Alıcı (inflexible_load > 0):
            reward = -net_cost
            - Alıcı maliyet minimize eder (negatif maliyet = iyi)

        FiT değeri config'den alınır; config bağlantısı yoksa 0.06 $/kWh kullanılır.
        """
        if not self.hourly_log:
            return 0.0

        last = self.hourly_log[-1]
        fit_price = getattr(self.config, "fit_price", 0.06)
        beta      = getattr(self.config, "reward_beta", 0.0)

        # Satıcı mı?
        # net_load negatif → sold_kwh > 0 veya curtailed > 0
        sold_kwh    = last.get("sold_kwh", 0.0)
        curtailed   = last.get("curtailed", 0.0)
        sell_income = last.get("sell_income", 0.0)

        is_seller = (sold_kwh + curtailed) > 1e-6

        if is_seller:
            # Temel reward: matched_qty × clearing_price + unmatched_qty × curtail_rate
            # curtail_rate = 0.0 → satamadıysan sıfır kazanırsın (eski: FiT=+$0.06)
            curtail_rate  = getattr(self.config, "reward_curtail_rate", 0.0)
            unmatched_reward = curtailed * curtail_rate
            reward = sell_income + unmatched_reward

            # β fiyat primi: FiT üzerinde satılan her kWh için ekstra ödül
            # Bu terim ajana "fiyatı gereksiz yere düşürme" sinyali verir.
            # price_premium = sold_kwh × (avg_clearing_price - FiT)
            if beta > 0.0 and sold_kwh > 1e-6:
                price_premium = sell_income - sold_kwh * fit_price
                reward += beta * price_premium
        else:
            # Alıcı: maliyet minimize (reward = -net_cost)
            reward = -last["net_cost"]

        return reward
    
    def __repr__(self) -> str:
        return f"EnergyAgent(id={self.agent_id}, type={self.agent_type}, cost={self.total_cost:.3f}$)"