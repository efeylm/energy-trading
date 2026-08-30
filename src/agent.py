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


# MB eğrisi türetilirken Q_max için güvenlik tabanı (kWh).
# src/extract_mb.py içindeki Q_MAX_FLOOR ile aynı değer olmalıdır.
MB_Q_MAX_FLOOR = 0.1


@dataclass
class Observation:
    """Agent's private observation at time step t.

    Makale Eq. (6):
        o_n,t = [P^inf_n,t, λ^avg_t, t]

    NOT: Eskiden buradaki gözlem vektörü best_bid (λ^b) ve best_ask (λ^s)
    alanlarını da içeriyordu. PartialMatchDoubleAuction her periyodun sonunda
    emir defterlerini boşalttığı için bu iki alan ajan gözleminde HER ZAMAN
    0.0 değerini alıyordu; yani öğrenmeye hiçbir bilgi katmıyor, buna karşılık
    NN state boyutunu 5'e şişiriyordu. Kaldırıldılar → state boyutu 3.
    """
    inflexible_load: float     # P^inf_n,t (kW)
    last_avg_price: float      # λ^avg_t ($/kWh) — önceki periyodun takas fiyatı
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
                    raw_mb_data = json.load(f)
                self.hourly_mb_data = self._rebuild_mb_curves(raw_mb_data)
            except Exception as e:
                print(f"Warning: Could not load MB data for agent {agent_id}: {e}")

        self._current_hour: int = 0
    
    def _rebuild_mb_curves(self, raw_mb_data: dict) -> dict:
        """Ausgrid JSON'undan MB eğrilerini config tarifelerine göre yeniden kurar.

        Tek kaynak ilkesi: MB eğrisinin tavanı ToU, tabanı FiT'tir ve her ikisi de
        SADECE src/config.py'den gelir. JSON dosyasında saklanan alpha/beta değerleri
        bilgilendirme amaçlıdır; burada Ausgrid'den gelen tek gerçek veri olan
        Q_max kullanılarak eğri baştan türetilir:

            alpha = ToU                                (q = 0'da ödeme isteği tavanı)
            beta  = -ln(FiT / ToU) / max(Q_MIN, Q_max) (q = Q_max'ta taban FiT'e iner)

        Böylece config'de FiT/ToU değiştiğinde JSON'ları yeniden üretmeye gerek kalmaz.
        Q_max bulunmayan (eski format) kayıtlarda JSON'daki alpha/beta aynen korunur.
        """
        tou = self.config.tou_price
        fit = self.config.fit_price
        beta_scale = -math.log(fit / tou)

        rebuilt = {}
        for time_key, params in raw_mb_data.items():
            q_max = params.get("Q_max")
            if q_max is None:
                rebuilt[time_key] = params
                continue
            safe_q_max = max(MB_Q_MAX_FLOOR, float(q_max))
            rebuilt[time_key] = {
                "alpha": tou,
                "beta": beta_scale / safe_q_max,
                "Q_max": float(q_max),
            }
        return rebuilt

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
        # Tüm parametreler DOĞRUDAN config'den okunur.
        # (Eskiden getattr(..., 0.4) şeklinde bir yedek varsayılan vardı; config'deki
        #  gerçek varsayılan 0.15 olduğu için yanıltıcıydı — kaldırıldı.)
        fit = self.config.fit_price
        tou = self.config.tou_price
        lam = self.config.lambda_uili
        return fit + (tou - fit) * math.exp(-lam * q)
    
    def decide_action(self, obs: Observation) -> Action:
        """Heuristic decision-making based on net energy position."""
        self._current_hour = obs.hour
        market_quantity = obs.inflexible_load  # positive=deficit, negative=surplus
        
        order = None
        if market_quantity > 0.01:
            # ALICI — makale Bölüm 3.3: rol yalnızca net enerji pozisyonuna bağlıdır
            # (Eq. 1). Üretici de gece/akşam net açık verdiğinde alıcı olabilir;
            # eskiden buradaki `agent_type != "producer"` kısıtı üreticinin alıcı
            # olmasını engelliyordu ve makaleyle çelişiyordu — kaldırıldı.
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
        grid_purchased: float = 0.0,
        grid_sold: float = 0.0,
    ):
        """Update internal state and tracking.

        grid_purchased : kWh bought from grid at ToU (buyer fallback)
        grid_sold      : kWh sold to grid at FiT    (seller fallback)
        """
        buy_cost    = sum(t.price * t.quantity for t in trades_as_buyer)
        sell_income = sum(t.price * t.quantity for t in trades_as_seller)
        bought_kwh  = sum(t.quantity for t in trades_as_buyer)
        sold_kwh    = sum(t.quantity for t in trades_as_seller)

        fit_price = self.config.fit_price
        tou_price = self.config.tou_price

        # Grid işlem maliyetleri
        grid_buy_cost    = grid_purchased * tou_price
        grid_sell_income = grid_sold * fit_price

        net_cost = (buy_cost + grid_buy_cost) - (sell_income + grid_sell_income)
        self.total_cost       += net_cost
        self.total_traded_kwh += bought_kwh + sold_kwh
        self.unmet_demand     += unmet_demand
        self.curtailed_energy += curtailed_surplus

        self.hourly_log.append({
            "hour":           hour,
            "buy_cost":       buy_cost,
            "sell_income":    sell_income,
            "net_cost":       net_cost,
            "bought_kwh":     bought_kwh,
            "sold_kwh":       sold_kwh,
            "unmet_demand":   unmet_demand,
            "curtailed":      curtailed_surplus,
            "grid_purchased": grid_purchased,
            "grid_sold":      grid_sold,
            "grid_buy_cost":  grid_buy_cost,
            "grid_sell_income": grid_sell_income,
        })
    
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

        FiT ve β değerleri doğrudan src/config.py'den okunur.
        """
        if not self.hourly_log:
            return 0.0

        last = self.hourly_log[-1]
        fit_price = self.config.fit_price
        beta      = self.config.reward_beta

        # Satıcı mı?
        # net_load negatif → sold_kwh > 0 veya curtailed > 0
        sold_kwh    = last.get("sold_kwh", 0.0)
        curtailed   = last.get("curtailed", 0.0)
        sell_income = last.get("sell_income", 0.0)

        grid_sold      = last.get("grid_sold", 0.0)
        grid_sell_inc  = last.get("grid_sell_income", 0.0)
        is_seller = (sold_kwh + curtailed + grid_sold) > 1e-6

        if is_seller:
            # Grid fallback varsa: eşleşemeyen enerji → grid'e FiT'ten satıldı.
            # P2P geliri > FiT geliri → ajan P2P'yi tercih etmeyi öğrenir.
            if self.config.grid_fallback:
                # Satıcı: P2P geliri + grid FiT geliri
                reward = sell_income + grid_sell_inc
            else:
                unmatched_reward = curtailed * self.config.reward_curtail_rate
                reward           = sell_income + unmatched_reward

            # β fiyat primi: FiT üzerinde P2P'de satılan her kWh için ekstra ödül
            # Ajan "daha yüksek fiyattan eşleş" sinyali alır.
            if beta > 0.0 and sold_kwh > 1e-6:
                price_premium = sell_income - sold_kwh * fit_price
                reward += beta * price_premium
        else:
            # Alıcı: tüm maliyet (P2P + grid ToU) minimize edilir.
            # net_cost zaten grid maliyetini içeriyor (process_trade_results'da eklendi).
            reward = -last["net_cost"]

        return reward
    
    def __repr__(self) -> str:
        return f"EnergyAgent(id={self.agent_id}, type={self.agent_type}, cost={self.total_cost:.3f}$)"