"""
Enerji ajanları için ortak taban sınıf: EnergyAgent.

Qiu et al. POMG gözlem/aksiyon yapısı (Section 3.3):
1. Gözlem : net (inflexible) yük, önceki periyodun takas fiyatı, saat
2. Karar  : piyasa emri (fiyat, miktar) — rol net enerji pozisyonuna göre (Eq. 1)
3. Sonuç  : takas / grid fallback muhasebesi ve ödül (process_trade_results,
            get_reward)

Fiyatlama stratejisi (decide_action) alt sınıflardadır:
    RandomAgent     (src/random_agent.py)    — rastgele (zero-intelligence) baseline
    QLearningAgent  (src/ql_agent.py)        — tabular Q-Learning
    REINFORCEAgent  (src/reinforce_agent.py) — REINFORCE politika ağı

Faz 2 Reward (satıcı için):
    reward = matched_qty × clearing_price + unmatched_qty × FiT
Faz 2 Reward (alıcı için):
    reward = -net_cost  (alıcı maliyet minimize eder)
"""

from dataclasses import dataclass
from typing import Optional

from src.config import SimConfig
from src.market import Order


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
        config: SimConfig,
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        
        # Tracking
        self.total_cost = 0.0           # Cumulative energy cost ($)
        self.total_traded_kwh = 0.0     # Cumulative traded energy
        self.unmet_demand = 0.0         # Cumulative unmet demand (kWh)
        self.curtailed_energy = 0.0     # Cumulative curtailed surplus (kWh)
        self.hourly_log = []            # Detailed per-hour records

    def decide_action(self, obs: Observation) -> Action:
        """Fiyatlama stratejisi — alt sınıflar (Random / QL / REINFORCE) uygular."""
        raise NotImplementedError

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