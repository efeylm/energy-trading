"""
Tabular Q-Learning Agent for P2P Energy Trading — Faz 2.

Hocanın Faz 2 spesifikasyonuna göre:

State  : (saat_bucket, pv_bucket)
         - saat_bucket : 3 ToU dönemi (gece/gün/akşam-tepe)
         - pv_bucket   : 4 PV üretim seviyesi (hiç / az / orta / yüksek)
         → 3 × 4 = 12 ayrık durum

Action : FiT ile ToU arasında 7 ayrık fiyat seviyesi
         price = FiT + a_idx / (N_ACTIONS-1) * (ToU - FiT)

Reward : matched_qty × clearing_price + unmatched_qty × FiT
         (Satıcı için: sattığı kWh × anlık fiyat + satamadığı kWh × taban FiT)

Update : Q(s,a) ← Q(s,a) + α [r + γ·max Q(s',·) − Q(s,a)]
         Sadece satıcı konumundayken (net_load < 0) öğrenilir.

Bu modül yalnızca Q-Learning mantığını içerir.
Baseline karşılaştırması için baseline_agent.py dosyasına bakınız.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from src.config import SimConfig, AgentMBParams, AgentMCParams
from src.agent import EnergyAgent, Observation, Action
from src.market import Order


# ---------------------------------------------------------------------------
# State-space definition — Faz 2 spesifikasyonu
# ---------------------------------------------------------------------------

# Saat dilimleri (ToU dönemleri): 3 kova
# 0 = Gece/off-peak  (0–8, 20–24 → step 0–16, 40–47)
# 1 = Güneş/mid      (8–17 → step 16–34)
# 2 = Akşam tepe     (17–20 → step 34–40)
def _hour_bucket(hour_step: int) -> int:
    """
    48 adımlık simülasyonu (her adım 0.5h) ToU dönemlerine ayırır.
    0 = gece/off-peak, 1 = gündüz (güneş), 2 = akşam tepe
    """
    real_hour = (hour_step * 0.5) % 24
    if 8.0 <= real_hour < 17.0:
        return 1  # Gündüz / güneş saatleri
    elif 17.0 <= real_hour < 20.0:
        return 2  # Akşam tepe yükü
    else:
        return 0  # Gece / off-peak


# PV üretim seviyeleri: 4 kova
# Negatif net_load = fazla üretim (PV fazlası)
PV_EDGES = np.array([0.0, 1.0, 3.0, 6.0, 20.0])   # 4 bin: [0,1), [1,3), [3,6), [6,+)

def _pv_bucket(surplus_kwh: float) -> int:
    """
    PV fazlası (kWh) değerini 4 kovaya ayırır.
    surplus_kwh = abs(inflexible_load) when net_load < 0
    """
    return int(np.clip(np.searchsorted(PV_EDGES, surplus_kwh, side="right") - 1, 0, len(PV_EDGES) - 2))


N_HOUR_BUCKETS = 3   # ToU dönemi sayısı
N_PV_BUCKETS   = 4   # PV seviye sayısı
# Toplam durum sayısı: 3 × 4 = 12

# ---------------------------------------------------------------------------
# Action space — FiT ile ToU arasında ayrık fiyat seviyeleri
# ---------------------------------------------------------------------------
# Fiyatlar runtime'da config'den hesaplanır; burada sadece indeks sayısı sabit.
N_ACTIONS = 7   # Hocanın "5–10 ayrık" tavsiyesinin ortası


def _action_prices(fit: float, tou: float) -> np.ndarray:
    """FiT ile ToU arasında N_ACTIONS eşit aralıklı fiyat dizisi döner."""
    return np.linspace(fit, tou, N_ACTIONS)


# ---------------------------------------------------------------------------
# Q-Learning Agent
# ---------------------------------------------------------------------------

class QLearningAgent(EnergyAgent):
    """
    Tabular Q-Learning enerji ticareti ajansı — Faz 2.

    State  : (saat_bucket, pv_bucket) — 3 × 4 = 12 durum
    Action : FiT–ToU arasında 7 fiyat seviyesi
    Reward : matched_qty × clearing_price + unmatched_qty × FiT

    Sadece satıcı konumundayken (net_load < 0) Q-tablosu güncellenir.
    Alıcı konumundayken (net_load > 0) heuristic MB fiyatı kullanılır.
    """

    def __init__(
        self,
        agent_id: int,
        agent_type: str,
        mb_params: AgentMBParams,
        mc_params: AgentMCParams,
        config: SimConfig,
        alpha_lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.97,
    ):
        super().__init__(agent_id, agent_type, mb_params, mc_params, config)

        self.alpha_lr      = alpha_lr
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Tariff referans fiyatları (config'den)
        self.fit_price: float = config.fit_price
        self.tou_price: float = config.tou_price

        # Q-table: shape (N_HOUR_BUCKETS, N_PV_BUCKETS, N_ACTIONS)
        # Başlangıçta sıfır; optimistik başlangıç gerekmiyor (reward pozitif)
        self.q_table: np.ndarray = np.zeros(
            (N_HOUR_BUCKETS, N_PV_BUCKETS, N_ACTIONS), dtype=np.float64
        )

        # Tek adım geçiş belleği
        self._prev_state: Optional[Tuple[int, int]] = None
        self._prev_action_idx: Optional[int] = None
        self._prev_obs: Optional[Observation] = None

        # Episode istatistikleri
        self.episode_rewards: list[float] = []
        self._episode_reward_acc: float = 0.0

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------

    def _encode_state(self, obs: Observation) -> Tuple[int, int]:
        """Sürekli gözlemi (saat_bucket, pv_bucket) çiftine dönüştürür."""
        s0 = _hour_bucket(obs.hour)
        surplus = max(0.0, -obs.inflexible_load)   # net_load < 0 → PV fazlası
        s1 = _pv_bucket(surplus)
        return s0, s1

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def _select_action(self, state: Tuple[int, int]) -> int:
        """ε-greedy aksiyon seçimi."""
        if np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        return int(np.argmax(self.q_table[state]))

    def _ask_price_from_action(self, action_idx: int) -> float:
        """Aksiyon indeksini $/kWh satış fiyatına çevirir."""
        prices = _action_prices(self.fit_price, self.tou_price)
        return float(prices[action_idx])

    # ------------------------------------------------------------------
    # Core overrides
    # ------------------------------------------------------------------

    def decide_action(self, obs: Observation) -> Action:
        """Q-tablosu kullanarak ε-greedy aksiyon seç."""
        self._current_hour = obs.hour
        state = self._encode_state(obs)
        action_idx = self._select_action(state)

        # Geçiş belleğine kaydet (Q-update için)
        self._prev_state = state
        self._prev_action_idx = action_idx
        self._prev_obs = obs

        net = obs.inflexible_load
        order = None

        if net > 0.01:
            # ALICI: Öğrenme yok — ham MB eğrisi fiyatı kullanılır (Faz 1 mantığı)
            bid_price = self.compute_mb(net, hour=obs.hour)
            order = Order(
                agent_id=self.agent_id,
                price=float(np.clip(bid_price, 0.001, 2.0)),
                quantity=net,
                is_buy=True,
                is_emergency=False,
            )
        elif net < -0.01:
            # SATICI: Q-tablosundan gelen fiyat seviyesini kullan
            ask_price = self._ask_price_from_action(action_idx)
            order = Order(
                agent_id=self.agent_id,
                price=float(np.clip(ask_price, 0.001, 2.0)),
                quantity=abs(net),
                is_buy=False,
                is_emergency=False,
            )

        return Action(order=order)

    def update_q(self, reward: float, next_obs: Optional[Observation]):
        """
        Q-öğrenme güncellemesi — reward gözlemlendikten sonra çağrılır.

        Faz 2 reward formülü:
            reward = matched_qty × clearing_price + unmatched_qty × FiT
        Bu hesaplama environment.py'de process_trade_results + get_reward
        üzerinden gelir; burada sadece Q-update yapılır.

        Sadece satıcı konumundayken (net_load < 0) tabloyu günceller.
        """
        if self._prev_state is None or self._prev_action_idx is None or self._prev_obs is None:
            return

        # Sadece satıcıyken öğren (Faz 2: "satıcı öğrenir")
        if self._prev_obs.inflexible_load > 0:
            return

        s = self._prev_state
        a = self._prev_action_idx

        # Sonraki durum için bootstrap
        if next_obs is not None:
            s_next = self._encode_state(next_obs)
            max_q_next = float(np.max(self.q_table[s_next]))
        else:
            max_q_next = 0.0

        # Q-update: Q(s,a) ← Q(s,a) + α [r + γ·max Q(s',·) − Q(s,a)]
        current_q  = self.q_table[s][a]
        td_target  = reward + self.gamma * max_q_next
        td_error   = td_target - current_q
        self.q_table[s][a] += self.alpha_lr * td_error

        self._episode_reward_acc += reward

    def end_episode(self):
        """Episode sonu: ε decay ve reward kaydı."""
        self.episode_rewards.append(self._episode_reward_acc)
        self._episode_reward_acc = 0.0
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Geçiş belleğini sıfırla
        self._prev_state      = None
        self._prev_action_idx = None
        self._prev_obs        = None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def q_table_stats(self) -> dict:
        """Q-tablosu için özet istatistikler döner."""
        return {
            "mean":    float(np.mean(self.q_table)),
            "std":     float(np.std(self.q_table)),
            "max":     float(np.max(self.q_table)),
            "min":     float(np.min(self.q_table)),
            "nonzero": int(np.count_nonzero(self.q_table)),
        }

    def greedy_policy_summary(self) -> np.ndarray:
        """Her durum için greedy aksiyon indeksini döner (2-D array: saat × PV)."""
        return np.argmax(self.q_table, axis=-1)

    def greedy_policy_prices(self) -> np.ndarray:
        """Her durum için greedy aksiyonun $/kWh fiyatını döner."""
        idx_grid = self.greedy_policy_summary()
        prices = _action_prices(self.fit_price, self.tou_price)
        return prices[idx_grid]

    def __repr__(self) -> str:
        return (
            f"QLearningAgent(id={self.agent_id}, type={self.agent_type}, "
            f"ε={self.epsilon:.3f}, cost={self.total_cost:.3f}$)"
        )