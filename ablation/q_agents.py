"""
Q-Learning varyantları (torch İÇE AKTARILMAZ).

q_base   : src QLearningAgent AYNEN (durum = (saat_kovası, miktar_kovası), 3×4 = 12).
q_priceK : durum sonuna önceki slotun ortalama takas fiyatı kovası eklenir:
               (saat_kovası, miktar_kovası, fiyat_kovası)  → 12 × (K+1) durum
           fiyat_kovası yalnızca obs.last_avg_price'tan hesaplanır (NN'nin gördüğü
           alanın aynısı), bkz. price_bucket().

obs.last_avg_price NE ZAMAN 0?  (src/market.py, src/environment.py)
  * Günün ilk slotu: EnergyTradingEnv.reset() her gün yeni bir
    PartialMatchDoubleAuction kurar; clearing_results boş olduğundan
    get_market_info() last_avg_price = 0.0 döndürür.
  * Önceki slotta hiç takas olmadıysa: PartialClearingResult.compute_stats()
    average_price = 0.0 yazar ve bu değer bir sonraki gözleme aynen geçer
    (EnergyTradingEnv._get_all_observations).
  Takas olduğunda değer hacim ağırlıklı ortalama orta-nokta fiyatıdır; tüm
  öğrenen emirleri [FiT, ToU] aralığında olduğundan pratikte bu aralıktadır
  (yine de kenar kovalara kırpılır).

Aksiyon uzayı, ödül, α, γ, ε takvimi ve güncelleme kuralı src ile birebir aynıdır:
decide_action / update_q / end_episode miras alınır. update_q bootstrap için
self._encode_state(next_obs) / self._encode_buyer_state(next_obs) çağırdığından
genişletilmiş durum hem seçimde hem TD hedefinde otomatik kullanılır; q_table[s]
(s 3'lü demet) ve argmax(axis=-1) özetleri ek boyutla da çalışır.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from src.config import SimConfig
from src.ql_agent import QLearningAgent, N_ACTIONS, N_HOUR_BUCKETS, N_PV_BUCKETS, N_DEMAND_BUCKETS
from src.ql_environment import QLearningEnv


def price_bucket(price: float, fit: float, tou: float, n_buckets: int,
                 edges: Optional[np.ndarray] = None) -> int:
    """0 = önceki slotta takas yok (price <= 0); 1..n = [FiT, ToU] üzerinde eşit genişlikli kovalar.

    Sınırlar e_j = linspace(FiT, ToU, n+1). Kova k (1..n) = [e_{k-1}, e_k); ToU son kovaya
    dahildir. FiT altı → 1, ToU üstü → n (kırpma). Tam sınırdaki değer üst kovaya gider
    (searchsorted side='right'; kayan nokta yuvarlamasından bağımsız, sınırlar
    bucket_edges() ile aynı dizidir).
    """
    if not (price > 0.0):
        return 0
    if edges is None:
        edges = bucket_edges(fit, tou, n_buckets)
    return 1 + int(np.searchsorted(edges[1:-1], price, side="right"))


def bucket_edges(fit: float, tou: float, n_buckets: int) -> np.ndarray:
    """Kova sınırları (n+1 değer), raporlama için."""
    return np.linspace(fit, tou, n_buckets + 1)


class PriceQLearningAgent(QLearningAgent):
    """src QLearningAgent + son takas fiyatı kovası (n_price_buckets + 1 değer)."""

    def __init__(self, *args, n_price_buckets: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_price_buckets = int(n_price_buckets)
        n_p = self.n_price_buckets + 1
        # src ile aynı dtype ve sıfır başlangıç; yalnızca ek boyut
        self.q_table = np.zeros((N_HOUR_BUCKETS, N_PV_BUCKETS, n_p, N_ACTIONS), dtype=np.float64)
        self.buy_q_table = np.zeros((N_HOUR_BUCKETS, N_DEMAND_BUCKETS, n_p, N_ACTIONS), dtype=np.float64)
        # Teşhis: karar anındaki fiyat kovası ziyaret sayıları (RNG tüketmez)
        self.price_visits = np.zeros(n_p, dtype=np.int64)
        self._price_edges = bucket_edges(self.fit_price, self.tou_price, self.n_price_buckets)

    def _pb(self, obs) -> int:
        return price_bucket(obs.last_avg_price, self.fit_price, self.tou_price, self.n_price_buckets,
                            self._price_edges)

    def _encode_state(self, obs) -> Tuple[int, int, int]:
        s0, s1 = super()._encode_state(obs)
        return s0, s1, self._pb(obs)

    def _encode_buyer_state(self, obs) -> Tuple[int, int, int]:
        s0, s1 = super()._encode_buyer_state(obs)
        return s0, s1, self._pb(obs)

    def decide_action(self, obs):
        if abs(obs.inflexible_load) > 0.01:   # yalnızca emir verilen (durum kodlanan) adımlar
            self.price_visits[self._pb(obs)] += 1
        return super().decide_action(obs)


def n_q_states(price_buckets: int) -> int:
    """Rol başına (satıcı / alıcı tablosu) durum sayısı."""
    return N_HOUR_BUCKETS * N_PV_BUCKETS * (price_buckets + 1 if price_buckets else 1)


def _agent_type(i: int, config: SimConfig) -> str:
    return "producer" if i < config.n_producers else "consumer"


class AblationQLEnv(QLearningEnv):
    """QLearningEnv + varyant ajanları.

    Ajanlar kurulumda (id sırasıyla) önceden oluşturulur; src'nin reset()'i onları
    her gün (ilk gün dahil) "mevcut ajan" dalından geçirir — yani src ile aynı yol.
    QLearningAgent kurulumu RNG kullanmaz.
    price_buckets = 0 → src QLearningAgent'ın kendisi (q_base).
    """

    def __init__(self, config: SimConfig, price_buckets: int = 0, **ql_kwargs):
        super().__init__(config, **ql_kwargs)
        self.price_buckets = int(price_buckets)
        self._eval_visit_snapshot: Optional[np.ndarray] = None
        for i in range(config.n_agents):
            common = dict(agent_id=i, agent_type=_agent_type(i, config),
                          config=config, alpha_lr=self.ql_alpha_lr,
                          gamma=self.ql_gamma, epsilon=self.ql_epsilon,
                          epsilon_min=self.ql_epsilon_min, epsilon_decay=self.ql_epsilon_decay)
            if self.price_buckets:
                self._ql_agents[i] = PriceQLearningAgent(n_price_buckets=self.price_buckets, **common)
            else:
                self._ql_agents[i] = QLearningAgent(**common)

    def total_price_visits(self) -> Optional[np.ndarray]:
        if not self.price_buckets:
            return None
        return sum(a.price_visits for a in self._ql_agents.values())

    def run_evaluation(self, seed=None, verbose=True):
        # Değerlendirme öncesi ziyaret sayıları (eğitim / değerlendirme ayrımı için)
        v = self.total_price_visits()
        self._eval_visit_snapshot = None if v is None else v.copy()
        return super().run_evaluation(seed=seed, verbose=verbose)

    def diagnostics(self) -> Dict:
        """Q-tablo doluluğu: en az bir aksiyonu sıfırdan farklı olan durum oranı."""
        def visited(t):
            return int(np.count_nonzero(np.any(t != 0.0, axis=-1)))
        sell = [visited(a.q_table) for a in self._ql_agents.values()]
        buy = [visited(a.buy_q_table) for a in self._ql_agents.values()]
        a0 = self._ql_agents[0]
        out = {
            "q_table_shape_sell": list(a0.q_table.shape),
            "q_table_shape_buy": list(a0.buy_q_table.shape),
            "states_per_table": int(np.prod(a0.q_table.shape[:-1])),
            "visited_states_sell_mean": float(np.mean(sell)),
            "visited_states_buy_mean": float(np.mean(buy)),
            "final_epsilon": float(a0.epsilon),
        }
        tot = self.total_price_visits()
        if tot is not None:
            snap = self._eval_visit_snapshot if self._eval_visit_snapshot is not None else tot
            out["price_bucket_visits_train"] = [int(x) for x in snap]
            out["price_bucket_visits_eval"] = [int(x) for x in (tot - snap)]
            out["price_bucket_edges"] = [float(x) for x in bucket_edges(
                self.config.fit_price, self.config.tou_price, self.price_buckets)]
        return out
