"""
Ortak parçalar: rejimler, rejim uygulayan ortamlar, ajan varyantları, gün koşucusu.

Rejim (d, s):
    d → her TÜKETİCİ ajanın (indeks >= n_producers) yük profili × d
    s → her ÜRETİCİ ajanın (indeks <  n_producers) PV profili   × s
Profiller src'nin ProfileManager'ı tarafından her zamanki gibi (aynı RNG çekiliş
sırasıyla) üretilir; çarpanlar ÜRETİMDEN SONRA, günün ilk gözleminden ÖNCE
uygulanır (bkz. RegimeMixin.reset). src/config.py'deki demand_multiplier /
supply_multiplier alanları KULLANILMAZ (bu paket onların var olmasına bağlı değil).

Bu modül torch'u içe aktarmaz; NN parçaları shock_lab.nn_agents içindedir.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# src/agent.py MB eğrisi dosyalarını GÖRELİ yoldan ("src/data/...") okur; bu yüzden
# her giriş noktası çalışma dizinini repo köküne sabitler.
def chdir_repo_root() -> None:
    os.chdir(REPO_ROOT)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)


from src.config import SimConfig                         # noqa: E402
from src.ql_agent import QLearningAgent, N_ACTIONS, N_HOUR_BUCKETS, N_PV_BUCKETS, N_DEMAND_BUCKETS  # noqa: E402
from src.ql_environment import QLearningEnv              # noqa: E402
from src.random_environment import community_balance     # noqa: E402

Regime = Tuple[float, float]

# ---------------------------------------------------------------------------
# Rejimler
# ---------------------------------------------------------------------------

SEEN_LEVELS = (0.5, 1.0, 2.0)
DEFAULT_TRAIN_REGIMES: List[Regime] = [(d, s) for d in SEEN_LEVELS for s in SEEN_LEVELS]
NORMAL: Regime = (1.0, 1.0)
DEFAULT_NORMAL_PROB = 1.0 / 3.0
DEFAULT_TEST_REGIMES: List[Regime] = DEFAULT_TRAIN_REGIMES + [
    (0.75, 1.0), (1.5, 1.0), (1.0, 0.75), (1.0, 1.5),        # görülmemiş tek faktör
    (1.5, 0.75), (0.75, 1.5), (1.5, 1.5), (0.75, 0.75),      # görülmemiş kombinasyon
]
# Zaman çizelgesi testi: (ad, rejim)
DEFAULT_TIMELINE_SHOCKS: List[Tuple[str, Regime]] = [
    ("demand_x2", (2.0, 1.0)),
    ("supply_x2", (1.0, 2.0)),
    ("demand_x1.5", (1.5, 1.0)),
    ("supply_x0.75", (1.0, 0.75)),
]

# Test günü seed'leri: tüm eğitim seed'lerinden AYRIK (eğitim: seed + episode < 10^6)
TEST_SEED_BASE = 10_000_000
TIMELINE_SEED_BASE = 20_000_000
MAX_TRAIN_DAY_SEED = 1_000_000   # train.py bunun altında kalmayı doğrular

BALANCE_REL_TOL = 1e-6


def rkey(r: Regime) -> str:
    """Rejim anahtarı: 'd=2:s=1' gibi."""
    return f"d={r[0]:g}:s={r[1]:g}"


def parse_regimes(text: str) -> List[Regime]:
    """'0.5:1,2:1,1:0.75' → [(0.5,1.0),(2.0,1.0),(1.0,0.75)]"""
    out: List[Regime] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        a, b = tok.split(":")
        out.append((float(a), float(b)))
    if not out:
        raise ValueError("boş rejim listesi")
    return out


def regime_probs(regimes: Sequence[Regime], normal_prob: Optional[float],
                 weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """Eğitim rejimlerinin örnekleme olasılıkları.

    weights verilirse normalize edilip kullanılır. Yoksa: (1,1) listede ise
    olasılığı normal_prob, kalan kütle diğer rejimlere eşit dağıtılır; tek rejim
    varsa olasılık 1.
    """
    n = len(regimes)
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if len(w) != n or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("--regime-weights rejim sayısıyla eşleşmeli ve pozitif olmalı")
        return w / w.sum()
    if n == 1:
        return np.ones(1)
    if NORMAL in regimes and normal_prob is not None:
        p = np.full(n, (1.0 - normal_prob) / (n - 1))
        p[regimes.index(NORMAL)] = normal_prob
        return p
    return np.full(n, 1.0 / n)


def nearest_level(x: float, levels: Sequence[float]) -> int:
    """Log ölçeğinde en yakın seviyenin indeksi (eşitlikte küçük olan)."""
    lx = math.log(x)
    dists = [abs(lx - math.log(l)) for l in levels]
    return int(np.argmin(dists))


def apply_regime(profiles, n_producers: int, regime: Regime) -> None:
    """ProfileManager'daki günlük profilleri yerinde ölçekler (RNG tüketmez)."""
    d, s = regime
    if d != 1.0:
        for i in range(n_producers, profiles.n_agents):
            profiles.load_profiles[i] = profiles.load_profiles[i] * d
    if s != 1.0:
        for i in range(n_producers):
            profiles.pv_profiles[i] = profiles.pv_profiles[i] * s


# ---------------------------------------------------------------------------
# Çıktı / log yardımcıları
# ---------------------------------------------------------------------------

class _NullWriter:
    def write(self, s: str) -> int:
        return len(s)

    def flush(self) -> None:
        pass


_OUT = sys.stdout


def log(msg: str) -> None:
    _OUT.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    _OUT.flush()


def silenced():
    """src ortamı her adımda çok fazla çıktı basar; bunu atar."""
    return contextlib.redirect_stdout(_NullWriter())


def write_json_atomic(data: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def fmt_hms(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


# ---------------------------------------------------------------------------
# Keşif takvimi
# ---------------------------------------------------------------------------

def derived_decay(start: float, floor: float, n_episodes: int, frac: float) -> float:
    """Taban değere eğitimin `frac` kesrinde ulaşan çarpımsal azalma:
        decay = (floor / start) ** (1 / (frac · N))
    """
    n = max(1.0, frac * n_episodes)
    return float((floor / start) ** (1.0 / n))


# ---------------------------------------------------------------------------
# Varyantlar
# ---------------------------------------------------------------------------

VARIANTS: Dict[str, Dict] = {
    "q_normal":  {"kind": "q",  "features": (),          "randomized": False},
    "q_blind":   {"kind": "q",  "features": (),          "randomized": True},
    "q_price":   {"kind": "q",  "features": ("price",),  "randomized": True},
    "q_oracle":  {"kind": "q",  "features": ("regime",), "randomized": True},
    "nn_normal": {"kind": "nn", "oracle": False,         "randomized": False},
    "nn_price":  {"kind": "nn", "oracle": False,         "randomized": True},
    "nn_oracle": {"kind": "nn", "oracle": True,          "randomized": True},
}
VARIANT_ORDER = list(VARIANTS)

DEFAULT_PRICE_THRESHOLDS = (0.135, 0.185)


def price_bucket(price: Optional[float], thresholds: Sequence[float]) -> int:
    """None (bugün henüz takas yok) → orta kova; yoksa eşiklere göre 0..len(thr)."""
    if price is None:
        return len(thresholds) // 2
    return int(np.searchsorted(np.asarray(thresholds), price, side="right"))


class ExtQLearningAgent(QLearningAgent):
    """src QLearningAgent + ek ayrık durum boyutları.

    Orijinal durum (saat_kovası, miktar_kovası) AYNEN korunur; sonuna eklenir:
      'price'  : önceki slotun ortalama takas fiyatı kovası (3 kova). Önceki
                 slotta takas olmadıysa (last_avg_price == 0) BUGÜN görülen son
                 pozitif takas fiyatı; bugün henüz hiç takas yoksa orta kova.
      'regime' : (d seviyesi, s seviyesi) — eğitim seviyelerinin log ölçeğinde
                 en yakını (görülmemiş değerler de en yakına eşlenir).
    Aksiyon uzayı, ödül, α, γ, güncelleme kuralı src ile birebir aynıdır
    (update_q / decide_action miras alınır; yalnızca durum kodlaması genişler).
    """

    def __init__(self, *args, features: Sequence[str] = (),
                 price_thresholds: Sequence[float] = DEFAULT_PRICE_THRESHOLDS,
                 d_levels: Sequence[float] = SEEN_LEVELS,
                 s_levels: Sequence[float] = SEEN_LEVELS, **kwargs):
        super().__init__(*args, **kwargs)
        self.features = tuple(features)
        self.price_thresholds = tuple(float(x) for x in price_thresholds)
        self.d_levels = tuple(float(x) for x in d_levels)
        self.s_levels = tuple(float(x) for x in s_levels)
        extra: List[int] = []
        for f in self.features:
            if f == "price":
                extra.append(len(self.price_thresholds) + 1)
            elif f == "regime":
                extra += [len(self.d_levels), len(self.s_levels)]
            else:
                raise ValueError(f"bilinmeyen özellik: {f}")
        self.extra_dims = tuple(extra)
        self.q_table = np.zeros((N_HOUR_BUCKETS, N_PV_BUCKETS) + self.extra_dims + (N_ACTIONS,))
        self.buy_q_table = np.zeros((N_HOUR_BUCKETS, N_DEMAND_BUCKETS) + self.extra_dims + (N_ACTIONS,))
        self._day_last_price: Optional[float] = None
        self._regime_idx: Tuple[int, int] = (nearest_level(1.0, self.d_levels),
                                             nearest_level(1.0, self.s_levels))

    def start_day(self, regime: Regime) -> None:
        self._day_last_price = None
        self._regime_idx = (nearest_level(regime[0], self.d_levels),
                            nearest_level(regime[1], self.s_levels))

    def _extra(self, obs) -> Tuple[int, ...]:
        out: Tuple[int, ...] = ()
        for f in self.features:
            if f == "price":
                p = obs.last_avg_price if obs.last_avg_price > 0.0 else self._day_last_price
                out += (price_bucket(p, self.price_thresholds),)
            else:
                out += self._regime_idx
        return out

    def _encode_state(self, obs):
        return tuple(super()._encode_state(obs)) + self._extra(obs)

    def _encode_buyer_state(self, obs):
        return tuple(super()._encode_buyer_state(obs)) + self._extra(obs)

    def decide_action(self, obs):
        action = super().decide_action(obs)
        # İz takibi KODLAMADAN SONRA güncellenir. update_q(next_obs) aynı kuralı
        # kullandığından bootstrap durumu bir sonraki adımda seçilen durumla aynıdır.
        if obs.last_avg_price > 0.0:
            self._day_last_price = obs.last_avg_price
        return action


# ---------------------------------------------------------------------------
# Ortamlar
# ---------------------------------------------------------------------------

class RegimeMixin:
    """reset(): src gününü kurar, sonra rejimi profillere uygular.

    Üst sınıfın reset()'i günün ilk gözlem sözlüğünü ölçeklenmemiş profillerle
    hesaplayıp döndürür; bu sözlük HİÇBİR ajana verilmez (src'nin eğitim/değerlendirme
    döngüleri de onu kullanmaz, step() gözlemleri kendisi yeniden hesaplar). Burada
    atılır ve ölçeklenmiş profillerden yeniden hesaplanır (_get_all_observations
    yan etkisizdir). Ajanların gördüğü her gözlem ölçeklenmiş profillerdendir.
    """

    regime: Regime = NORMAL

    def set_regime(self, regime: Regime) -> None:
        self.regime = (float(regime[0]), float(regime[1]))

    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)
        apply_regime(self.profiles, self.config.n_producers, self.regime)
        for a in self.agents:
            if hasattr(a, "start_day"):
                a.start_day(self.regime)
        return self._get_all_observations()


def _agent_type(i: int, config: SimConfig) -> str:
    return "producer" if i < config.n_producers else "consumer"


def _base_params(i: int, config: SimConfig):
    if i < config.n_producers:
        return config.producer_mb, config.producer_mc
    return config.consumer_mb, config.consumer_mc


class ShockQLEnv(RegimeMixin, QLearningEnv):
    """QLearningEnv + rejim. Ajanlar kurulumda (tüm id'ler için, sırayla) oluşturulur;
    src'nin reset()'i onları "mevcut ajan" dalından geçirip mb/mc parametrelerini her
    gün yeniler — yani src ile aynı yol. QLearningAgent kurulumu RNG kullanmaz.
    """

    def __init__(self, config: SimConfig, features: Sequence[str] = (),
                 price_thresholds: Sequence[float] = DEFAULT_PRICE_THRESHOLDS,
                 d_levels: Sequence[float] = SEEN_LEVELS, s_levels: Sequence[float] = SEEN_LEVELS,
                 **ql_kwargs):
        super().__init__(config, **ql_kwargs)
        for i in range(config.n_agents):
            mb, mc = _base_params(i, config)
            common = dict(agent_id=i, agent_type=_agent_type(i, config), mb_params=mb,
                          mc_params=mc, config=config, alpha_lr=self.ql_alpha_lr,
                          gamma=self.ql_gamma, epsilon=self.ql_epsilon,
                          epsilon_min=self.ql_epsilon_min, epsilon_decay=self.ql_epsilon_decay)
            if features:
                ag = ExtQLearningAgent(features=features, price_thresholds=price_thresholds,
                                       d_levels=d_levels, s_levels=s_levels, **common)
            else:
                ag = QLearningAgent(**common)
            self._ql_agents[i] = ag

    def exploration(self) -> float:
        return float(self._ql_agents[0].epsilon)

    def set_greedy(self) -> None:
        self.training_enabled = False
        for a in self._ql_agents.values():
            a.epsilon = 0.0

    def policy_arrays(self) -> Dict[str, np.ndarray]:
        out = {}
        for i, a in self._ql_agents.items():
            out[f"sell_{i}"] = a.q_table
            out[f"buy_{i}"] = a.buy_q_table
        return out

    def load_policy_arrays(self, arrs) -> None:
        for i, a in self._ql_agents.items():
            sell, buy = arrs[f"sell_{i}"], arrs[f"buy_{i}"]
            if sell.shape != a.q_table.shape or buy.shape != a.buy_q_table.shape:
                raise ValueError(f"Q-tablo şekli uyuşmuyor (ajan {i}): {sell.shape} vs {a.q_table.shape}")
            a.q_table = np.array(sell, dtype=np.float64)
            a.buy_q_table = np.array(buy, dtype=np.float64)


# ---------------------------------------------------------------------------
# Gün koşucusu + metrikler
# ---------------------------------------------------------------------------

DAY_KEYS = ["avg_price", "p2p_kwh", "e_max_kwh", "eta_pct", "grid_buy_kwh", "grid_sell_kwh",
            "seller_revenue", "buyer_cost", "demand_kwh", "supply_kwh"]


def run_day(env, seed: int, regime: Regime, learn: bool) -> Tuple[Dict[str, float], bool]:
    """Bir simülasyon günü: reset → 48 adım → (öğreniyorsa) end_episode.

    Döner: (gün metrikleri, enerji dengesi ihlali var mı)
    """
    env.set_regime(regime)
    with silenced():
        env.reset(seed=seed)
        bal = community_balance(env.profiles, env.config)   # RNG tüketmez
        done = False
        while not done:
            _, _, done, _ = env.step()
        if learn:
            env.end_episode()
    m = env.metrics
    n_p = env.config.n_producers
    p2p = float(m.total_internal_trading())
    gb = float(m.total_grid_purchased())
    gs = float(m.total_grid_sold())
    D, S, E = float(bal["deficit_kwh"]), float(bal["surplus_kwh"]), float(bal["e_max_kwh"])
    dev_d, dev_s = abs(D - p2p - gb), abs(S - p2p - gs)
    viol = dev_d > BALANCE_REL_TOL * max(1.0, D) or dev_s > BALANCE_REL_TOL * max(1.0, S)
    out = {
        "avg_price": float(m.average_clearing_price()),
        "p2p_kwh": p2p,
        "e_max_kwh": E,
        "eta_pct": p2p / E * 100.0 if E > 0 else 0.0,
        "grid_buy_kwh": gb,
        "grid_sell_kwh": gs,
        "seller_revenue": float(m.total_seller_revenue(n_p)),
        "buyer_cost": float(m.total_buyer_cost(n_p)),
        "demand_kwh": D,
        "supply_kwh": S,
        "_dev_d": dev_d,
        "_dev_s": dev_s,
    }
    return out, viol


class BalanceTracker:
    def __init__(self):
        self.n_days = 0
        self.n_viol = 0
        self.max_dev_d = 0.0
        self.max_dev_s = 0.0

    def add(self, day: Dict[str, float], viol: bool) -> None:
        self.n_days += 1
        self.n_viol += int(viol)
        self.max_dev_d = max(self.max_dev_d, day["_dev_d"])
        self.max_dev_s = max(self.max_dev_s, day["_dev_s"])

    def merge(self, other: Dict) -> None:
        self.n_days += other["n_days"]
        self.n_viol += other["n_violations"]
        self.max_dev_d = max(self.max_dev_d, other["max_abs_D_minus_p2p_minus_gridbuy"])
        self.max_dev_s = max(self.max_dev_s, other["max_abs_S_minus_p2p_minus_gridsell"])

    def as_dict(self) -> Dict:
        return {"n_days": self.n_days, "n_violations": self.n_viol, "rel_tol": BALANCE_REL_TOL,
                "max_abs_D_minus_p2p_minus_gridbuy": self.max_dev_d,
                "max_abs_S_minus_p2p_minus_gridsell": self.max_dev_s}
