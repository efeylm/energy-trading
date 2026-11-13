"""
Rastgele (Zero-Intelligence) Baseline ortamı ve çoklu çekiliş koşucusu.

BaselineEnv (src/ql_environment.py) ile aynı desen: EnergyTradingEnv'i miras
alır, reset()'te ajanları RandomAgent ile değiştirir. Profiller, market ve
grid fallback birebir aynıdır; yalnızca fiyatlama rastgeledir.

Tek bir rastgele gün gürültülü olduğundan baseline, AYNI gün (aynı profil
seed'i) üzerinde n_draws farklı fiyat çekilişinin ORTALAMASI olarak raporlanır:

    price_seed = base_price_seed + k,   k = 0 .. n_draws-1

Her skaler metrik için çekiliş başına değerler ile ortalama / std (ddof=1) /
min / max tutulur. Ayrıca yalnızca profillerden hesaplanan üst sınır

    E_max = Σ_t min(S_t, D_t)      (S_t / D_t = topluluk fazlası / açığı)

ve eşleştirme verimliliği η = P2P / E_max × 100 hesaplanır.

Döndürülen RandomBaselineMetrics nesnesi bir MetricsCollector alt sınıfıdır;
comparison.py'de baseline'ın MetricsCollector'ı yerine doğrudan kullanılabilir
(tablolar, summary(), summary_welfare(), grafikler) ve çekiliş ortalamalı
değerler döndürür.

Kullanım:
    python3 -m src.random_environment --draws 200 --seed 1

NOT: Bu modül torch veya src.comparison İÇE AKTARMAZ — PyTorch kurulu
olmayan makinede de çalışmalıdır.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
from typing import Dict, List, Optional

import numpy as np

from src.config import SimConfig
from src.environment import EnergyTradingEnv
from src.metrics import MetricsCollector, HourlyMetrics
from src.random_agent import RandomAgent


class RandomBaselineEnv(EnergyTradingEnv):
    """EnergyTradingEnv + RandomAgent factory."""

    def __init__(self, config: SimConfig, price_seed: int = 0):
        super().__init__(config)
        # Sadece rastgele fiyatları belirler; profil/gün seed'inden bağımsızdır.
        self.price_seed = price_seed

    def reset(self, seed: Optional[int] = None) -> Dict:
        obs = super().reset(seed=seed)

        # Parent EnergyAgent'ları RandomAgent ile değiştir
        new_agents = []
        for agent in self.agents:
            rnd_agent = RandomAgent(
                agent_id=agent.agent_id,
                agent_type=agent.agent_type,
                mb_params=agent.mb_params,
                mc_params=agent.mc_params,
                config=self.config,
                price_seed=self.price_seed,
            )
            new_agents.append(rnd_agent)
        self.agents = new_agents
        return obs

    def run_random(
        self,
        seed: Optional[int] = None,
        price_seed: int = 0,
        verbose: bool = True,
    ) -> MetricsCollector:
        """Rastgele ajanlarla tek bir simülasyon günü çalıştır.

        seed       : gün/profil seed'i (repo genelindeki anlamıyla aynı)
        price_seed : yalnızca rastgele fiyatları kontrol eder
        """
        run_seed = seed if seed is not None else self.config.seed
        self.price_seed = price_seed
        self.reset(seed=run_seed)

        if verbose:
            print("\n" + "=" * 60)
            print(f"  Random Baseline (Zero-Intelligence) — price_seed={price_seed}")
            print("=" * 60)

        done = False
        while not done:
            _, _, done, _ = self.step()

        if verbose:
            print()
            print(self.metrics.summary())
        return self.metrics

    def max_p2p_energy(self) -> float:
        """Profil tabanlı üst sınır E_max = Σ_t min(S_t, D_t) (kWh).

        Ajan rolleriyle aynı eşikler kullanılır: |net| <= 0.01 olan ajan emir
        vermediği için ne S_t'ye ne D_t'ye katılır.
        """
        return community_balance(self.profiles, self.config)["e_max_kwh"]


def community_balance(profiles, config: SimConfig) -> Dict[str, float]:
    """Bir günün profil tabanlı topluluk dengesi (fiyattan / stratejiden bağımsız).

        surplus_kwh = S = Σ_t S_t,   deficit_kwh = D = Σ_t D_t,
        e_max_kwh   = E_max = Σ_t min(S_t, D_t)

    RandomBaselineEnv.max_p2p_energy() ile aynı eşikler ve aynı toplama sırası.
    """
    e_max = 0.0
    s_tot = 0.0
    d_tot = 0.0
    for t in range(config.t_hours):
        surplus = 0.0
        deficit = 0.0
        for i in range(config.n_agents):
            net = profiles.get_inflexible_load(i, t)
            if net > 0.01:
                deficit += net
            elif net < -0.01:
                surplus += -net
        e_max += min(surplus, deficit)
        s_tot += surplus
        d_tot += deficit
    return {"surplus_kwh": s_tot, "deficit_kwh": d_tot, "e_max_kwh": e_max}


# ---------------------------------------------------------------------------
# Çekiliş ortalamalı sonuç nesnesi (MetricsCollector adaptörü)
# ---------------------------------------------------------------------------

# (anahtar, tablo etiketi, format) — comparison._print_comparison_3way satırları
# + E_max ve η.
METRIC_ROWS = [
    ("p2p_ratio_pct",      "P2P Orani (%)",                 "{:.2f}"),
    ("p2p_kwh",            "P2P Ticaret (kWh)",             "{:.2f}"),
    ("grid_purchased_kwh", "Grid Alim/ToU (kWh)",           "{:.2f}"),
    ("grid_sold_kwh",      "Grid Satim/FiT (kWh)",          "{:.2f}"),
    ("avg_price",          "Ort. P2P Fiyati ($/kWh)",       "{:.4f}"),
    ("seller_p2p_income",  "SATICI: P2P Geliri ($)",        "{:.2f}"),
    ("seller_grid_income", "SATICI: Grid FiT Geliri ($)",   "{:.2f}"),
    ("seller_revenue",     "SATICI: TOPLAM GELIR ($)",      "{:.2f}"),
    ("buyer_p2p_cost",     "ALICI: P2P Alim Maliyeti ($)",  "{:.2f}"),
    ("buyer_grid_cost",    "ALICI: Grid ToU Maliyeti ($)",  "{:.2f}"),
    ("buyer_cost",         "ALICI: TOPLAM MALIYET ($)",     "{:.2f}"),
    ("e_max_kwh",          "E_max = Σ min(S_t,D_t) (kWh)",  "{:.2f}"),
    ("efficiency_pct",     "Verimlilik η = P2P/E_max (%)",  "{:.2f}"),
]


def _draw_values(m: MetricsCollector, n_producers: int, e_max: float) -> Dict[str, float]:
    """Tek bir çekilişin skaler metrikleri (mevcut MetricsCollector metodlarıyla)."""
    p2p = m.total_internal_trading()
    return {
        "p2p_ratio_pct":      m.p2p_ratio() * 100,
        "p2p_kwh":            p2p,
        "grid_purchased_kwh": m.total_grid_purchased(),
        "grid_sold_kwh":      m.total_grid_sold(),
        "avg_price":          m.average_clearing_price(),
        "seller_p2p_income":  m.total_p2p_sell_income(n_producers),
        "seller_grid_income": m.total_grid_sell_income(n_producers),
        "seller_revenue":     m.total_seller_revenue(n_producers),
        "buyer_p2p_cost":     m.total_p2p_buy_cost(n_producers),
        "buyer_grid_cost":    m.total_grid_buy_cost(n_producers),
        "buyer_cost":         m.total_buyer_cost(n_producers),
        "e_max_kwh":          e_max,
        "efficiency_pct":     p2p / e_max * 100 if e_max > 0 else 0.0,
    }


class RandomBaselineMetrics(MetricsCollector):
    """n_draws rastgele çekilişin ortalaması — MetricsCollector yerine geçer.

    * Ajan bazlı sözlükler (agent_total_cost, agent_p2p_sell_income, ...) çekiliş
      ortalamasıdır; bu yüzden total_seller_revenue() vb. doğrusal toplamlar
      otomatik olarak çekiliş ortalamasını verir.
    * hourly: saat başına ortalama HourlyMetrics. average_price yalnızca o
      slotta işlem olan çekilişler üzerinden ortalanır (hiç yoksa 0 = "işlem yok").
    * p2p_ratio() ve average_clearing_price() doğrusal olmadığından çekiliş
      başına değerlerin ortalaması olarak döndürülür.
    * values[anahtar] → çekiliş başına değerler (np.ndarray), stats() → özet.
    """

    def __init__(
        self,
        config: SimConfig,
        draws: List[MetricsCollector],
        e_max: float,
        seed: int,
        base_price_seed: int,
    ):
        super().__init__(config.n_agents)
        self.n_draws = len(draws)
        self.seed = seed
        self.base_price_seed = base_price_seed
        self.e_max = e_max
        self.fit_price = config.fit_price
        self.tou_price = config.tou_price

        # Çekiliş başına skaler değerler
        per_draw = [_draw_values(m, config.n_producers, e_max) for m in draws]
        self.values: Dict[str, np.ndarray] = {
            key: np.array([d[key] for d in per_draw]) for key, _, _ in METRIC_ROWS
        }

        # Ajan bazlı sözlükler: çekiliş ortalaması
        for attr in (
            "agent_total_cost", "agent_total_bought", "agent_total_sold",
            "agent_total_unmet", "agent_total_curtailed",
            "agent_grid_purchased", "agent_grid_sold",
            "agent_p2p_sell_income", "agent_p2p_buy_cost",
            "agent_grid_sell_income", "agent_grid_buy_cost",
        ):
            setattr(self, attr, {
                i: float(np.mean([getattr(m, attr)[i] for m in draws]))
                for i in range(self.n_agents)
            })

        # Saatlik metrikler: çekiliş ortalaması
        n_hours = len(draws[0].hourly)
        for t in range(n_hours):
            hs = [m.hourly[t] for m in draws]
            traded_prices = [h.average_price for h in hs if h.total_traded_kwh > 0]
            self.hourly.append(HourlyMetrics(
                hour=t,
                average_price=float(np.mean(traded_prices)) if traded_prices else 0.0,
                total_traded_kwh=float(np.mean([h.total_traded_kwh for h in hs])),
                n_trades=float(np.mean([h.n_trades for h in hs])),
                total_unmet_demand=float(np.mean([h.total_unmet_demand for h in hs])),
                total_curtailed=float(np.mean([h.total_curtailed for h in hs])),
                n_buyers=float(np.mean([h.n_buyers for h in hs])),
                n_sellers=float(np.mean([h.n_sellers for h in hs])),
                total_grid_purchased=float(np.mean([h.total_grid_purchased for h in hs])),
                total_grid_sold=float(np.mean([h.total_grid_sold for h in hs])),
            ))

    # ---- Doğrusal olmayan metrikler: çekiliş başına değerlerin ortalaması ----

    def p2p_ratio(self) -> float:
        return float(np.mean(self.values["p2p_ratio_pct"])) / 100

    def average_clearing_price(self) -> float:
        return float(np.mean(self.values["avg_price"]))

    def efficiency(self) -> float:
        """Ortalama eşleştirme verimliliği η (%)."""
        return float(np.mean(self.values["efficiency_pct"]))

    # ---- Özet ----

    def stats(self, key: str) -> Dict[str, float]:
        """Bir metriğin çekilişler üzerinden ortalama / std (ddof=1) / min / max'ı."""
        v = self.values[key]
        return {
            "mean": float(np.mean(v)),
            "std":  float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "min":  float(np.min(v)),
            "max":  float(np.max(v)),
        }

    def label(self) -> str:
        return f"Random Baseline (ZI, {self.n_draws} çekiliş ort.)"

    def summary(self) -> str:
        return (
            f"  [{self.label()} — tüm değerler çekiliş ortalamasıdır]\n"
            + super().summary()
        )

    def format_table(self) -> str:
        """metrik | ortalama | std | min | max tablosu."""
        last = self.base_price_seed + self.n_draws - 1
        width = 84
        lines = [
            "=" * width,
            f"{'RANDOM (ZERO-INTELLIGENCE) BASELINE — ' + str(self.n_draws) + ' FİYAT ÇEKİLİŞİ':^{width}}",
            "=" * width,
            f"  Gün/profil seed : {self.seed}   |   price_seed : "
            f"{self.base_price_seed}..{last}",
            f"  Fiyat ~ U[FiT, ToU] = U[{self.fit_price:.2f}, {self.tou_price:.2f}] $/kWh, "
            f"ajan başına slot başına tek toplu emir",
            "-" * width,
            f"  {'Metrik':<30} {'Ortalama':>11} {'Std':>11} {'Min':>11} {'Max':>11}",
            "-" * width,
        ]
        for key, label, fmt in METRIC_ROWS:
            s = self.stats(key)
            lines.append(
                f"  {label:<30} "
                + " ".join(f"{fmt.format(s[k]):>11}" for k in ("mean", "std", "min", "max"))
            )
        lines += [
            "-" * width,
            "  * Std: örneklem standart sapması (ddof=1), çekilişler arası.",
            "  * E_max yalnızca profillere bağlıdır (çekilişten bağımsız).",
            "=" * width,
        ]
        return "\n".join(lines)


def run_random_baseline(
    config: SimConfig,
    n_draws: int = 200,
    seed: Optional[int] = None,
    base_price_seed: int = 0,
    verbose: bool = False,
) -> RandomBaselineMetrics:
    """Aynı gün üzerinde n_draws rastgele fiyat çekilişi çalıştır ve ortala.

    price_seed = base_price_seed + k, k = 0..n_draws-1. Global RNG'ler
    tüketilmez (bkz. src/random_agent.py).
    """
    if n_draws < 1:
        raise ValueError("n_draws en az 1 olmalıdır")
    day_seed = seed if seed is not None else config.seed

    env = RandomBaselineEnv(config)
    draws: List[MetricsCollector] = []
    e_max = 0.0
    for k in range(n_draws):
        # Ortam her adımda çok fazla çıktı basar; verbose değilse sustur.
        sink = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
        with sink:
            m = env.run_random(seed=day_seed, price_seed=base_price_seed + k, verbose=verbose)
        draws.append(m)
        if k == 0:
            e_max = env.max_p2p_energy()

    return RandomBaselineMetrics(config, draws, e_max, day_seed, base_price_seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Random (Zero-Intelligence) Baseline — çoklu fiyat çekilişi"
    )
    parser.add_argument("--draws", type=int, default=200,
                        help="Fiyat çekilişi sayısı (varsayılan: 200)")
    parser.add_argument("--seed", type=int, default=1,
                        help="Gün/profil seed'i (varsayılan: 1)")
    parser.add_argument("--price-seed", type=int, default=0,
                        help="İlk price_seed; çekiliş k → price-seed + k (varsayılan: 0)")
    parser.add_argument("--save_dir", type=str, default=".",
                        help="Log dosyasının kaydedileceği dizin")
    parser.add_argument("--log_name", type=str, default="random_baseline_summary.txt",
                        help="Log dosyası adı (varsayılan: random_baseline_summary.txt)")
    args = parser.parse_args()

    cfg = SimConfig(seed=args.seed)
    result = run_random_baseline(
        cfg, n_draws=args.draws, seed=args.seed, base_price_seed=args.price_seed,
    )
    table = result.format_table()
    print(table)

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, args.log_name)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\nLog dosyası oluşturuldu: {log_path}")
