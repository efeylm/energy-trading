"""
Q-Learning vs Baseline vs Deep-Q Karşılaştırma Modülü

Kullanım:
    python -m src.comparison                      # Q-Learning lambda sweep (varsayılan)
    python -m src.comparison --agent nn           # Deep-Q lambda sweep
    python -m src.comparison --agent all          # Baseline + Q-Learning + Deep-Q tek çalıştırma
    python -m src.comparison --episodes 200
    python -m src.comparison --agent all --lambdas 0.2

--agent seçenekleri:
  ql   : Baseline vs Q-Learning (lambda sweep) — mevcut davranış
  nn   : Baseline vs Deep-Q (lambda sweep)
  all  : Baseline + Q-Learning + Deep-Q tek lambda, yan yana karşılaştırma

Adımlar (ql / nn):
1. Baseline ajanları aynı seed ile tek bir günde çalıştır.
2. Seçilen ajan tipi n_episodes boyunca eğitilir.
3. Greedy (ε=0) modunda değerlendirilir.
4. Karşılaştırma metrikleri ve grafikler üretilir.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import SimConfig
from src.ql_environment import QLearningEnv, BaselineEnv
from src.metrics import MetricsCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_avg(arr: np.ndarray) -> float:
    valid = arr[arr > 0]
    return float(np.mean(valid)) if len(valid) > 0 else 0.0


def _print_comparison(bl_metrics: MetricsCollector, ql_metrics: MetricsCollector, file=None,
                      label_b: str = "Q-Learn"):
    """Konsola yan yana karşılaştırma tablosu yazdır."""
    bl_cost   = sum(bl_metrics.agent_total_cost.values())
    ql_cost   = sum(ql_metrics.agent_total_cost.values())
    bl_trade  = bl_metrics.total_internal_trading()
    ql_trade  = ql_metrics.total_internal_trading()
    bl_unmet  = bl_metrics.total_unmet_demand()
    ql_unmet  = ql_metrics.total_unmet_demand()
    bl_curt   = bl_metrics.total_curtailment()
    ql_curt   = ql_metrics.total_curtailment()
    bl_price  = bl_metrics.average_clearing_price()
    ql_price  = ql_metrics.average_clearing_price()

    # System Reward Hesaplama (Rapor için kritik başarı metriği)
    bl_reward = - (bl_unmet * 5.0 + bl_curt * 2.0) 
    ql_reward = - (ql_unmet * 5.0 + ql_curt * 2.0)

    # Agent 0-3 üretici olduğu için onların negatif maliyetlerini topluyoruz
    bl_seller_rev = sum(-bl_metrics.agent_total_cost[i] for i in range(4))
    ql_seller_rev = sum(-ql_metrics.agent_total_cost[i] for i in range(4))

    def _pct(a, b):
        """(a - b) / |b| * 100, None if b == 0."""
        if abs(b) < 1e-9:
            return "N/A"
        return f"{(a - b) / abs(b) * 100:+.1f}%"

    width = 54
    sep   = "-" * width

    def _w(line: str = ""):
        """Hem terminale hem özet dosyasına yaz."""
        print(line)
        if file is not None:
            file.write(line + "\n")

    rows = [
        ("Sistem Basari Skoru (Rew)", f"{bl_reward:>10.2f}", f"{ql_reward:>10.2f}", _pct(ql_reward, bl_reward)),
        ("Toplam Satici Geliri ($)",   f"{bl_seller_rev:>10.2f}", f"{ql_seller_rev:>10.2f}", _pct(ql_seller_rev, bl_seller_rev)),
        ("Net Transfer Dengesi ($)",  f"{bl_cost:>10.4f}", f"{ql_cost:>10.4f}", "OK"),
        ("Toplam Ticaret (kWh)",      f"{bl_trade:>10.2f}", f"{ql_trade:>10.2f}", _pct(ql_trade, bl_trade)),
        ("Karsilanmayan (kWh)",       f"{bl_unmet:>10.2f}", f"{ql_unmet:>10.2f}", _pct(ql_unmet, bl_unmet)),
        ("Kisitlama (kWh)",           f"{bl_curt:>10.2f}", f"{ql_curt:>10.2f}", _pct(ql_curt, bl_curt)),
        ("Ort. Takas Fiyati ($/kWh)", f"{bl_price:>10.4f}", f"{ql_price:>10.4f}", _pct(ql_price, bl_price)),
    ]

    title = f"  KARSILASTIRMA: {label_b.upper()} vs BASELINE"
    _w()
    _w("+" + "-" * width + "+")
    _w(f"|{title:^{width}}|")
    _w("+" + "-" * width + "+")
    _w(f"|  {'Metrik':<26} {'Baseline':>10} {label_b:>10}  |")
    _w(f"|  {sep}  |")
    for label, bv, qv, delta in rows:
        _w(f"|  {label:<26} {bv} {qv}  [{delta}]  |")
    _w("+" + "-" * width + "+")
    _w()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_comparison(
    bl_metrics: MetricsCollector,
    ql_metrics: MetricsCollector,
    episode_costs: List[float],
    save_dir: str = ".",
    label_b: str = "Q-Learning",
    filename: str = "ql_vs_baseline_comparison.png",
):
    """4 panel karşılaştırma grafiği oluşturur."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"{label_b} vs Baseline — Karşılaştırma",
        fontsize=15, fontweight="bold",
    )

    hours_bl = np.arange(len(bl_metrics.hourly)) * 0.5
    hours_ql = np.arange(len(ql_metrics.hourly)) * 0.5

    # ── 1. Saatlik ortalama fiyat ─────────────────────────────────────
    ax = axes[0, 0]
    bl_prices = bl_metrics.get_hourly_prices()
    ql_prices = ql_metrics.get_hourly_prices()
    bl_prices_d = np.where(bl_prices > 0, bl_prices, np.nan)
    ql_prices_d = np.where(ql_prices > 0, ql_prices, np.nan)

    ax.plot(hours_bl, bl_prices_d, "b--o", markersize=3, linewidth=1.4,
            label="Baseline", alpha=0.8)
    ax.plot(hours_ql, ql_prices_d, "r-s",  markersize=3, linewidth=1.4,
            label=f"{label_b} (eval)", alpha=0.8)
    ax.set_title("Saatlik Ortalama Takas Fiyatı")
    ax.set_xlabel("Gün İçi Saat")
    ax.set_ylabel("Fiyat ($/kWh)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── 2. Saatlik işlem hacmi ────────────────────────────────────────
    ax = axes[0, 1]
    bl_vol = bl_metrics.get_hourly_volumes()
    ql_vol = ql_metrics.get_hourly_volumes()
    w = 0.18
    ax.bar(hours_bl - w, bl_vol, width=w*1.8, color="steelblue",
           label="Baseline", alpha=0.75)
    ax.bar(hours_ql + w, ql_vol, width=w*1.8, color="tomato",
           label=label_b, alpha=0.75)
    ax.set_title("Saatlik İşlem Hacmi (kWh)")
    ax.set_xlabel("Gün İçi Saat")
    ax.set_ylabel("Enerji (kWh)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ── 3. Ajan bazlı net maliyet ─────────────────────────────────────
    ax = axes[1, 0]
    agent_ids = sorted(bl_metrics.agent_total_cost.keys())
    bl_bills  = [bl_metrics.agent_total_cost[i] for i in agent_ids]
    ql_bills  = [ql_metrics.agent_total_cost[i] for i in agent_ids]
    x = np.arange(len(agent_ids))
    w = 0.35
    bars_bl = ax.bar(x - w/2, bl_bills, width=w, color="steelblue",
                     label="Baseline", alpha=0.8, edgecolor="black", linewidth=0.5)
    bars_ql = ax.bar(x + w/2, ql_bills, width=w, color="tomato",
                     label=label_b, alpha=0.8, edgecolor="black", linewidth=0.5)

    n_prod = sum(1 for agent_id in agent_ids
                 if agent_id < bl_metrics.n_agents // 2)  # yaklaşık
    labels = [f"Pr{i+1}" if i < len(agent_ids)//2 else f"C{i - len(agent_ids)//2 + 1}"
              for i in range(len(agent_ids))]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Ajan Net Maliyeti ($ | negatif = gelir)")
    ax.set_ylabel("Net Maliyet ($)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ── 4. Eğitim sürecinde Sistem Ödülü ─────────────────────────────
    ax = axes[1, 1]
    if episode_costs:
        ep_x = np.arange(1, len(episode_costs) + 1)
        ax.plot(ep_x, episode_costs, "r-", linewidth=1.2, alpha=0.6, label="Sistem Ödülü")

        # Hareketli ortalama
        window = max(5, len(episode_costs) // 10)
        if len(episode_costs) >= window:
            ma = np.convolve(episode_costs, np.ones(window) / window, mode="valid")
            ax.plot(
                np.arange(window, len(episode_costs) + 1),
                ma, "darkred", linewidth=2.0, label=f"Hareketli Ort. (w={window})"
            )

        ax.set_title(f"{label_b} Eğitim Seyri (Performans)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Kümülatif Sistem Ödülü (Reward)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Eğitim verisi yok",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nKarşılaştırma grafiği kaydedildi: {path}")
    return path


def plot_training_curve(episode_costs: List[float], save_dir: str = "."):
    """Sadece eğitim eğrisini ayrı bir dosyaya çiz."""
    if not episode_costs:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ep_x = np.arange(1, len(episode_costs) + 1)
    ax.plot(ep_x, episode_costs, color="tomato", linewidth=1.0, alpha=0.5, label="Episode")

    window = max(5, len(episode_costs) // 10)
    if len(episode_costs) >= window:
        ma = np.convolve(episode_costs, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window, len(episode_costs) + 1), ma,
                "darkred", linewidth=2.5, label=f"Hareketli Ort. (w={window})")

    ax.set_title("Q-Learning Eğitim Seyri — Kümülatif Sistem Ödülü")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Toplam Sistem Ödülü (Reward)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "ql_training_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Eğitim eğrisi kaydedildi: {path}")
    return path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_comparison(
    config: SimConfig,
    n_episodes: int = 400,
    save_dir: str = ".",
    verbose: bool = True,
    summary_file=None,
) -> Tuple:
    """Tam karşılaştırma pipeline'ını çalıştır.

    1. Baseline tek gün
    2. Q-Learning n_episodes eğitim
    3. Q-Learning greedy değerlendirme
    4. Grafik & konsol özeti

    Returns
    -------
    (bl_metrics, ql_metrics, episode_costs)
    """
    from typing import Tuple

    # ── 1. Baseline ──────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print("  ADIM 1/3 — Baseline (Kural Tabanlı) Çalıştırılıyor")
        print("=" * 60)
    bl_env = BaselineEnv(config)
    bl_metrics = bl_env.run_baseline(seed=config.seed, verbose=verbose)

    # ── 2. Q-Learning Eğitim ─────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print(f"  ADIM 2/3 — Q-Learning Eğitimi ({n_episodes} Episode)")
        print("=" * 60)
    ql_env = QLearningEnv(config)
    episode_costs = ql_env.train(n_episodes=n_episodes, verbose=verbose)

    # ── 3. Q-Learning Greedy Değerlendirme ───────────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print("  ADIM 3/3 — Q-Learning Greedy Değerlendirme (ε=0)")
        print("=" * 60)
    ql_metrics = ql_env.run_evaluation(seed=config.seed, verbose=verbose)

    # ── 4. Karşılaştırma çıktıları ───────────────────────────────────
    _print_comparison(bl_metrics, ql_metrics, file=summary_file, label_b="Q-Learn")
    plot_comparison(bl_metrics, ql_metrics, episode_costs, save_dir=save_dir,
                    label_b="Q-Learning", filename="ql_vs_baseline_comparison.png")
    plot_training_curve(episode_costs, save_dir=save_dir)

    return bl_metrics, ql_metrics, episode_costs


def run_nn_comparison(
    config: SimConfig,
    n_episodes: int = 400,
    save_dir: str = ".",
    verbose: bool = True,
    summary_file=None,
):
    """Deep-Q eğitimi + Baseline karşılaştırması.

    run_comparison'ın birebir NN karşılığı.

    Returns
    -------
    (bl_metrics, nn_metrics, episode_rewards)
    """
    from typing import Tuple
    from src.nn_environment import NNQLearningEnv

    # ── 1. Baseline ──────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print("  ADIM 1/3 — Baseline (Kural Tabanlı) Çalıştırılıyor")
        print("=" * 60)
    bl_env = BaselineEnv(config)
    bl_metrics = bl_env.run_baseline(seed=config.seed, verbose=verbose)

    # ── 2. Deep-Q Eğitim ─────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print(f"  ADIM 2/3 — Deep-Q Eğitimi ({n_episodes} Episode)")
        print("=" * 60)
    nn_env = NNQLearningEnv(config)
    episode_rewards = nn_env.train(n_episodes=n_episodes, verbose=verbose)

    # ── 3. Deep-Q Greedy Değerlendirme ───────────────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print("  ADIM 3/3 — Deep-Q Greedy Değerlendirme (ε=0)")
        print("=" * 60)
    nn_metrics = nn_env.run_evaluation(seed=config.seed, verbose=verbose)

    # ── 4. Karşılaştırma çıktıları ───────────────────────────────────
    _print_comparison(bl_metrics, nn_metrics, file=summary_file, label_b="Deep-Q")
    plot_comparison(bl_metrics, nn_metrics, episode_rewards, save_dir=save_dir,
                    label_b="Deep-Q", filename="nn_vs_baseline_comparison.png")
    plot_training_curve(episode_rewards, save_dir=save_dir)

    return bl_metrics, nn_metrics, episode_rewards


def run_all_comparison(
    config: SimConfig,
    n_episodes: int = 400,
    save_dir: str = ".",
    verbose: bool = True,
    summary_file=None,
):
    """Baseline + Q-Learning + Deep-Q — üçlü karşılaştırma (tek lambda).

    Tüm üç ajanı aynı config ile çalıştırır ve yan yana karşılaştırır.

    Returns
    -------
    (bl_metrics, ql_metrics, nn_metrics, ql_rewards, nn_rewards)
    """
    from typing import Tuple
    from src.nn_environment import NNQLearningEnv

    # ── 1. Baseline ──────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print("  ADIM 1/4 — Baseline (Kural Tabanlı)")
        print("=" * 60)
    bl_env = BaselineEnv(config)
    bl_metrics = bl_env.run_baseline(seed=config.seed, verbose=verbose)

    # ── 2. Q-Learning ────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print(f"  ADIM 2/4 — Q-Learning Eğitimi ({n_episodes} Episode)")
        print("=" * 60)
    ql_env = QLearningEnv(config)
    ql_rewards = ql_env.train(n_episodes=n_episodes, verbose=verbose)

    if verbose:
        print("\n" + "=" * 60)
        print("  ADIM 3/4 — Q-Learning Greedy Değerlendirme (ε=0)")
        print("=" * 60)
    ql_metrics = ql_env.run_evaluation(seed=config.seed, verbose=verbose)

    # ── 3. Deep-Q ────────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print(f"  ADIM 4/4 — Deep-Q Eğitimi ({n_episodes} Episode)")
        print("=" * 60)
    nn_env = NNQLearningEnv(config)
    nn_rewards = nn_env.train(n_episodes=n_episodes, verbose=verbose)

    if verbose:
        print("\n" + "=" * 60)
        print("  ADIM 4/4 — Deep-Q Greedy Değerlendirme (ε=0)")
        print("=" * 60)
    nn_metrics = nn_env.run_evaluation(seed=config.seed, verbose=verbose)

    # ── 4. Karşılaştırma çıktıları ───────────────────────────────────
    _print_comparison(bl_metrics, ql_metrics, file=summary_file, label_b="Q-Learn")
    _print_comparison(bl_metrics, nn_metrics, file=summary_file, label_b="Deep-Q")
    plot_comparison(bl_metrics, ql_metrics, ql_rewards, save_dir=save_dir,
                    label_b="Q-Learning", filename="ql_vs_baseline_comparison.png")
    plot_comparison(bl_metrics, nn_metrics, nn_rewards, save_dir=save_dir,
                    label_b="Deep-Q", filename="nn_vs_baseline_comparison.png")
    _plot_3way_comparison(bl_metrics, ql_metrics, nn_metrics, ql_rewards, nn_rewards, save_dir)

    return bl_metrics, ql_metrics, nn_metrics, ql_rewards, nn_rewards


def _plot_3way_comparison(
    bl_metrics: MetricsCollector,
    ql_metrics: MetricsCollector,
    nn_metrics: MetricsCollector,
    ql_rewards: List[float],
    nn_rewards: List[float],
    save_dir: str = ".",
):
    """Baseline vs Q-Learning vs Deep-Q üçlü karşılaştırma grafiği."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Baseline vs Q-Learning vs Deep-Q — Üçlü Karşılaştırma",
                 fontsize=15, fontweight="bold")

    hours = np.arange(48) * 0.5

    # ── 1. Saatlik ortalama fiyat ─────────────────────────────────────
    ax = axes[0, 0]
    for metrics, label, style in [
        (bl_metrics, "Baseline",    "b--o"),
        (ql_metrics, "Q-Learning",  "r-s"),
        (nn_metrics, "Deep-Q",      "g-^"),
    ]:
        prices = metrics.get_hourly_prices()
        prices_d = np.where(prices > 0, prices, np.nan)
        ax.plot(hours[:len(prices_d)], prices_d, style, markersize=3,
                linewidth=1.4, label=label, alpha=0.8)
    ax.set_title("Saatlik Ortalama Takas Fiyatı")
    ax.set_xlabel("Gün İçi Saat")
    ax.set_ylabel("Fiyat ($/kWh)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── 2. Ajan bazlı net maliyet (üç grup çubuk) ─────────────────────
    ax = axes[0, 1]
    agent_ids = sorted(bl_metrics.agent_total_cost.keys())
    bl_bills = [bl_metrics.agent_total_cost[i] for i in agent_ids]
    ql_bills = [ql_metrics.agent_total_cost[i] for i in agent_ids]
    nn_bills = [nn_metrics.agent_total_cost[i] for i in agent_ids]
    x = np.arange(len(agent_ids))
    w = 0.25
    ax.bar(x - w, bl_bills, width=w, color="steelblue", label="Baseline", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.bar(x,     ql_bills, width=w, color="tomato",    label="Q-Learning", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.bar(x + w, nn_bills, width=w, color="seagreen",  label="Deep-Q",    alpha=0.8, edgecolor="black", linewidth=0.5)
    labels = [f"Pr{i+1}" if i < len(agent_ids)//2 else f"C{i - len(agent_ids)//2 + 1}"
              for i in range(len(agent_ids))]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Ajan Net Maliyeti ($ | negatif = gelir)")
    ax.set_ylabel("Net Maliyet ($)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ── 3. Q-Learning eğitim eğrisi ───────────────────────────────────
    ax = axes[1, 0]
    if ql_rewards:
        ep_x = np.arange(1, len(ql_rewards) + 1)
        ax.plot(ep_x, ql_rewards, "r-", linewidth=1.0, alpha=0.5, label="Q-Learning")
        window = max(5, len(ql_rewards) // 10)
        if len(ql_rewards) >= window:
            ma = np.convolve(ql_rewards, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(window, len(ql_rewards) + 1), ma, "darkred", linewidth=2.0,
                    label=f"Q-Learn MA (w={window})")
    ax.set_title("Q-Learning Eğitim Seyri")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Sistem Ödülü")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── 4. Deep-Q eğitim eğrisi ───────────────────────────────────────
    ax = axes[1, 1]
    if nn_rewards:
        ep_x = np.arange(1, len(nn_rewards) + 1)
        ax.plot(ep_x, nn_rewards, "g-", linewidth=1.0, alpha=0.5, label="Deep-Q")
        window = max(5, len(nn_rewards) // 10)
        if len(nn_rewards) >= window:
            ma = np.convolve(nn_rewards, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(window, len(nn_rewards) + 1), ma, "darkgreen", linewidth=2.0,
                    label=f"Deep-Q MA (w={window})")
    ax.set_title("Deep-Q Eğitim Seyri")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Sistem Ödülü")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "all_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nÜçlü karşılaştırma grafiği kaydedildi: {path}")
    return path


# Lambda sweep varsayılan değerleri (log ile aynı sıra)
_DEFAULT_LAMBDAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-Learning / Deep-Q / Baseline Karşılaştırması")
    parser.add_argument("--agent", type=str, default="ql",
                        choices=["ql", "nn", "all"],
                        help="Ajan tipi: ql=Q-Learning sweep, nn=Deep-Q sweep, all=üçlü tek çalıştırma")
    parser.add_argument("--episodes", type=int, default=400,
                        help="Eğitim episode sayısı (varsayılan: 400)")
    parser.add_argument("--save_dir", type=str, default=".",
                        help="Grafiklerin kaydedileceği dizin")
    parser.add_argument(
        "--lambdas", type=float, nargs="+", default=_DEFAULT_LAMBDAS,
        help="Lambda_UILI sweep değerleri (varsayılan: 0.05..0.50). "
             "--agent all için sadece ilk değer kullanılır.",
    )
    parser.add_argument(
        "--beta", type=float, default=0.3,
        help="Reward fiyat primi katsayisi β (varsayılan: 0.3)",
    )
    args = parser.parse_args()

    summary_path = os.path.join(args.save_dir, "simulation_summary_log.txt")

    # ── Üçlü karşılaştırma (all) — lambda sweep yok ───────────────────
    if args.agent == "all":
        lam = args.lambdas[0]
        print(f"\n[ALL] Baseline + Q-Learning + Deep-Q | λ={lam:.2f} | {args.episodes} ep")
        os.makedirs(args.save_dir, exist_ok=True)
        cfg = SimConfig(lambda_uili=lam, reward_beta=args.beta)
        with open(summary_path, "w", encoding="utf-8") as sf:
            run_all_comparison(cfg, n_episodes=args.episodes,
                               save_dir=args.save_dir, verbose=True, summary_file=sf)
        print(f"\nOzet log dosyasi: {summary_path}")

    # ── Lambda sweep (ql veya nn) ─────────────────────────────────────
    else:
        runner = run_comparison if args.agent == "ql" else run_nn_comparison
        agent_label = "Q-LEARNING" if args.agent == "ql" else "DEEP-Q"

        with open(summary_path, "w", encoding="utf-8") as sf:
            top_header = (
                f"\n{'='*60}\n"
                f"{f'HYPERPARAMETER SEARCH: LAMBDA_UILI ({agent_label})':^60}\n"
                f"  beta={args.beta:.2f}  |  episodes={args.episodes}\n"
                f"{'='*60}\n"
            )
            print(top_header, end="")
            sf.write(top_header)

            for lam in args.lambdas:
                lam_header = (
                    f"\n{'='*60}\n"
                    f" LAMBDA_UILI = {lam:.2f} RESULTS\n"
                    f"{'='*60}\n"
                )
                print(lam_header, end="")
                sf.write(lam_header)

                lam_tag = f"lambda_{lam:.2f}".replace(".", "_")
                lam_dir = os.path.join(args.save_dir, lam_tag)
                os.makedirs(lam_dir, exist_ok=True)

                cfg = SimConfig(lambda_uili=lam, reward_beta=args.beta)
                runner(cfg, n_episodes=args.episodes, save_dir=lam_dir,
                       verbose=False, summary_file=sf)

        print(f"\nOzet log dosyasi olusturuldu: {summary_path}")