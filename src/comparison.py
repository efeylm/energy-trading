"""
Q-Learning vs Baseline Karşılaştırma Modülü

Kullanım:
    python -m src.comparison          # Varsayılan ayarlar
    python -m src.comparison --episodes 200

Adımlar:
1. Baseline ajanları aynı seed ile tek bir günde çalıştır.
2. Q-Learning ajanları n_episodes boyunca eğit.
3. Q-Learning ajanlarını greedy (ε=0) modunda değerlendir.
4. Her iki sonucu karşılaştıran metrikler ve grafikler üret.
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


def _print_comparison(bl_metrics: MetricsCollector, ql_metrics: MetricsCollector):
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
    sep   = "─" * width

    print()
    print("╔" + "═" * width + "╗")
    print(f"║{'  KARŞILAŞTIRMA: Q-LEARNING vs BASELINE':^{width}}║")
    print("╠" + "═" * width + "╣")
    print(f"║  {'Metrik':<26} {'Baseline':>10} {'Q-Learn':>10}  ║")
    print(f"║  {sep}  ║")

    rows = [
        ("Sistem Başarı Skoru (Rew)", f"{bl_reward:>10.2f}", f"{ql_reward:>10.2f}", _pct(ql_reward, bl_reward)),
        ("Toplam Satıcı Geliri ($)",   f"{bl_seller_rev:>10.2f}", f"{ql_seller_rev:>10.2f}", _pct(ql_seller_rev, bl_seller_rev)),
        ("Net Transfer Dengesi ($)",  f"{bl_cost:>10.4f}", f"{ql_cost:>10.4f}", "OK"),
        ("Toplam Ticaret (kWh)",    f"{bl_trade:>10.2f}", f"{ql_trade:>10.2f}", _pct(ql_trade, bl_trade)),
        ("Karşılanmayan (kWh)",     f"{bl_unmet:>10.2f}", f"{ql_unmet:>10.2f}", _pct(ql_unmet, bl_unmet)),
        ("Kısıtlama (kWh)",         f"{bl_curt:>10.2f}", f"{ql_curt:>10.2f}", _pct(ql_curt, bl_curt)),
        ("Ort. Takas Fiyatı ($/kWh)", f"{bl_price:>10.4f}", f"{ql_price:>10.4f}", _pct(ql_price, bl_price)),
    ]

    for label, bv, qv, delta in rows:
        print(f"║  {label:<26} {bv} {qv}  [{delta}]  ║")

    print("╚" + "═" * width + "╝")
    print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_comparison(
    bl_metrics: MetricsCollector,
    ql_metrics: MetricsCollector,
    episode_costs: List[float],
    save_dir: str = ".",
):
    """4 panel karşılaştırma grafiği oluşturur."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Q-Learning vs Baseline — Karşılaştırma",
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
            label="Q-Learning (eval)", alpha=0.8)
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
           label="Q-Learning", alpha=0.75)
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
                     label="Q-Learning", alpha=0.8, edgecolor="black", linewidth=0.5)

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

        ax.set_title("Q-Learning Eğitim Seyri (Performans)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Kümülatif Sistem Ödülü (Reward)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Eğitim verisi yok",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(save_dir, "ql_vs_baseline_comparison.png")
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
    n_episodes: int = 100,
    save_dir: str = ".",
    verbose: bool = True,
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
    print("\n" + "━" * 60)
    print("  ADIM 1/3 — Baseline (Kural Tabanlı) Çalıştırılıyor")
    print("━" * 60)
    bl_env = BaselineEnv(config)
    bl_metrics = bl_env.run_baseline(seed=config.seed)

    # ── 2. Q-Learning Eğitim ─────────────────────────────────────────
    print("\n" + "━" * 60)
    print(f"  ADIM 2/3 — Q-Learning Eğitimi ({n_episodes} Episode)")
    print("━" * 60)
    ql_env = QLearningEnv(config)
    episode_costs = ql_env.train(n_episodes=n_episodes, verbose=verbose)

    # ── 3. Q-Learning Greedy Değerlendirme ───────────────────────────
    print("\n" + "━" * 60)
    print("  ADIM 3/3 — Q-Learning Greedy Değerlendirme (ε=0)")
    print("━" * 60)
    ql_metrics = ql_env.run_evaluation(seed=config.seed)

    # ── 4. Karşılaştırma çıktıları ───────────────────────────────────
    _print_comparison(bl_metrics, ql_metrics)
    plot_comparison(bl_metrics, ql_metrics, episode_costs, save_dir=save_dir)
    plot_training_curve(episode_costs, save_dir=save_dir)

    return bl_metrics, ql_metrics, episode_costs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-Learning vs Baseline Karşılaştırması")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Q-Learning eğitim episode sayısı (varsayılan: 100)")
    parser.add_argument("--save_dir", type=str, default=".",
                        help="Grafiklerin kaydedileceği dizin")
    args = parser.parse_args()

    cfg = SimConfig()
    run_comparison(cfg, n_episodes=args.episodes, save_dir=args.save_dir)