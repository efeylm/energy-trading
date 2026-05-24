"""
Deep-Q Standalone Eğitim ve Değerlendirme.

Sadece NN ajanını eğitmek ve sonuçlarını görmek için.
comparison.py'nin tam sweep'ini çalıştırmadan hızlı test için kullanın.

Kullanım:
    python -m src.nn_run
    python -m src.nn_run --episodes 500
    python -m src.nn_run --episodes 300 --lambda-uili 0.2 --save-dir results/
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import SimConfig
from src.nn_environment import NNQLearningEnv
from src.ql_environment import BaselineEnv
from src.nn_agent import DeepSellerQAgent


def run_nn(
    config: SimConfig,
    n_episodes: int = 400,
    save_dir: str = ".",
    verbose: bool = True,
):
    """Deep-Q eğitimi + baseline karşılaştırması + grafik."""

    os.makedirs(save_dir, exist_ok=True)

    # ── 1. Baseline ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ADIM 1/3 — Baseline Çalıştırılıyor")
    print("=" * 60)
    bl_env     = BaselineEnv(config)
    bl_metrics = bl_env.run_baseline(seed=config.seed, verbose=verbose)

    # ── 2. Deep-Q Eğitim ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  ADIM 2/3 — Deep-Q Eğitimi ({n_episodes} Episode)")
    print("=" * 60)
    nn_env      = NNQLearningEnv(config)
    ep_rewards  = nn_env.train(n_episodes=n_episodes, verbose=verbose)

    # ── 3. Greedy Değerlendirme ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ADIM 3/3 — Deep-Q Greedy Değerlendirme (ε=0)")
    print("=" * 60)
    nn_metrics = nn_env.run_evaluation(seed=config.seed, verbose=verbose)

    # ── 4. Konsol özeti ───────────────────────────────────────────────
    _print_summary(bl_metrics, nn_metrics, ep_rewards)

    # ── 5. Grafikler ──────────────────────────────────────────────────
    _plot_training(ep_rewards, save_dir)
    _plot_comparison(bl_metrics, nn_metrics, ep_rewards, save_dir)

    # ── 6. Ajan istatistikleri ────────────────────────────────────────
    print("\n--- Deep-Q Ajan İstatistikleri ---")
    for agent in nn_env.agents:
        if isinstance(agent, DeepSellerQAgent):
            stats = agent.agent_stats()
            print(f"  Ajan {agent.agent_id}: {stats}")

    return bl_metrics, nn_metrics, ep_rewards


def _print_summary(bl_metrics, nn_metrics, ep_rewards):
    """Konsola özet tablosu yazdır."""
    bl_unmet = bl_metrics.total_unmet_demand()
    nn_unmet = nn_metrics.total_unmet_demand()
    bl_curt  = bl_metrics.total_curtailment()
    nn_curt  = nn_metrics.total_curtailment()
    bl_price = bl_metrics.average_clearing_price()
    nn_price = nn_metrics.average_clearing_price()
    bl_trade = bl_metrics.total_internal_trading()
    nn_trade = nn_metrics.total_internal_trading()

    bl_seller_rev = sum(-bl_metrics.agent_total_cost[i] for i in range(4))
    nn_seller_rev = sum(-nn_metrics.agent_total_cost[i] for i in range(4))

    def pct(a, b):
        if abs(b) < 1e-9: return "N/A"
        return f"{(a-b)/abs(b)*100:+.1f}%"

    width = 54
    print()
    print("+" + "-" * width + "+")
    print(f"|{'  DEEP-Q vs BASELINE KARSILASTIRMASI':^{width}}|")
    print("+" + "-" * width + "+")
    print(f"|  {'Metrik':<26} {'Baseline':>10} {'Deep-Q':>10}  |")
    print(f"|  {'-'*width}  |")
    rows = [
        ("Satici Geliri ($)",         f"{bl_seller_rev:>10.2f}", f"{nn_seller_rev:>10.2f}", pct(nn_seller_rev, bl_seller_rev)),
        ("Toplam Ticaret (kWh)",      f"{bl_trade:>10.2f}",      f"{nn_trade:>10.2f}",      pct(nn_trade, bl_trade)),
        ("Karsilanmayan (kWh)",       f"{bl_unmet:>10.2f}",      f"{nn_unmet:>10.2f}",      pct(nn_unmet, bl_unmet)),
        ("Kisitlama (kWh)",           f"{bl_curt:>10.2f}",       f"{nn_curt:>10.2f}",       pct(nn_curt, bl_curt)),
        ("Ort. Takas Fiyati ($/kWh)", f"{bl_price:>10.4f}",      f"{nn_price:>10.4f}",      pct(nn_price, bl_price)),
    ]
    for label, bv, nv, delta in rows:
        print(f"|  {label:<26} {bv} {nv}  [{delta}]  |")
    print("+" + "-" * width + "+")

    if ep_rewards:
        last_10 = ep_rewards[-10:]
        print(f"\n  Son 10 episode ort. ödül: {sum(last_10)/len(last_10):.2f}")
        print(f"  Toplam eğitim episode:    {len(ep_rewards)}")


def _plot_training(ep_rewards, save_dir):
    if not ep_rewards:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ep_x = np.arange(1, len(ep_rewards) + 1)
    ax.plot(ep_x, ep_rewards, "g-", linewidth=1.0, alpha=0.5, label="Episode Ödülü")
    window = max(5, len(ep_rewards) // 10)
    if len(ep_rewards) >= window:
        ma = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window, len(ep_rewards) + 1), ma,
                "darkgreen", linewidth=2.5, label=f"Hareketli Ort. (w={window})")
    ax.set_title("Deep-Q Eğitim Seyri — Sistem Ödülü")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Toplam Sistem Ödülü")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "nn_training_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Eğitim eğrisi kaydedildi: {path}")


def _plot_comparison(bl_metrics, nn_metrics, ep_rewards, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Deep-Q vs Baseline", fontsize=14, fontweight="bold")

    hours = np.arange(48) * 0.5

    # Saatlik fiyat
    ax = axes[0]
    bl_prices = np.where(bl_metrics.get_hourly_prices() > 0, bl_metrics.get_hourly_prices(), np.nan)
    nn_prices = np.where(nn_metrics.get_hourly_prices() > 0, nn_metrics.get_hourly_prices(), np.nan)
    ax.plot(hours[:len(bl_prices)], bl_prices, "b--o", markersize=3, label="Baseline", alpha=0.8)
    ax.plot(hours[:len(nn_prices)], nn_prices, "g-^",  markersize=3, label="Deep-Q (eval)", alpha=0.8)
    ax.set_title("Saatlik Ortalama Takas Fiyatı")
    ax.set_xlabel("Saat")
    ax.set_ylabel("$/kWh")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ajan net maliyet
    ax = axes[1]
    agent_ids = sorted(bl_metrics.agent_total_cost.keys())
    x = np.arange(len(agent_ids))
    w = 0.35
    ax.bar(x - w/2, [bl_metrics.agent_total_cost[i] for i in agent_ids], width=w,
           color="steelblue", label="Baseline", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, [nn_metrics.agent_total_cost[i] for i in agent_ids], width=w,
           color="seagreen",  label="Deep-Q",   alpha=0.8, edgecolor="black", linewidth=0.5)
    labels = [f"Pr{i+1}" if i < len(agent_ids)//2 else f"C{i - len(agent_ids)//2 + 1}"
              for i in range(len(agent_ids))]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Ajan Net Maliyeti (negatif = gelir)")
    ax.set_ylabel("Net Maliyet ($)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(save_dir, "nn_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Karşılaştırma grafiği kaydedildi: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep-Q Standalone Eğitim")
    parser.add_argument("--episodes",     type=int,   default=400,   help="Eğitim episode sayısı")
    parser.add_argument("--lambda-uili",  type=float, default=0.2,   help="UILI fiyat decay oranı")
    parser.add_argument("--beta",         type=float, default=1.0,   help="Reward β katsayısı (fiyat primi)")
    parser.add_argument("--save-dir",     type=str,   default=".",   help="Grafik kayıt dizini")
    parser.add_argument("--quiet",        action="store_true",        help="Verbose çıktıyı kapat")
    args = parser.parse_args()

    cfg = SimConfig(
        lambda_uili=args.lambda_uili,
        reward_beta=args.beta,
    )

    run_nn(
        config=cfg,
        n_episodes=args.episodes,
        save_dir=args.save_dir,
        verbose=not args.quiet,
    )
