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
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import SimConfig
from src.ql_environment import QLearningEnv, BaselineEnv
from src.reinforce_environment import REINFORCEEnv
from src.metrics import MetricsCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_avg(arr: np.ndarray) -> float:
    valid = arr[arr > 0]
    return float(np.mean(valid)) if len(valid) > 0 else 0.0


def _print_comparison(bl_metrics: MetricsCollector, ql_metrics: MetricsCollector, file=None):
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

    _w()
    _w("+" + "-" * width + "+")
    _w(f"|{'  KARSILASTIRMA: Q-LEARNING vs BASELINE':^{width}}|")
    _w("+" + "-" * width + "+")
    _w(f"|  {'Metrik':<26} {'Baseline':>10} {'Q-Learn':>10}  |")
    _w(f"|  {sep}  |")
    for label, bv, qv, delta in rows:
        _w(f"|  {label:<26} {bv} {qv}  [{delta}]  |")
    _w("+" + "-" * width + "+")
    _w()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _print_comparison_3way(
    bl_metrics: MetricsCollector,
    ql_metrics: MetricsCollector,
    nn_metrics: MetricsCollector,
    file=None,
):
    """Baseline / Q-Learning / REINFORCE-NN üç sütunlu karşılaştırma tablosu."""
    bl_cost  = sum(bl_metrics.agent_total_cost.values())
    ql_cost  = sum(ql_metrics.agent_total_cost.values())
    nn_cost  = sum(nn_metrics.agent_total_cost.values())

    bl_trade = bl_metrics.total_internal_trading()
    ql_trade = ql_metrics.total_internal_trading()
    nn_trade = nn_metrics.total_internal_trading()

    bl_unmet = bl_metrics.total_unmet_demand()
    ql_unmet = ql_metrics.total_unmet_demand()
    nn_unmet = nn_metrics.total_unmet_demand()

    bl_curt  = bl_metrics.total_curtailment()
    ql_curt  = ql_metrics.total_curtailment()
    nn_curt  = nn_metrics.total_curtailment()

    bl_price = bl_metrics.average_clearing_price()
    ql_price = ql_metrics.average_clearing_price()
    nn_price = nn_metrics.average_clearing_price()

    bl_reward = -(bl_unmet * 5.0 + bl_curt * 2.0)
    ql_reward = -(ql_unmet * 5.0 + ql_curt * 2.0)
    nn_reward = -(nn_unmet * 5.0 + nn_curt * 2.0)

    n_prod = bl_metrics.n_agents // 2
    bl_seller_rev = sum(-bl_metrics.agent_total_cost[i] for i in range(n_prod))
    ql_seller_rev = sum(-ql_metrics.agent_total_cost[i] for i in range(n_prod))
    nn_seller_rev = sum(-nn_metrics.agent_total_cost[i] for i in range(n_prod))

    def _pct(a, b):
        if abs(b) < 1e-9:
            return "   N/A"
        return f"{(a - b) / abs(b) * 100:+6.1f}%"

    width = 82

    def _w(line: str = ""):
        print(line)
        if file is not None:
            file.write(line + "\n")

    rows = [
        ("Sistem Basari Skoru",
         f"{bl_reward:>10.2f}", f"{ql_reward:>10.2f}", f"{nn_reward:>10.2f}",
         _pct(ql_reward, bl_reward), _pct(nn_reward, bl_reward)),
        ("Satici Geliri ($)",
         f"{bl_seller_rev:>10.2f}", f"{ql_seller_rev:>10.2f}", f"{nn_seller_rev:>10.2f}",
         _pct(ql_seller_rev, bl_seller_rev), _pct(nn_seller_rev, bl_seller_rev)),
        ("Net Transfer Dengesi ($)",
         f"{bl_cost:>10.4f}", f"{ql_cost:>10.4f}", f"{nn_cost:>10.4f}",
         "      OK", "      OK"),
        ("Toplam Ticaret (kWh)",
         f"{bl_trade:>10.2f}", f"{ql_trade:>10.2f}", f"{nn_trade:>10.2f}",
         _pct(ql_trade, bl_trade), _pct(nn_trade, bl_trade)),
        ("Karsilanmayan (kWh)",
         f"{bl_unmet:>10.2f}", f"{ql_unmet:>10.2f}", f"{nn_unmet:>10.2f}",
         _pct(ql_unmet, bl_unmet), _pct(nn_unmet, bl_unmet)),
        ("Kisitlama (kWh)",
         f"{bl_curt:>10.2f}", f"{ql_curt:>10.2f}", f"{nn_curt:>10.2f}",
         _pct(ql_curt, bl_curt), _pct(nn_curt, bl_curt)),
        ("Ort. Takas Fiyati ($/kWh)",
         f"{bl_price:>10.4f}", f"{ql_price:>10.4f}", f"{nn_price:>10.4f}",
         _pct(ql_price, bl_price), _pct(nn_price, bl_price)),
    ]

    _w()
    _w("+" + "=" * width + "+")
    _w(f"|{'  3-YOL KARSILASTIRMA: Baseline | QL(Alici+Satici) | NN(Alici+Satici)':^{width}}|")
    _w("+" + "=" * width + "+")
    _w(f"|  {'Metrik':<26} {'Baseline':>10} {'QL(B+S)':>10} {'NN(B+S)':>10}  {'ΔQL':>7}  {'ΔNN':>7}  |")
    _w(f"|  {'-' * (width - 4)}  |")
    for label, bv, qv, nv, dql, dnn in rows:
        _w(f"|  {label:<26} {bv} {qv} {nv}  {dql}  {dnn}  |")
    _w("+" + "=" * width + "+")
    _w("  (ΔQL, ΔNN = Baseline'a göre değişim yüzdesi)")
    _w()


def plot_comparison(
    bl_metrics: MetricsCollector,
    ql_metrics: MetricsCollector,
    episode_costs: List[float],
    save_dir: str = ".",
    nn_metrics: Optional[MetricsCollector] = None,
    nn_episode_costs: Optional[List[float]] = None,
):
    """4 panel karşılaştırma grafiği oluşturur. NN verisi varsa 3 model gösterilir."""
    has_nn = nn_metrics is not None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title = ("3-Yol Karşılaştırma: Baseline | QL(Alıcı+Satıcı) | NN(Alıcı+Satıcı)"
             if has_nn else "Q-Learning vs Baseline — Karşılaştırma")
    fig.suptitle(title, fontsize=15, fontweight="bold")

    hours_bl = np.arange(len(bl_metrics.hourly)) * 0.5
    hours_ql = np.arange(len(ql_metrics.hourly)) * 0.5

    # ── 1. Saatlik ortalama fiyat ─────────────────────────────────────
    ax = axes[0, 0]
    bl_prices   = bl_metrics.get_hourly_prices()
    ql_prices   = ql_metrics.get_hourly_prices()
    bl_prices_d = np.where(bl_prices > 0, bl_prices, np.nan)
    ql_prices_d = np.where(ql_prices > 0, ql_prices, np.nan)

    ax.plot(hours_bl, bl_prices_d, "b--o", markersize=3, linewidth=1.4,
            label="Baseline", alpha=0.8)
    ax.plot(hours_ql, ql_prices_d, "r-s",  markersize=3, linewidth=1.4,
            label="Q-Learning (eval)", alpha=0.8)
    if has_nn:
        hours_nn    = np.arange(len(nn_metrics.hourly)) * 0.5
        nn_prices   = nn_metrics.get_hourly_prices()
        nn_prices_d = np.where(nn_prices > 0, nn_prices, np.nan)
        ax.plot(hours_nn, nn_prices_d, "g-^", markersize=3, linewidth=1.4,
                label="REINFORCE-NN (eval)", alpha=0.8)
    ax.set_title("Saatlik Ortalama Takas Fiyatı")
    ax.set_xlabel("Gün İçi Saat")
    ax.set_ylabel("Fiyat ($/kWh)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── 2. Saatlik işlem hacmi ────────────────────────────────────────
    ax = axes[0, 1]
    bl_vol = bl_metrics.get_hourly_volumes()
    ql_vol = ql_metrics.get_hourly_volumes()
    if has_nn:
        nn_vol = nn_metrics.get_hourly_volumes()
        hours_nn = np.arange(len(nn_metrics.hourly)) * 0.5
        w = 0.13
        ax.bar(hours_bl - w, bl_vol, width=w*1.8, color="steelblue",
               label="Baseline", alpha=0.75)
        ax.bar(hours_ql,      ql_vol, width=w*1.8, color="tomato",
               label="Q-Learning", alpha=0.75)
        ax.bar(hours_nn + w, nn_vol, width=w*1.8, color="mediumseagreen",
               label="REINFORCE-NN", alpha=0.75)
    else:
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
    labels = [f"Pr{i+1}" if i < len(agent_ids)//2 else f"C{i - len(agent_ids)//2 + 1}"
              for i in range(len(agent_ids))]

    if has_nn:
        nn_bills = [nn_metrics.agent_total_cost[i] for i in agent_ids]
        w = 0.25
        ax.bar(x - w, bl_bills, width=w, color="steelblue",
               label="Baseline", alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.bar(x,     ql_bills, width=w, color="tomato",
               label="Q-Learning", alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.bar(x + w, nn_bills, width=w, color="mediumseagreen",
               label="REINFORCE-NN", alpha=0.8, edgecolor="black", linewidth=0.5)
    else:
        w = 0.35
        ax.bar(x - w/2, bl_bills, width=w, color="steelblue",
               label="Baseline", alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.bar(x + w/2, ql_bills, width=w, color="tomato",
               label="Q-Learning", alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Ajan Net Maliyeti ($ | negatif = gelir)")
    ax.set_ylabel("Net Maliyet ($)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ── 4. Eğitim seyri ───────────────────────────────────────────────
    ax = axes[1, 1]
    plotted = False

    if episode_costs:
        ep_x = np.arange(1, len(episode_costs) + 1)
        ax.plot(ep_x, episode_costs, "r-", linewidth=1.0, alpha=0.45, label="Q-Learning")
        window = max(5, len(episode_costs) // 10)
        if len(episode_costs) >= window:
            ma = np.convolve(episode_costs, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(window, len(episode_costs) + 1), ma,
                    "darkred", linewidth=2.0, label=f"QL Hareketli Ort. (w={window})")
        plotted = True

    if has_nn and nn_episode_costs:
        ep_x_nn = np.arange(1, len(nn_episode_costs) + 1)
        ax.plot(ep_x_nn, nn_episode_costs, "g-", linewidth=1.0, alpha=0.45,
                label="REINFORCE-NN")
        window_nn = max(5, len(nn_episode_costs) // 10)
        if len(nn_episode_costs) >= window_nn:
            ma_nn = np.convolve(nn_episode_costs, np.ones(window_nn) / window_nn, mode="valid")
            ax.plot(np.arange(window_nn, len(nn_episode_costs) + 1), ma_nn,
                    "darkgreen", linewidth=2.0,
                    label=f"NN Hareketli Ort. (w={window_nn})")
        plotted = True

    if plotted:
        ax.set_title("Eğitim Seyri — Kümülatif Sistem Ödülü")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Sistem Ödülü")
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
    n_episodes: int = 400,
    nn_episodes: int = 800,
    save_dir: str = ".",
    verbose: bool = True,
    summary_file=None,
) -> Tuple:
    """Tam 3-yol karşılaştırma pipeline'ını çalıştır.

    1. Baseline tek gün
    2. Q-Learning  → n_episodes eğitim + greedy değerlendirme
    3. REINFORCE-NN → nn_episodes eğitim + deterministik değerlendirme
    4. 3-yol tablo + grafik + log dosyası

    Returns
    -------
    (bl_metrics, ql_metrics, nn_metrics, ql_episode_costs, nn_episode_costs)
    """
    def _log(line: str = ""):
        """Hem terminal hem log dosyasına yaz."""
        print(line)
        if summary_file is not None:
            summary_file.write(line + "\n")

    # ── 1. Baseline ──────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print("  ADIM 1/3 — Baseline (Kural Tabanlı) Çalıştırılıyor")
        print("=" * 60)
    bl_env     = BaselineEnv(config)
    bl_metrics = bl_env.run_baseline(seed=config.seed, verbose=verbose)

    if summary_file is not None:
        summary_file.write("\n[ADIM 1/3] BASELINE SONUÇLARI\n")
        summary_file.write(bl_metrics.summary() + "\n")

    # ── 2. Q-Learning Eğitim + Değerlendirme ─────────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print(f"  ADIM 2/3 — Q-Learning Eğitimi ({n_episodes} Episode)")
        print("=" * 60)
    ql_env        = QLearningEnv(config)
    ql_ep_costs   = ql_env.train(n_episodes=n_episodes, verbose=verbose)

    if verbose:
        print("\n" + "-" * 60)
        print("  Q-Learning Greedy Değerlendirme (ε=0)")
        print("-" * 60)
    ql_metrics = ql_env.run_evaluation(seed=config.seed, verbose=verbose)

    if summary_file is not None:
        summary_file.write(f"\n[ADIM 2/3] Q-LEARNING SONUÇLARI ({n_episodes} ep)\n")
        summary_file.write(ql_metrics.summary() + "\n")

    # ── 3. REINFORCE-NN Eğitim + Değerlendirme ───────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print(f"  ADIM 3/3 — REINFORCE-NN Eğitimi ({nn_episodes} Episode)")
        print("=" * 60)
    nn_env      = REINFORCEEnv(config)
    nn_ep_costs = nn_env.train(n_episodes=nn_episodes, verbose=verbose)

    if verbose:
        print("\n" + "-" * 60)
        print("  REINFORCE-NN Deterministik Değerlendirme (σ=0)")
        print("-" * 60)
    nn_metrics = nn_env.run_evaluation(seed=config.seed, verbose=verbose)

    if summary_file is not None:
        summary_file.write(f"\n[ADIM 3/3] REINFORCE-NN SONUÇLARI ({nn_episodes} ep)\n")
        summary_file.write(nn_metrics.summary() + "\n")

    # ── 4. 3-Yol Karşılaştırma Tablosu ───────────────────────────────
    _print_comparison_3way(bl_metrics, ql_metrics, nn_metrics, file=summary_file)
    plot_comparison(
        bl_metrics, ql_metrics, ql_ep_costs, save_dir=save_dir,
        nn_metrics=nn_metrics, nn_episode_costs=nn_ep_costs,
    )
    plot_training_curve(ql_ep_costs, save_dir=save_dir)

    return bl_metrics, ql_metrics, nn_metrics, ql_ep_costs, nn_ep_costs


# Lambda sweep varsayılan değerleri
_DEFAULT_LAMBDAS = [0.15]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline | Q-Learning | REINFORCE-NN — 3-Yol Karşılaştırma"
    )
    parser.add_argument("--episodes", type=int, default=400,
                        help="Q-Learning eğitim episode sayısı (varsayılan: 400)")
    parser.add_argument("--nn_episodes", type=int, default=800,
                        help="REINFORCE-NN eğitim episode sayısı (varsayılan: 800)")
    parser.add_argument("--save_dir", type=str, default=".",
                        help="Grafik ve log dosyalarının kaydedileceği dizin")
    parser.add_argument(
        "--lambdas", type=float, nargs="+", default=_DEFAULT_LAMBDAS,
        help="Lambda_UILI sweep değerleri (varsayılan: 0.15)",
    )
    parser.add_argument(
        "--beta", type=float, default=0.3,
        help="Reward fiyat primi katsayisi β (varsayılan: 0.3)",
    )
    parser.add_argument(
        "--curtail-rate", type=float, default=-0.03,
        help="Satılamayan enerji ödül oranı (varsayılan: -0.03)",
    )
    parser.add_argument(
        "--log_name", type=str, default="overall_summary.txt",
        help="Log dosyası adı (varsayılan: overall_summary.txt)",
    )
    args = parser.parse_args()

    log_path = os.path.join(args.save_dir, args.log_name)

    with open(log_path, "w", encoding="utf-8") as sf:

        # ── Log dosyası başlığı ───────────────────────────────────────
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = (
            f"\n{'='*84}\n"
            f"{'P2P ENERGY TRADING — OVERALL COMPARISON (Buyer+Seller Learning)':^84}\n"
            f"  Tarih : {ts}\n"
            f"  QL Episodes : {args.episodes}  |  NN Episodes : {args.nn_episodes}\n"
            f"  beta={args.beta:.2f}  |  curtail_rate={args.curtail_rate:.2f}\n"
            f"  Modeller: Baseline | QL(Alici+Satici) | NN(Alici+Satici)\n"
            f"{'='*84}\n"
        )
        print(header, end="")
        sf.write(header)

        for lam in args.lambdas:
            lam_header = (
                f"\n{'='*84}\n"
                f"  LAMBDA_UILI = {lam:.2f}\n"
                f"{'='*84}\n"
            )
            print(lam_header, end="")
            sf.write(lam_header)

            lam_tag = f"lambda_{lam:.2f}".replace(".", "_")
            lam_dir = os.path.join(args.save_dir, lam_tag)
            os.makedirs(lam_dir, exist_ok=True)

            cfg = SimConfig(
                lambda_uili=lam,
                reward_beta=args.beta,
                reward_curtail_rate=args.curtail_rate,
            )
            run_comparison(
                cfg,
                n_episodes=args.episodes,
                nn_episodes=args.nn_episodes,
                save_dir=lam_dir,
                verbose=False,
                summary_file=sf,
            )

    print(f"\nSimülasyon log dosyasi olusturuldu: {log_path}")