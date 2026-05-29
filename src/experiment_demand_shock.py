"""
Talep Şoku Deneyi — Demand Shock Experiment

Üç fazlı eğitim:
  Faz 1 (denge)   : episodes=6000, demand_multiplier=1.0
  Faz 2 (şok)     : episodes=5000, demand_multiplier=2.0  ← epsilon resetlenir
  Faz 3 (toparlanma): episodes=5000, demand_multiplier=1.0 ← epsilon resetlenir

Çıktılar:
  - demand_shock_results/timeseries.png    : 4 panelli zaman serisi
  - demand_shock_results/supply_demand.png : denge arz-talep eğrisi

Kullanım:
    python3 -m src.experiment_demand_shock
    python3 -m src.experiment_demand_shock --p1 6000 --p2 5000 --p3 5000
"""

from __future__ import annotations

import argparse
import os
from typing import List, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.config import SimConfig
from src.ql_environment import QLearningEnv


# ---------------------------------------------------------------------------
# Yardımcı: hareketli ortalama
# ---------------------------------------------------------------------------

def _moving_avg(arr: np.ndarray, w: int) -> np.ndarray:
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="valid")


# ---------------------------------------------------------------------------
# Arz-Talep Eğrisi
# ---------------------------------------------------------------------------

def extract_supply_demand_curve(env: QLearningEnv) -> Dict:
    """Greedy evaluation'dan arz ve talep eğrilerini çıkarır."""
    # Epsilon=0 ile tek gün çalıştır, sipariş log'unu topla
    saved_training = env.training_enabled
    env.training_enabled = False
    saved_eps = {}
    for agent in env._ql_agents.values():
        saved_eps[agent.agent_id] = agent.epsilon
        agent.epsilon = 0.0

    env._suppress_profile_print = True
    env.reset(seed=env.config.seed + 99999)
    env._suppress_profile_print = False

    done = False
    while not done:
        _, _, done, _ = env.step()

    # Sipariş log'undan alış ve satış emirlerini topla
    buy_orders: List[tuple] = []
    sell_orders: List[tuple] = []
    for hour_log in env.bid_ask_log:
        for orders in hour_log.values():
            for o in orders:
                if o.is_buy:
                    buy_orders.append((o.price, o.quantity))
                else:
                    sell_orders.append((o.price, o.quantity))

    # Geri yükle
    env.training_enabled = saved_training
    for agent in env._ql_agents.values():
        agent.epsilon = saved_eps[agent.agent_id]

    # Arz eğrisi: fiyata göre artan sıra, kümülatif miktar
    sell_orders.sort(key=lambda x: x[0])
    # Talep eğrisi: fiyata göre azalan sıra, kümülatif miktar
    buy_orders.sort(key=lambda x: -x[0])

    def to_curve(orders):
        if not orders:
            return np.array([0.0]), np.array([0.0])
        prices = np.array([p for p, _ in orders])
        qtys   = np.array([q for _, q in orders])
        cum    = np.cumsum(qtys)
        return prices, cum

    sell_p, sell_cum = to_curve(sell_orders)
    buy_p,  buy_cum  = to_curve(buy_orders)

    return {"sell_prices": sell_p, "sell_cum": sell_cum,
            "buy_prices": buy_p,   "buy_cum": buy_cum}


# ---------------------------------------------------------------------------
# Grafikler
# ---------------------------------------------------------------------------

def plot_timeseries(metrics: List[dict], phase_boundaries: List[int], save_path: str):
    """4 panelli zaman serisi grafiği."""
    n = len(metrics)
    eps_arr     = np.arange(1, n + 1)
    price_arr   = np.array([m["avg_price"]  for m in metrics])
    traded_arr  = np.array([m["traded_kwh"] for m in metrics])
    unmet_arr   = np.array([m["unmet"]      for m in metrics])
    demand_arr  = np.array([m["demand"]     for m in metrics])
    supply_arr  = np.array([m["supply"]     for m in metrics])

    w = max(50, n // 80)   # hareketli ortalama penceresi

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle("Talep Şoku Deneyi — Zaman Serisi", fontsize=14, fontweight="bold")

    phase_colors = ["#d0e8ff", "#ffd6d6", "#d6ffd6"]
    phase_labels = ["Faz 1: Denge (×1.0)", "Faz 2: Talep Şoku (×2.0)", "Faz 3: Toparlanma (×1.0)"]
    boundaries   = [0] + phase_boundaries + [n]

    def add_phase_bg(ax):
        for i, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            ax.axvspan(a + 1, b, alpha=0.15, color=phase_colors[i % 3], label=phase_labels[i])
        for b in phase_boundaries:
            ax.axvline(b, color="red", linestyle="--", linewidth=1.2, alpha=0.7)

    # ── Panel 1: Ortalama takas fiyatı ──────────────────────────────────────
    ax = axes[0]
    ax.plot(eps_arr, price_arr, color="steelblue", alpha=0.3, linewidth=0.6)
    if len(price_arr) >= w:
        ma = _moving_avg(price_arr, w)
        ax.plot(np.arange(w, n + 1), ma, color="steelblue", linewidth=2.0, label=f"Ort. Fiyat (MA{w})")
    add_phase_bg(ax)
    ax.set_ylabel("Ort. Takas Fiyatı ($/kWh)")
    ax.set_title("Ortalama Takas Fiyatı")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Talep vs Arz ────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(eps_arr, demand_arr, color="tomato",   alpha=0.4, linewidth=0.6)
    ax.plot(eps_arr, supply_arr, color="seagreen", alpha=0.4, linewidth=0.6)
    if len(demand_arr) >= w:
        ax.plot(np.arange(w, n + 1), _moving_avg(demand_arr, w),
                color="tomato",   linewidth=2.0, label=f"Talep (MA{w})")
        ax.plot(np.arange(w, n + 1), _moving_avg(supply_arr, w),
                color="seagreen", linewidth=2.0, label=f"Arz (MA{w})")
    add_phase_bg(ax)
    ax.set_ylabel("Enerji (kWh/gün)")
    ax.set_title("Toplam Talep vs Toplam Arz")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── Panel 3: İşlem hacmi ─────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(eps_arr, traded_arr, color="darkorchid", alpha=0.3, linewidth=0.6)
    if len(traded_arr) >= w:
        ax.plot(np.arange(w, n + 1), _moving_avg(traded_arr, w),
                color="darkorchid", linewidth=2.0, label=f"P2P Ticaret (MA{w})")
    add_phase_bg(ax)
    ax.set_ylabel("Ticaret (kWh/gün)")
    ax.set_title("P2P İşlem Hacmi")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Karşılanmayan talep ─────────────────────────────────────────
    ax = axes[3]
    ax.plot(eps_arr, unmet_arr, color="orangered", alpha=0.3, linewidth=0.6)
    if len(unmet_arr) >= w:
        ax.plot(np.arange(w, n + 1), _moving_avg(unmet_arr, w),
                color="orangered", linewidth=2.0, label=f"Karşılanmayan (MA{w})")
    add_phase_bg(ax)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Karşılanmayan Talep (kWh)")
    ax.set_title("Karşılanmayan Talep")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Faz legend'ı sadece son panel'de
    patches = [mpatches.Patch(color=phase_colors[i], alpha=0.4, label=phase_labels[i])
               for i in range(3)]
    axes[3].legend(handles=patches + axes[3].get_lines(), fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Zaman serisi grafiği kaydedildi: {save_path}")


def plot_supply_demand(curve: Dict, save_path: str, title: str = "Denge Arz-Talep Eğrisi"):
    """Arz ve talep step-function grafiği."""
    fig, ax = plt.subplots(figsize=(9, 6))

    sp, sc = curve["sell_prices"], curve["sell_cum"]
    bp, bc = curve["buy_prices"],  curve["buy_cum"]

    # Step fonksiyon olarak çiz
    ax.step(sc, sp, color="seagreen",  linewidth=2.0, where="post", label="Arz (Satıcı ask'ları)")
    ax.step(bc, bp, color="tomato",    linewidth=2.0, where="post", label="Talep (Alıcı bid'leri)")

    # Denge noktasını bul (eğriler kesiştiğinde)
    # Basit yaklaşım: bid >= ask olan son noktayı ara
    eq_qty, eq_price = None, None
    for i, (bprice, bqty) in enumerate(zip(bp, bc)):
        for j, (sprice, sqty) in enumerate(zip(sp, sc)):
            if bprice >= sprice and bqty <= sqty:
                eq_qty   = min(bqty, sqty)
                eq_price = (bprice + sprice) / 2.0
                break
        if eq_qty is not None:
            break

    if eq_qty is not None:
        ax.axvline(eq_qty,   color="gray", linestyle=":", alpha=0.7)
        ax.axhline(eq_price, color="gray", linestyle=":", alpha=0.7)
        ax.scatter([eq_qty], [eq_price], color="black", zorder=5,
                   s=80, label=f"Denge: Q≈{eq_qty:.1f}kWh, P≈${eq_price:.3f}")

    ax.set_xlabel("Kümülatif Miktar (kWh)")
    ax.set_ylabel("Fiyat ($/kWh)")
    ax.set_title(title)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Arz-talep grafiği kaydedildi: {save_path}")


# ---------------------------------------------------------------------------
# Ana deney fonksiyonu
# ---------------------------------------------------------------------------

def run_demand_shock(
    phase1_ep: int = 6000,
    phase2_ep: int = 5000,
    phase3_ep: int = 5000,
    demand_mult: float = 2.0,
    epsilon_reset: float = 0.5,
    beta: float = 1.5,
    curtail_rate: float = 0.0,
    save_dir: str = "demand_shock_results",
):
    os.makedirs(save_dir, exist_ok=True)

    config = SimConfig(reward_beta=beta, reward_curtail_rate=curtail_rate)
    env = QLearningEnv(config, alpha_lr=0.1, gamma=0.95,
                       epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9985)

    print("=" * 60)
    print("  FAZ 1: DENGE — demand_multiplier=1.0")
    print(f"  {phase1_ep} episode")
    print("=" * 60)
    env.train(n_episodes=phase1_ep, verbose=True)

    # Faz 1 sonunda arz-talep eğrisi çıkar
    print("\n  Faz 1 denge eğrisi çıkarılıyor...")
    curve_eq = extract_supply_demand_curve(env)
    plot_supply_demand(curve_eq,
                       save_path=os.path.join(save_dir, "supply_demand_phase1.png"),
                       title="Faz 1 Denge — Arz-Talep Eğrisi")

    print("\n" + "=" * 60)
    print(f"  FAZ 2: TALEP ŞOKU — demand_multiplier={demand_mult:.1f}")
    print(f"  Epsilon {epsilon_reset:.2f}'e resetlendi")
    print(f"  {phase2_ep} episode")
    print("=" * 60)
    env.set_demand_multiplier(demand_mult)
    env.reset_epsilon(epsilon_reset)
    env.train(n_episodes=phase2_ep, verbose=True)

    # Faz 2 sonunda arz-talep eğrisi
    print("\n  Faz 2 şok denge eğrisi çıkarılıyor...")
    curve_shock = extract_supply_demand_curve(env)
    plot_supply_demand(curve_shock,
                       save_path=os.path.join(save_dir, "supply_demand_phase2.png"),
                       title=f"Faz 2 Şok Dengesi (demand×{demand_mult:.1f}) — Arz-Talep Eğrisi")

    print("\n" + "=" * 60)
    print("  FAZ 3: TOPARLANMA — demand_multiplier=1.0")
    print(f"  Epsilon {epsilon_reset:.2f}'e resetlendi")
    print(f"  {phase3_ep} episode")
    print("=" * 60)
    env.set_demand_multiplier(1.0)
    env.reset_epsilon(epsilon_reset)
    env.train(n_episodes=phase3_ep, verbose=True)

    # Tüm fazların time series verisi
    all_metrics = env.episode_metrics   # tüm fazlar burada birikmeli
    phase_boundaries = [phase1_ep, phase1_ep + phase2_ep]

    plot_timeseries(
        metrics=all_metrics,
        phase_boundaries=phase_boundaries,
        save_path=os.path.join(save_dir, "timeseries.png"),
    )

    # Faz 3 denge eğrisi
    print("\n  Faz 3 toparlanma eğrisi çıkarılıyor...")
    curve_recovery = extract_supply_demand_curve(env)
    plot_supply_demand(curve_recovery,
                       save_path=os.path.join(save_dir, "supply_demand_phase3.png"),
                       title="Faz 3 Toparlanma — Arz-Talep Eğrisi")

    # Özet
    n1 = phase1_ep
    n2 = phase1_ep + phase2_ep
    m1 = all_metrics[:n1]
    m2 = all_metrics[n1:n2]
    m3 = all_metrics[n2:]

    def avg(lst, key, last_n=200):
        vals = [m[key] for m in lst[-last_n:]]
        return float(np.mean(vals)) if vals else 0.0

    print("\n" + "=" * 60)
    print("  ÖZET (son 200 episode ortalaması)")
    print("=" * 60)
    for label, ms in [("Faz 1 (denge)", m1), ("Faz 2 (şok)", m2), ("Faz 3 (toparlanma)", m3)]:
        print(f"  {label}:")
        print(f"    Ort. fiyat  : ${avg(ms, 'avg_price'):.4f}/kWh")
        print(f"    Ticaret     : {avg(ms, 'traded_kwh'):.1f} kWh/gün")
        print(f"    Karşılanmayan: {avg(ms, 'unmet'):.1f} kWh/gün")
        print(f"    Talep       : {avg(ms, 'demand'):.1f} kWh/gün")
        print()

    print(f"Tüm grafikler kaydedildi: {save_dir}/")
    return env, all_metrics


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Talep Şoku Deneyi")
    parser.add_argument("--p1",      type=int,   default=6000, help="Faz 1 episode sayısı")
    parser.add_argument("--p2",      type=int,   default=5000, help="Faz 2 episode sayısı")
    parser.add_argument("--p3",      type=int,   default=5000, help="Faz 3 episode sayısı")
    parser.add_argument("--mult",    type=float, default=2.0,  help="Talep çarpanı (Faz 2)")
    parser.add_argument("--eps-reset", type=float, default=0.5, help="Faz geçişinde epsilon reset değeri")
    parser.add_argument("--save-dir", type=str,  default="demand_shock_results")
    args = parser.parse_args()

    run_demand_shock(
        phase1_ep=args.p1,
        phase2_ep=args.p2,
        phase3_ep=args.p3,
        demand_mult=args.mult,
        epsilon_reset=args.eps_reset,
        save_dir=args.save_dir,
    )
