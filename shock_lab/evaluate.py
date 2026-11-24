"""
Eğitilmiş tüm koşuları DONDURULMUŞ politikalarla (öğrenme kapalı, ε = 0 / NN ortalama
aksiyonu) değerlendirir.

    python -m shock_lab.evaluate --runs shock_lab/runs --out shock_lab/eval --specialists --workers 8

a) Rejim testi    : her test rejimi için K test günü, seed = 10_000_000 + k (eğitim
                    seed'lerinden ayrık; tüm varyant/rejimlerde AYNI günler).
b) Zaman çizelgesi: 60 ardışık test günü (seed = 20_000_000 + gün), 1–20 normal,
                    21–40 şoklu, 41–60 normal; her şok için.
c) --specialists  : her test rejimi × seed için YALNIZCA o rejimde sıfırdan eğitilen
                    q_blind tipi Q-Learning seti ("tam yeniden eğitilmiş" referans),
                    aynı K günde değerlendirilir.

Çıktılar (--out): eval_results.json, eval_summary.txt, fig_*.png.
Ara sonuçlar <out>/cache/ altında tutulur; yarıda kalan değerlendirme kaldığı yerden
devam eder (yapılandırma değiştiyse önbellek kullanılmaz).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from shock_lab.common import (
    chdir_repo_root, DEFAULT_TEST_REGIMES, DEFAULT_TIMELINE_SHOCKS, DEFAULT_TRAIN_REGIMES,
    TEST_SEED_BASE, TIMELINE_SEED_BASE, NORMAL, DAY_KEYS, VARIANT_ORDER, Regime,
    BalanceTracker, parse_regimes, rkey, run_day, log, write_json_atomic, fmt_hms,
)

METRICS_TABLE = [
    ("avg_price", "Avg. P2P clearing price ($/kWh)", "{:.4f}"),
    ("eta_pct", "Matching efficiency eta = P2P/E_max (%)", "{:.1f}"),
    ("buyer_cost", "Buyer cost ($/day)", "{:.1f}"),
    ("seller_revenue", "Seller revenue ($/day)", "{:.1f}"),
]
SPEC = "specialist"


# ---------------------------------------------------------------------------
# İşçi görevleri (ayrı süreçlerde çalışır)
# ---------------------------------------------------------------------------

def _worker_init() -> None:
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    chdir_repo_root()


def _eval_regimes(env, is_nn: bool, regimes: List[Regime], k_days: int, bal: BalanceTracker) -> Dict:
    out = {}
    for r in regimes:
        if is_nn:
            env.reset_price_history()
        days = {k: [] for k in DAY_KEYS}
        for k in range(k_days):
            day, viol = run_day(env, TEST_SEED_BASE + k, r, learn=False)
            bal.add(day, viol)
            for key in DAY_KEYS:
                days[key].append(day[key])
        out[rkey(r)] = {"regime": list(r), "days": days,
                        "mean": {key: float(np.mean(v)) for key, v in days.items()}}
    return out


def _no_grad(is_nn: bool):
    if is_nn:
        import torch
        return torch.no_grad()
    import contextlib
    return contextlib.nullcontext()


def task_eval_run(run_dir: str, cfg: Dict) -> Dict:
    from shock_lab.train import build_env, load_policy, prepare_torch
    from src.seeding import seed_everything
    with open(os.path.join(run_dir, "run.json"), encoding="utf-8") as f:
        data = json.load(f)
    meta = data["meta"]
    is_nn = meta["kind"] == "nn"
    if is_nn:
        prepare_torch()
    t0 = time.perf_counter()
    seed_everything(meta["seed"])
    env = build_env(meta)
    load_policy(env, meta, run_dir)
    env.set_greedy()
    bal = BalanceTracker()
    regimes = [tuple(r) for r in cfg["test_regimes"]]
    with _no_grad(is_nn):
        regime_test = _eval_regimes(env, is_nn, regimes, cfg["test_days"], bal)
        timeline = {}
        a, b, c = cfg["timeline_days"]
        for name, r in cfg["timeline_shocks"]:
            if is_nn:
                env.reset_price_history()
            sched = [NORMAL] * a + [tuple(r)] * b + [NORMAL] * c
            days = {k: [] for k in DAY_KEYS}
            for t, rr in enumerate(sched):
                day, viol = run_day(env, TIMELINE_SEED_BASE + t, rr, learn=False)
                bal.add(day, viol)
                for key in DAY_KEYS:
                    days[key].append(day[key])
            timeline[name] = {"regime": list(r), "days": days}
    return {"kind": "run", "name": data["name"], "variant": meta["variant"], "seed": meta["seed"],
            "train_regimes": meta["train_regimes"], "train_episodes": meta["episodes"],
            "regime_test": regime_test, "timeline": timeline, "balance_check": bal.as_dict(),
            "eval_wall_seconds": time.perf_counter() - t0, "cfg": cfg}


def task_specialist(regime: Regime, seed: int, producers: int, consumers: int,
                    episodes: int, cfg: Dict) -> Dict:
    from shock_lab.train import make_meta, train
    t0 = time.perf_counter()
    meta = make_meta("q_blind", seed, producers, consumers, episodes, [tuple(regime)], [1.0])
    meta["role"] = SPEC
    env, series, bal_train, timings = train(meta, progress=False)
    env.set_greedy()
    bal = BalanceTracker()
    res = _eval_regimes(env, False, [tuple(regime)], cfg["test_days"], bal)
    return {"kind": SPEC, "regime": list(regime), "seed": seed, "episodes": episodes,
            "eps_decay": meta["hyper"]["eps_decay"], "result": res[rkey(tuple(regime))],
            "balance_check": bal.as_dict(), "train_balance_check": bal_train,
            "train_wall_seconds": timings["train_wall_seconds"],
            "eval_wall_seconds": time.perf_counter() - t0 - timings["train_wall_seconds"],
            "cfg": cfg}


# ---------------------------------------------------------------------------
# Toplama
# ---------------------------------------------------------------------------

def _ms(vals: List[float]) -> Dict:
    v = np.asarray(vals, dtype=float)
    return {"mean": float(v.mean()) if len(v) else float("nan"),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}


def aggregate(runs: List[Dict], specs: List[Dict], test_regimes: List[Regime]) -> Dict:
    variants = [v for v in VARIANT_ORDER if any(r["variant"] == v for r in runs)]
    keys = [rkey(r) for r in test_regimes]
    nkey = rkey(NORMAL)
    agg: Dict = {"variants": variants, "regime_test": {}, "shock_response": {}, "timeline": {}}
    groups: Dict[str, Dict[int, Dict]] = {}
    for v in variants:
        groups[v] = {r["seed"]: r["regime_test"] for r in runs if r["variant"] == v}
    if specs:
        groups[SPEC] = {}
        for s in specs:
            groups[SPEC].setdefault(s["seed"], {})[rkey(tuple(s["regime"]))] = s["result"]
    for g, by_seed in groups.items():
        agg["regime_test"][g] = {}
        agg["shock_response"][g] = {}
        for k in keys:
            per_seed = {sd: rt[k]["mean"] for sd, rt in by_seed.items() if k in rt}
            if not per_seed:
                continue
            agg["regime_test"][g][k] = {m: _ms([d[m] for d in per_seed.values()]) for m in DAY_KEYS}
            deltas = [rt[k]["mean"]["avg_price"] - rt[nkey]["mean"]["avg_price"]
                      for rt in by_seed.values() if k in rt and nkey in rt]
            if deltas:
                agg["shock_response"][g][k] = _ms(deltas)
    # zaman çizelgesi: varyant × şok × metrik → gün başına seed ortalaması/std
    for v in variants:
        agg["timeline"][v] = {}
        rs = [r for r in runs if r["variant"] == v]
        for name in rs[0]["timeline"]:
            agg["timeline"][v][name] = {}
            for m in ("avg_price", "p2p_kwh", "eta_pct"):
                arr = np.array([r["timeline"][name]["days"][m] for r in rs])
                agg["timeline"][v][name][m] = {
                    "mean": arr.mean(axis=0).tolist(),
                    "std": (arr.std(axis=0, ddof=1) if len(rs) > 1 else np.zeros(arr.shape[1])).tolist(),
                    "n": len(rs)}
    return agg


def scores(agg: Dict, seen: List[str], unseen: List[str]) -> Dict:
    if SPEC not in agg["regime_test"]:
        return {}
    sp = agg["regime_test"][SPEC]
    sd = agg["shock_response"][SPEC]
    out = {}
    for v in agg["variants"]:
        rt = agg["regime_test"][v]
        sr = agg["shock_response"][v]
        row = {}
        for label, ks in (("seen", seen), ("unseen", unseen)):
            lv = [abs(rt[k]["avg_price"]["mean"] - sp[k]["avg_price"]["mean"])
                  for k in ks if k in rt and k in sp]
            dl = [abs(sr[k]["mean"] - sd[k]["mean"]) for k in ks
                  if k in sr and k in sd and k != rkey(NORMAL)]
            row[label] = {"price_mae": float(np.mean(lv)) if lv else float("nan"),
                          "delta_mae": float(np.mean(dl)) if dl else float("nan"),
                          "n_regimes": len(lv)}
        out[v] = row
    return out


# ---------------------------------------------------------------------------
# Özet metni
# ---------------------------------------------------------------------------

def _cell(st: Optional[Dict], fmt: str) -> str:
    if not st:
        return "-"
    if st["n"] > 1:
        return f"{fmt.format(st['mean'])}±{fmt.replace('+', '').format(st['std'])}"
    return fmt.format(st["mean"])


def format_summary(res: Dict) -> str:
    agg = res["aggregate"]
    groups = list(agg["variants"]) + ([SPEC] if SPEC in agg["regime_test"] else [])
    seen, unseen = res["seen_keys"], res["unseen_keys"]
    cw = 17
    width = 18 + (cw + 1) * len(groups)
    L: List[str] = []
    L.append("=" * width)
    L.append("SHOCK LAB — frozen policies under supply/demand regimes (no retraining)")
    L.append("=" * width)
    c = res["config"]
    L.append(f"  population {c['producers']}+{c['consumers']} | test days per regime {c['test_days']} "
             f"(seeds {TEST_SEED_BASE}..{TEST_SEED_BASE + c['test_days'] - 1}) | timeline "
             f"{'/'.join(map(str, c['timeline_days']))} days (seeds {TIMELINE_SEED_BASE}+)")
    L.append(f"  regime (d, s): consumer load x d, producer PV x s | seen = randomized training grid "
             f"{', '.join(seen)}")
    L.append("  NOTE: *_normal variants were trained ONLY on d=1:s=1; for them every other regime is unseen.")
    L.append("  cells: mean ± std over seeds (each seed = mean over test days)")
    for g in groups:
        n_seeds = max((v[rkey(NORMAL)]["avg_price"]["n"] for v in [agg["regime_test"][g]]
                       if rkey(NORMAL) in v), default=0)
        extra = ""
        if g == SPEC:
            extra = f" (q_blind-type, trained only on that regime, {c.get('specialist_episodes')} ep)"
        L.append(f"    {g:<11} seeds={n_seeds}{extra}")
    head = f"  {'regime':<16}" + "".join(f"{g:>{cw}} " for g in groups)

    def block(metric: str, fmt: str, keys: List[str], title: str) -> None:
        L.append(f"  -- {title} --")
        for k in keys:
            L.append(f"  {k:<16}" + "".join(
                f"{_cell(agg['regime_test'][g].get(k, {}).get(metric), fmt):>{cw}} " for g in groups))

    for metric, label, fmt in METRICS_TABLE:
        L.append("")
        L.append("-" * width)
        L.append(f"{label}")
        L.append("-" * width)
        L.append(head)
        block(metric, fmt, seen, "seen regimes")
        block(metric, fmt, unseen, "unseen regimes")

    L.append("")
    L.append("-" * width)
    L.append("SHOCK RESPONSE: change in avg. price vs. the normal regime d=1:s=1 ($/kWh), per seed then averaged")
    L.append("  specialist column = what a fully retrained agent does for that regime")
    L.append("-" * width)
    L.append(head)
    for title, ks in (("seen regimes", seen), ("unseen regimes", unseen)):
        L.append(f"  -- {title} --")
        for k in ks:
            if k == rkey(NORMAL):
                continue
            L.append(f"  {k:<16}" + "".join(
                f"{_cell(agg['shock_response'][g].get(k), '{:+.4f}'):>{cw}} " for g in groups))

    sc = res.get("scores") or {}
    if sc:
        L.append("")
        L.append("-" * width)
        L.append("SCORE vs. specialist (lower = closer to a fully retrained agent), $/kWh")
        L.append("  price_mae = mean |price_variant - price_specialist| over regimes")
        L.append("  delta_mae = mean |Δprice_variant - Δprice_specialist| (Δ vs. normal regime)")
        L.append("-" * width)
        L.append(f"  {'variant':<12}{'seen price_mae':>16}{'seen delta_mae':>16}"
                 f"{'unseen price_mae':>18}{'unseen delta_mae':>18}")
        for v, row in sc.items():
            L.append(f"  {v:<12}{row['seen']['price_mae']:>16.4f}{row['seen']['delta_mae']:>16.4f}"
                     f"{row['unseen']['price_mae']:>18.4f}{row['unseen']['delta_mae']:>18.4f}")
    else:
        L.append("")
        L.append("  (no specialists run → no score; use --specialists)")

    L.append("")
    L.append("-" * width)
    L.append("TIMELINE: mean price per phase (days averaged within phase, then over seeds)")
    L.append("-" * width)
    a, b, cc = c["timeline_days"]
    L.append(f"  {'variant':<12}{'shock':<15}{'before':>10}{'during':>10}{'after':>10}{'Δ during':>11}")
    for v in agg["variants"]:
        for name, tl in agg["timeline"][v].items():
            m = np.asarray(tl["avg_price"]["mean"])
            p1, p2, p3 = m[:a].mean(), m[a:a + b].mean(), m[a + b:].mean()
            L.append(f"  {v:<12}{name:<15}{p1:>10.4f}{p2:>10.4f}{p3:>10.4f}{p2 - p1:>+11.4f}")

    L.append("")
    L.append("-" * width)
    bal = res["balance_check"]
    L.append(f"ENERGY BALANCE CHECK (|D - P2P - grid buy|, |S - P2P - grid sell| <= {bal['rel_tol']:g}·max(1,·)):")
    L.append(f"  evaluation days {bal['n_days']} | violations {bal['n_violations']} | max dev "
             f"{bal['max_abs_D_minus_p2p_minus_gridbuy']:.2e} / {bal['max_abs_S_minus_p2p_minus_gridsell']:.2e} kWh")
    tb = res.get("train_balance_check")
    if tb:
        L.append(f"  training days {tb['n_days']} (runs + specialists) | violations {tb['n_violations']}")
    L.append("=" * width)
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Grafikler
# ---------------------------------------------------------------------------

COLORS = {"q_normal": "#9ecae1", "q_blind": "#4292c6", "q_price": "#08519c", "q_oracle": "#31a354",
          "nn_normal": "#fdae6b", "nn_price": "#e6550d", "nn_oracle": "#a63603"}


def _bar_figure(res: Dict, metric: str, ylabel: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    agg = res["aggregate"]
    variants = agg["variants"]
    has_spec = SPEC in agg["regime_test"]
    rows = [("Seen regimes (training grid)", res["seen_keys"]),
            ("Unseen regimes", res["unseen_keys"])]
    rows = [r for r in rows if r[1]]
    fig, axes = plt.subplots(len(rows), 1, figsize=(max(10, 1.25 * max(len(k) for _, k in rows) + 3),
                                                    4.2 * len(rows)), squeeze=False)
    nv = len(variants)
    w = 0.8 / max(1, nv)
    for ax, (title, keys) in zip(axes[:, 0], rows):
        x = np.arange(len(keys))
        for j, v in enumerate(variants):
            rt = agg["regime_test"][v]
            m = [rt[k][metric]["mean"] if k in rt else np.nan for k in keys]
            s = [rt[k][metric]["std"] if k in rt else 0 for k in keys]
            ax.bar(x - 0.4 + w * (j + 0.5), m, w, yerr=s, color=COLORS.get(v), label=v,
                   edgecolor="white", linewidth=0.4, error_kw=dict(lw=0.7, capsize=1.5))
        if has_spec:
            sp = agg["regime_test"][SPEC]
            for i, k in enumerate(keys):
                if k in sp:
                    ax.hlines(sp[k][metric]["mean"], i - 0.42, i + 0.42, colors="black",
                              linestyles="--", linewidth=1.3,
                              label="specialist (retrained on regime)" if i == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace(":", "\n") for k in keys], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        sts = [agg["regime_test"][g][k][metric] for g in agg["regime_test"]
               for k in keys if k in agg["regime_test"][g]]
        if sts and metric == "avg_price":
            lo = min(st["mean"] - st["std"] for st in sts)
            hi = max(st["mean"] + st["std"] for st in sts)
            pad = 0.15 * (hi - lo + 1e-3)
            ax.set_ylim(max(0.0, lo - pad), hi + pad)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 8), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _timeline_figure(res: Dict, shock: str, metric: str, ylabel: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    agg = res["aggregate"]
    a, b, c = res["config"]["timeline_days"]
    n = a + b + c
    days = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axvspan(a + 0.5, a + b + 0.5, color="red", alpha=0.07, lw=0)
    regime = None
    for v in agg["variants"]:
        tl = agg["timeline"][v].get(shock)
        if not tl:
            continue
        m = np.asarray(tl[metric]["mean"])
        s = np.asarray(tl[metric]["std"])
        ls = "-" if v.startswith("q_") else "--"
        ax.plot(days, m, ls, color=COLORS.get(v), lw=1.5, label=v)
        ax.fill_between(days, m - s, m + s, color=COLORS.get(v), alpha=0.10, lw=0)
    # specialist düzeyleri (yalnızca fiyat): normal fazlarda (1,1), şok fazında şok rejimi
    sh_regime = None
    for name, r in res["config"]["timeline_shocks"]:
        if name == shock:
            sh_regime = tuple(r)
    if metric == "avg_price" and SPEC in agg["regime_test"] and sh_regime is not None:
        sp = agg["regime_test"][SPEC]
        kn, ks = rkey(NORMAL), rkey(sh_regime)
        if kn in sp and ks in sp:
            yn, ys = sp[kn]["avg_price"]["mean"], sp[ks]["avg_price"]["mean"]
            xs = [0.5, a + 0.5, a + 0.5, a + b + 0.5, a + b + 0.5, n + 0.5]
            ys_ = [yn, yn, ys, ys, yn, yn]
            ax.plot(xs, ys_, color="black", lw=1.2, ls=":", label="specialist level (retrained)")
    ax.set_xlim(0.5, n + 0.5)
    ax.set_xlabel("Test day")
    ax.set_ylabel(ylabel)
    title_r = f" (d={sh_regime[0]:g}, s={sh_regime[1]:g})" if sh_regime else ""
    ax.set_title(f"Frozen policies, shock '{shock}'{title_r} on days {a + 1}-{a + b}"
                 " (mean ± std over seeds)", fontsize=10, pad=16)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    tr = ax.get_xaxis_transform()   # x: veri, y: eksen kesri → başlıkla çakışmaz
    ax.text(a / 2 + 0.5, 1.005, "normal", transform=tr, ha="center", va="bottom", fontsize=8)
    ax.text(a + b / 2 + 0.5, 1.005, "shock", transform=tr, ha="center", va="bottom", fontsize=8,
            color="darkred")
    ax.text(a + b + c / 2 + 0.5, 1.005, "normal", transform=tr, ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_figures(res: Dict, out_dir: str) -> List[str]:
    paths = []
    p = os.path.join(out_dir, "fig_price_by_regime.png")
    _bar_figure(res, "avg_price", "Avg. P2P price ($/kWh)", p); paths.append(p)
    p = os.path.join(out_dir, "fig_eta_by_regime.png")
    _bar_figure(res, "eta_pct", "Matching efficiency η (%)", p); paths.append(p)
    for name, _ in res["config"]["timeline_shocks"]:
        safe = name.replace(".", "p")
        p = os.path.join(out_dir, f"fig_timeline_price_{safe}.png")
        _timeline_figure(res, name, "avg_price", "Avg. P2P price ($/kWh)", p); paths.append(p)
        p = os.path.join(out_dir, f"fig_timeline_p2p_{safe}.png")
        _timeline_figure(res, name, "p2p_kwh", "P2P volume (kWh/day)", p); paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------

def parse_shocks(text: str) -> List[Tuple[str, Regime]]:
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        name, r = tok.split("=")
        a, b = r.split(":")
        out.append((name.strip(), (float(a), float(b))))
    return out


def discover_runs(runs_dir: str) -> List[str]:
    if not os.path.isdir(runs_dir):
        return []
    out = []
    for d in sorted(os.listdir(runs_dir)):
        p = os.path.join(runs_dir, d)
        if os.path.isfile(os.path.join(p, "run.json")):
            out.append(p)
    return out


def _cache_load(path: str, cfg: Dict) -> Optional[Dict]:
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d if d.get("cfg") == cfg else None
    except (OSError, ValueError):
        return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="shock_lab: dondurulmuş politikaların şok testi")
    p.add_argument("--runs", default="shock_lab/runs")
    p.add_argument("--out", default="shock_lab/eval")
    p.add_argument("--test-days", type=int, default=30)
    p.add_argument("--test-regimes", default=None, help="'d:s,...' (varsayılan: 9 görülen + 8 görülmemiş)")
    p.add_argument("--timeline-shocks", default=None,
                   help="'ad=d:s,...' (varsayılan: demand_x2=2:1,supply_x2=1:2,demand_x1.5=1.5:1,supply_x0.75=1:0.75)")
    p.add_argument("--timeline-days", default="20,20,20", help="normal,şok,normal gün sayıları")
    p.add_argument("--specialists", action="store_true", help="rejim uzmanı referanslarını da eğit")
    p.add_argument("--specialist-episodes", type=int, default=5000)
    p.add_argument("--specialist-seeds", default=None, help="'1,2,3' (varsayılan: koşulardaki seed'ler)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--plot-only", action="store_true", help="eval_results.json'dan yalnızca özet+grafik")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    runs_dir, out_dir = os.path.abspath(args.runs), os.path.abspath(args.out)
    chdir_repo_root()
    os.makedirs(out_dir, exist_ok=True)
    res_path = os.path.join(out_dir, "eval_results.json")

    if args.plot_only:
        with open(res_path, encoding="utf-8") as f:
            res = json.load(f)
        _finish(res, out_dir)
        return 0

    run_dirs = discover_runs(runs_dir)
    if not run_dirs:
        log(f"HATA: {runs_dir} altında run.json bulunamadı")
        return 1
    metas = []
    for d in run_dirs:
        with open(os.path.join(d, "run.json"), encoding="utf-8") as f:
            metas.append(json.load(f)["meta"])
    pops = {(m["producers"], m["consumers"]) for m in metas}
    if len(pops) != 1:
        log(f"HATA: koşular farklı popülasyonlara ait: {pops}")
        return 1
    producers, consumers = pops.pop()
    test_regimes = parse_regimes(args.test_regimes) if args.test_regimes else list(DEFAULT_TEST_REGIMES)
    if rkey(NORMAL) not in [rkey(r) for r in test_regimes]:
        test_regimes = [NORMAL] + test_regimes
    shocks = parse_shocks(args.timeline_shocks) if args.timeline_shocks else list(DEFAULT_TIMELINE_SHOCKS)
    tl_days = [int(x) for x in args.timeline_days.split(",")]
    assert len(tl_days) == 3
    seeds = sorted({m["seed"] for m in metas})
    spec_seeds = [int(x) for x in args.specialist_seeds.split(",")] if args.specialist_seeds else seeds

    # Görülen rejimler = rastgeleleştirilmiş varyantların eğitim ızgarası (birleşim)
    grid = {rkey(tuple(r)) for m in metas if m["randomized"] for r in m["train_regimes"]}
    if not grid:
        grid = {rkey(r) for r in DEFAULT_TRAIN_REGIMES}
    keys = [rkey(r) for r in test_regimes]
    seen = [k for k in keys if k in grid]
    unseen = [k for k in keys if k not in grid]

    cfg = {"test_regimes": [list(r) for r in test_regimes], "test_days": args.test_days,
           "timeline_shocks": [[n, list(r)] for n, r in shocks], "timeline_days": tl_days}
    max_train_seed = max(m["seed"] + m["episodes"] for m in metas)
    if args.specialists:
        max_train_seed = max(max_train_seed, max(spec_seeds) + args.specialist_episodes)
    assert max_train_seed < TEST_SEED_BASE, "eğitim ve test seed'leri çakışıyor"

    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    tasks = []   # (cost, cache_path, fn, args)
    for d, m in zip(run_dirs, metas):
        cost = (8.0 if m["kind"] == "nn" else 1.0) * (len(test_regimes) * args.test_days + len(shocks) * sum(tl_days))
        tasks.append((cost, os.path.join(cache_dir, f"run_{os.path.basename(d)}.json"),
                      task_eval_run, (d, cfg), f"değerlendirme {os.path.basename(d)}"))
    spec_cfg = dict(cfg, specialist_episodes=args.specialist_episodes)
    if args.specialists:
        for r in test_regimes:
            for sd in spec_seeds:
                tasks.append((1.5 * args.specialist_episodes + args.test_days,
                              os.path.join(cache_dir, f"spec_{rkey(r).replace(':', '_')}_seed{sd}.json"),
                              task_specialist, (r, sd, producers, consumers, args.specialist_episodes, spec_cfg),
                              f"uzman {rkey(r)} seed {sd}"))
    tasks.sort(key=lambda t: -t[0])

    results: List[Dict] = []
    todo = []
    for t in tasks:
        cached = _cache_load(t[1], t[3][-1])
        if cached is not None:
            results.append(cached)
        else:
            todo.append(t)
    log(f"DEĞERLENDİRME: {len(run_dirs)} koşu, {len(test_regimes)} test rejimi × {args.test_days} gün, "
        f"{len(shocks)} zaman çizelgesi | uzman görevleri {len(tasks) - len(run_dirs)} | "
        f"önbellekten {len(results)}, çalışacak {len(todo)} | workers={args.workers}")
    t0 = time.perf_counter()
    failed = []
    if todo:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max(1, args.workers), mp_context=ctx,
                                 initializer=_worker_init) as ex:
            futs = {ex.submit(t[2], *t[3]): t for t in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                t = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:   # noqa: BLE001
                    failed.append(t[4])
                    log(f"  [{i}/{len(todo)}] HATA {t[4]}: {e!r}")
                    continue
                write_json_atomic(r, t[1])
                results.append(r)
                log(f"  [{i}/{len(todo)}] bitti {t[4]} | geçen {fmt_hms(time.perf_counter() - t0)}")
    if failed:
        log(f"HATA: {len(failed)} görev başarısız: {failed}")
        return 1

    runs = sorted([r for r in results if r["kind"] == "run"], key=lambda r: r["name"])
    specs = sorted([r for r in results if r["kind"] == SPEC], key=lambda r: (rkey(tuple(r["regime"])), r["seed"]))
    bal = BalanceTracker()
    for r in runs + specs:
        bal.merge(r["balance_check"])
    tbal = BalanceTracker()
    for d in run_dirs:
        with open(os.path.join(d, "run.json"), encoding="utf-8") as f:
            tbal.merge(json.load(f)["balance_check"])
    for s in specs:
        tbal.merge(s["train_balance_check"])
    agg = aggregate(runs, specs, test_regimes)
    res = {
        "format": "shock-lab-eval/1",
        "config": dict(cfg, producers=producers, consumers=consumers,
                       specialists=bool(args.specialists),
                       specialist_episodes=args.specialist_episodes if args.specialists else None,
                       specialist_seeds=spec_seeds if args.specialists else [],
                       test_seed_base=TEST_SEED_BASE, timeline_seed_base=TIMELINE_SEED_BASE,
                       max_train_day_seed=max_train_seed - 1),
        "seen_keys": seen, "unseen_keys": unseen,
        "runs": {r["name"]: {k: v for k, v in r.items() if k != "cfg"} for r in runs},
        "specialists": [{k: v for k, v in s.items() if k != "cfg"} for s in specs],
        "aggregate": agg,
        "scores": scores(agg, seen, unseen),
        "balance_check": bal.as_dict(),
        "train_balance_check": tbal.as_dict(),
        "hostname": socket.gethostname(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "eval_wall_seconds": time.perf_counter() - t0,
        "argv": sys.argv[1:] if argv is None else list(argv),
    }
    write_json_atomic(res, res_path)
    _finish(res, out_dir)
    log(f"bitti | {fmt_hms(time.perf_counter() - t0)} | çıktı: {out_dir}")
    return 0


def _finish(res: Dict, out_dir: str) -> None:
    text = format_summary(res)
    with open(os.path.join(out_dir, "eval_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    for p in make_figures(res, out_dir):
        log(f"grafik: {p}")


if __name__ == "__main__":
    sys.exit(main())
