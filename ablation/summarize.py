"""
Ablasyon sonuçlarını tablolara ve grafiklere döker.

Kullanım (repo kökünden):
    python -m ablation.summarize                       # ablation/results
    python -m ablation.summarize ablation/results_smoke

Çıktılar (aynı dizine):
    ablation_summary.txt      İngilizce kutu tablolar (makale için) — ekrana da basılır
    ablation_summary.json     aynı sayılar, makine okunur
    fig_ablation_eta.png      η: varyant başına ortalama ± std (+ seed noktaları), popülasyon başına panel
    fig_ablation_price.png    ortalama takas fiyatı, aynı düzen
    fig_ablation_learning.png eğitim eğrileri (seed ortalaması, kayan ortalama N/50); Q ve NN ayrı panel

std: seed'ler arası örneklem std (ddof=1; tek seed'de 0). Eşleştirilmiş farklar aynı
seed'ler üzerinden (a − b) hesaplanır. Metrik anahtarları src özetiyle aynıdır.
Yeniden üretim kontrolü: results/A_p<P>c<C>_seed<S>.json varsa q_base / nn_full
metrikleri oradaki ql / nn metrikleriyle karşılaştırılır (yalnızca rapor; hata vermez).

Ausgrid sonuçları (JSON'da profile_source = "ausgrid"; ör. ablation/results_ausgrid):
başlıkta profil kaynağı ve 73 senaryoluk değerlendirme belirtilir; yeniden üretim
kontrolü results_ausgrid/A_*.json ile yapılır (yoksa açık bir mesajla atlanır).
Profil kaynağı alanı olmayan eski JSON'lar sentetik sayılır. Bir dizinde iki kaynak
karışıksa uyarı basılır.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from ablation.variants import VARIANTS, VARIANT_ORDER, CROSS_PAIRS, REPO_ROOT, chdir_repo_root

chdir_repo_root()

from src.summarize_experiments import box_table, _style   # noqa: E402

# (anahtar, başlık, format) — src METRIC_ROWS anahtarları
METRICS = [
    ("efficiency_pct", "eta (%)", "{:.2f}"),
    ("p2p_kwh", "P2P (kWh)", "{:.1f}"),
    ("avg_price", "price ($/kWh)", "{:.4f}"),
    ("buyer_cost", "buyer cost ($)", "{:.2f}"),
    ("seller_revenue", "seller rev. ($)", "{:.2f}"),
]
DIFF_KEYS = ["efficiency_pct", "avg_price", "p2p_kwh", "buyer_cost", "seller_revenue"]

SHORT_INPUTS = {
    "q_base": "hour, qty",
    "q_price3": "hour, qty, last price (3+1 bins)",
    "q_price5": "hour, qty, last price (5+1 bins)",
    "q_price3_x2": "hour, qty, last price (3+1 bins)",
    "q_price5_x2": "hour, qty, last price (5+1 bins)",
    "nn_full": "hour, qty, last price, 12-price history",
    "nn_nohist": "hour, qty, last price",
    "nn_noprice": "hour, qty",
}
# Renk varlığı izler (her grafikte aynı); ×2 varyantlar aynı renk + tarama / kesikli çizgi.
COLORS = {"q_base": "#eb6834", "q_price3": "#2a78d6", "q_price5": "#6250d6",
          "q_price3_x2": "#2a78d6", "q_price5_x2": "#6250d6",
          "nn_full": "#1baf7a", "nn_nohist": "#eda100", "nn_noprice": "#e87ba4"}
POP_ORDER = ["4+4", "10+10", "20+20", "10+20"]


# ---------------------------------------------------------------------------
# Yükleme
# ---------------------------------------------------------------------------

def load_runs(d: str) -> List[Dict]:
    runs = []
    for path in sorted(glob.glob(os.path.join(d, "*.json"))):
        if os.path.basename(path).startswith("ablation_summary"):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                r = json.load(f)
        except (OSError, ValueError) as e:
            print(f"  [WARN] cannot read {path}: {e}")
            continue
        if str(r.get("format", "")).startswith("energy-trading-ablation"):
            runs.append(r)
    return runs


def profile_source(r: Dict) -> str:
    """Koşunun profil kaynağı; alanı olmayan (eski) JSON'lar sentetiktir."""
    return r.get("profile_source") or r.get("sim_config", {}).get("profile_source") or "synthetic"


def ref_dir(source: str) -> str:
    """Yeniden üretim kontrolü için src sonuç dizini (göreli ad)."""
    return "results_ausgrid" if source == "ausgrid" else "results"


def group(runs: List[Dict]) -> Dict[str, Dict[str, Dict[int, Dict]]]:
    """pop → varyant → seed → run"""
    g: Dict[str, Dict[str, Dict[int, Dict]]] = {}
    for r in runs:
        g.setdefault(r["population"]["label"], {}).setdefault(r["variant"], {})[int(r["seed"])] = r
    return g


def pops_sorted(g) -> List[str]:
    return [p for p in POP_ORDER if p in g] + sorted(set(g) - set(POP_ORDER))


def _ms(v) -> Tuple[float, float]:
    v = np.asarray(v, dtype=float)
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)


def metric(r: Dict, key: str) -> float:
    if key == "train_seconds":
        return float(r["result"]["train_seconds"])
    return float(r["result"]["metrics"][key])


def dims_str(r: Dict) -> str:
    a = r["architecture"]
    if a["kind"] == "tabular":
        return f"{a['n_states_per_table']} states"
    return f"in={a['head_input_dim']}, {a['n_params_per_net']} par"


# ---------------------------------------------------------------------------
# Hesaplamalar
# ---------------------------------------------------------------------------

def variant_stats(by_seed: Dict[int, Dict]) -> Dict:
    seeds = sorted(by_seed)
    any_r = by_seed[seeds[0]]
    out = {"seeds": seeds, "n": len(seeds), "episodes": any_r["episodes"],
           "architecture": any_r["architecture"], "inputs": any_r["inputs"]}
    for key in [k for k, _, _ in METRICS] + ["train_seconds"]:
        vals = [metric(by_seed[s], key) for s in seeds]
        m, sd = _ms(vals)
        out[key] = {"mean": m, "std": sd, "per_seed": dict(zip(map(str, seeds), vals))}
    return out


def paired(a: Dict[int, Dict], b: Dict[int, Dict]) -> Optional[Dict]:
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    out = {"seeds": seeds, "n": len(seeds)}
    for key in DIFF_KEYS:
        d = [metric(a[s], key) - metric(b[s], key) for s in seeds]
        m, sd = _ms(d)
        out[key] = {"mean": m, "std": sd, "per_seed": dict(zip(map(str, seeds), d))}
    return out


def reproduction(g_pop: Dict[str, Dict[int, Dict]], pop: str, source: str = "synthetic") -> List[Dict]:
    rows = []
    p, c = pop.split("+")
    for variant, method in (("q_base", "ql"), ("nn_full", "nn")):
        for s, r in sorted(g_pop.get(variant, {}).items()):
            path = os.path.join(REPO_ROOT, ref_dir(source), f"A_p{p}c{c}_seed{s}.json")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                ref = json.load(f)
            if ref.get("profile_source", "synthetic") != source:
                continue                       # farklı profil kaynağı: karşılaştırılamaz
            res = ref.get("results", {}).get(method)
            if res is None:
                continue
            same_budget = int(res.get("episodes", -1)) == int(r["episodes"])
            mine = r["result"]["metrics"]
            keys = sorted(set(mine) & set(res["metrics"]))
            mx = max(abs(float(mine[k]) - float(res["metrics"][k])) for k in keys)
            rows.append({"variant": variant, "seed": s, "src_file": os.path.relpath(path, REPO_ROOT),
                         "same_budget": same_budget, "episodes": r["episodes"],
                         "src_episodes": res.get("episodes"), "max_abs_diff": mx,
                         "eta": float(mine["efficiency_pct"]), "src_eta": float(res["metrics"]["efficiency_pct"]),
                         "src_host": ref.get("hostname"), "host": r.get("hostname"),
                         "train_curve_identical": r["result"]["train_curve"] == res.get("train_curve")})
    return rows


# ---------------------------------------------------------------------------
# Metin
# ---------------------------------------------------------------------------

def _pm(d: Dict, fmt: str) -> str:
    return f"{fmt.format(d['mean'])} ± {fmt.format(d['std'])}"


def _spm(d: Dict, fmt: str) -> str:
    return f"{fmt.format(d['mean']).replace('+-', '-')} ± {fmt.format(d['std']).lstrip('+')}"


def build_text(g, d: str) -> Tuple[str, Dict]:
    out_json: Dict = {"source_dir": os.path.abspath(d), "populations": {}}
    all_runs = [r for pg in g.values() for vg in pg.values() for r in vg.values()]
    hosts = sorted({r.get("hostname", "?") for r in all_runs})
    sources = sorted({profile_source(r) for r in all_runs})
    source = "ausgrid" if sources == ["ausgrid"] else "synthetic"
    head = [
        f"INPUT-ABLATION SUMMARY — {os.path.abspath(d)}",
        f"  Host(s): {', '.join(hosts)}",
        "  Cells: mean ± sample std over seeds (ddof=1). Paired Δ = variant − reference on the same seed."]
    if source == "ausgrid":
        out_json["profile_source"] = "ausgrid"
        split_ids = sorted({str(r.get("ausgrid", {}).get("split_id", "?")) for r in all_runs})
        n_sc = sorted({len(r.get("eval_scenarios", {}).get("eval_seed", [])) for r in all_runs})
        head += [
            f"  Profiles: Ausgrid real household data (split_id {', '.join(split_ids)}); training days = "
            "Ausgrid train scenarios (train homes × train days).",
            "  Every variant: seed_everything(seed) → env.train(N) → greedy evaluation on all "
            f"{'/'.join(map(str, n_sc))} Ausgrid test scenarios (identical to src/experiment.py, "
            "--profiles ausgrid); energies / $ summed over scenarios, η = 100·ΣP2P/ΣE_max, "
            "price volume-weighted."]
    else:
        head.append("  Every variant: seed_everything(seed) → env.train(N) → env.run_evaluation(seed) "
                    "(identical to src/experiment.py); evaluation day = seed.")
    if len(sources) > 1:
        head.append(f"  WARNING: mixed profile sources in this directory ({', '.join(sources)}); "
                    "tables pool them — use separate directories.")
        out_json["profile_source"] = "mixed"
    sec: List[str] = ["\n".join(head)]

    for pop in pops_sorted(g):
        gp = g[pop]
        present = [v for v in VARIANT_ORDER if v in gp]
        stats = {v: variant_stats(gp[v]) for v in present}
        pj: Dict = {"variants": stats, "paired_vs_reference": {}, "cross_family": {}, "reproduction": []}

        # Tablo 1
        header = ["Variant", "Inputs", "States / input", "Episodes", "n"] + [h for _, h, _ in METRICS] + ["train (s)"]
        rows = []
        for v in present:
            st, r0 = stats[v], next(iter(gp[v].values()))
            rows.append([v, SHORT_INPUTS.get(v, ""), dims_str(r0), str(st["episodes"]), str(st["n"])]
                        + [_pm(st[k], f) for k, _, f in METRICS] + [_pm(st["train_seconds"], "{:.0f}")])
        sec += [f"### POPULATION {pop} (producers + consumers)",
                box_table(f"Table A1 — Input ablation, population {pop}", header, rows,
                          ["States = discrete states per Q-table (seller and buyer tables each); "
                           "in = policy-head input dimension, par = parameters per policy net "
                           "(two nets per agent: seller, buyer).",
                           "q_*_x2 = same as q_* with twice the training episodes (budget control)."])]

        # Tablo 2: aile içi eşleştirilmiş farklar
        rows = []
        for v in present:
            ref = VARIANTS[v]["ref"]
            if v == ref or ref not in gp:
                continue
            pr = paired(gp[v], gp[ref])
            if pr is None:
                continue
            pj["paired_vs_reference"][v] = dict(pr, reference=ref)
            rows.append([f"{v} − {ref}", str(pr["n"]),
                         _spm(pr["efficiency_pct"], "{:+.2f}"), _spm(pr["avg_price"], "{:+.4f}"),
                         _spm(pr["p2p_kwh"], "{:+.1f}"), _spm(pr["buyer_cost"], "{:+.2f}"),
                         _spm(pr["seller_revenue"], "{:+.2f}")])
        if rows:
            sec.append(box_table(f"Table A2 — Paired differences vs. family reference, {pop}",
                                 ["Comparison", "n", "Δη (pp)", "Δprice ($/kWh)", "ΔP2P (kWh)",
                                  "Δbuyer cost ($)", "Δseller rev. ($)"], rows))

        # Tablo 3: aileler arası
        rows, notes = [], []
        for i, (a, b) in enumerate(CROSS_PAIRS):
            if a not in gp or b not in gp:
                continue
            pr = paired(gp[a], gp[b])
            if pr is None:
                continue
            pj["cross_family"][f"{a} - {b}"] = pr
            tag = " [HEADLINE: identical information]" if i == 0 else ""
            rows.append([f"{a} − {b}{tag}", str(pr["n"]),
                         _spm(pr["efficiency_pct"], "{:+.2f}"), _spm(pr["avg_price"], "{:+.4f}"),
                         _spm(pr["p2p_kwh"], "{:+.1f}"), _spm(pr["buyer_cost"], "{:+.2f}"),
                         _spm(pr["seller_revenue"], "{:+.2f}")])
        if "nn_full - q_base" in pj["cross_family"] and "nn_noprice - q_base" in pj["cross_family"]:
            full = pj["cross_family"]["nn_full - q_base"]["efficiency_pct"]["mean"]
            nop = pj["cross_family"]["nn_noprice - q_base"]["efficiency_pct"]["mean"]
            if abs(full) > 1e-9:
                frac = nop / full
                pj["gap_retained_without_price"] = frac
                notes.append(f"Share of the original NN − Q efficiency gap ({full:+.2f} pp) that remains when "
                             f"the NN receives no price input: {nop:+.2f} pp = {frac * 100:.0f}%.")
        notes.append("nn_noprice and q_base observe exactly the same quantities (time of day, own net "
                     "quantity); they differ only in the learning algorithm and its representation.")
        notes.append("nn_nohist − q_price5: both observe the previous-slot clearing price (continuous vs. binned).")
        if rows:
            sec.append(box_table(f"Table A3 — Cross-family paired differences (a − b), {pop}",
                                 ["Comparison", "n", "Δη (pp)", "Δprice ($/kWh)", "ΔP2P (kWh)",
                                  "Δbuyer cost ($)", "Δseller rev. ($)"], rows, notes))

        # Tablo 4: seed başına η
        seeds = sorted({s for v in present for s in gp[v]})
        rows = [[v] + [f"{metric(gp[v][s], 'efficiency_pct'):.2f}" if s in gp[v] else "-" for s in seeds]
                for v in present]
        sec.append(box_table(f"Table A4 — Matching efficiency η (%) per seed, {pop}",
                             ["Variant"] + [f"seed {s}" for s in seeds], rows))

        # Q tanılama: fiyat kovası ziyaretleri ve ε
        rows = []
        for v in present:
            if VARIANTS[v]["family"] != "q":
                continue
            ds = [r.get("diagnostics", {}) for r in gp[v].values()]
            tr = [x["price_bucket_visits_train"] for x in ds if "price_bucket_visits_train" in x]
            ev = [x["price_bucket_visits_eval"] for x in ds if "price_bucket_visits_eval" in x]
            share = (np.sum(tr, axis=0) / max(1, np.sum(tr)) * 100) if tr else None
            share_ev = (np.sum(ev, axis=0) / max(1, np.sum(ev)) * 100) if ev else None
            rows.append([v,
                         f"{np.mean([x['visited_states_sell_mean'] for x in ds]):.1f} / "
                         f"{np.mean([x['visited_states_buy_mean'] for x in ds]):.1f}",
                         f"{np.mean([x['final_epsilon'] for x in ds]):.3f}",
                         " ".join(f"{x:.0f}" for x in share) if share is not None else "-",
                         " ".join(f"{x:.0f}" for x in share_ev) if share_ev is not None else "-"])
        if rows:
            sec.append(box_table(f"Table A5 — Q-Learning diagnostics, {pop}",
                                 ["Variant", "visited states sell/buy", "final ε",
                                  "price-bin share train (%)", "price-bin share eval (%)"], rows,
                                 ["Price bins: [no trade in previous slot, bin 1 .. K]; "
                                  "share of order-placing decisions (all agents)."]))

        # Tablo 5: yeniden üretim
        rep = reproduction(gp, pop, source)
        pj["reproduction"] = rep
        if rep:
            rows = [[x["variant"], str(x["seed"]), x["src_file"], f"{x['episodes']}/{x['src_episodes']}",
                     f"{x['max_abs_diff']:.3e}" if x["same_budget"] else "n/a (budget differs)",
                     "yes" if x["train_curve_identical"] else "no",
                     f"{x['eta']:.3f} / {x['src_eta']:.3f}"] for x in rep]
            sec.append(box_table(f"Reproduction check vs. published src results, {pop}",
                                 ["Variant", "seed", "src file", "episodes (abl/src)",
                                  "max |Δ metric|", "curve identical", "η abl / src"], rows,
                                 ["Expected ≈ 0 when budgets match and the machine/library versions match; "
                                  "report only (not a failure)."]))
        else:
            sec.append(f"  Reproduction check: skipped — no src result with profile source '{source}' "
                       f"({ref_dir(source)}/A_p{pop.replace('+', 'c')}_seed*.json) "
                       f"for population {pop}.")

        # Denge
        viol = [f"{r['variant']} s{r['seed']}" for vg in gp.values() for r in vg.values()
                if not r.get("balance_check", {}).get("ok", True)]
        sec.append(f"  Energy balance on {'every evaluation scenario' if source == 'ausgrid' else 'evaluation day'} "
                   f"(D = P2P + grid buy, S = P2P + grid sell): "
                   f"{'all jobs OK' if not viol else 'VIOLATIONS: ' + ', '.join(viol)}")
        out_json["populations"][pop] = pj

    return "\n\n".join(sec) + "\n", out_json


# ---------------------------------------------------------------------------
# Grafikler
# ---------------------------------------------------------------------------

def _bar_fig(g, key: str, ylabel: str, title: str, path: str, plt) -> Optional[str]:
    pops = pops_sorted(g)
    if not pops:
        return None
    fig, axes = plt.subplots(1, len(pops), figsize=(max(6.5, 1.0 * 8 * len(pops)), 4.4), squeeze=False)
    for ax, pop in zip(axes[0], pops):
        gp = g[pop]
        present = [v for v in VARIANT_ORDER if v in gp]
        x = np.arange(len(present))
        for i, v in enumerate(present):
            vals = [metric(gp[v][s], key) for s in sorted(gp[v])]
            m, sd = _ms(vals)
            ax.bar(i, m, width=0.72, color=COLORS[v], hatch="///" if v.endswith("_x2") else None,
                   edgecolor="white", linewidth=0.8, yerr=sd, capsize=3,
                   error_kw={"elinewidth": 1, "ecolor": "#52514e"})
            ax.scatter(i + np.linspace(-0.18, 0.18, len(vals)) if len(vals) > 1 else [i], vals,
                       s=12, color="#2b2a27", zorder=3, linewidths=0)
        nq = sum(1 for v in present if VARIANTS[v]["family"] == "q")
        if 0 < nq < len(present):
            ax.axvline(nq - 0.5, color="#b9b8b3", linewidth=1, linestyle=":")
        ax.set_xticks(x)
        ax.set_xticklabels([VARIANTS[v]["label"] for v in present], rotation=30, ha="right", fontsize=9)
        ax.set_title(f"Population {pop} (n = {max(len(gp[v]) for v in present)} seeds)", fontsize=10)
        ax.set_ylabel(ylabel)
        if key == "efficiency_pct":
            lo = min(min(metric(r, key) for r in gp[v].values()) for v in present)
            ax.set_ylim(max(0.0, min(lo - 5, 80.0)), 102)
        _style(ax)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _learning_fig(g, path: str, plt) -> Optional[str]:
    pops = pops_sorted(g)
    if not pops:
        return None
    fig, axes = plt.subplots(len(pops), 2, figsize=(11, 3.8 * len(pops)), squeeze=False)
    for i, pop in enumerate(pops):
        gp = g[pop]
        for j, (fam, title) in enumerate((("q", "Q-Learning variants"), ("nn", "REINFORCE variants"))):
            ax = axes[i, j]
            drew = False
            for v in [v for v in VARIANT_ORDER if v in gp and VARIANTS[v]["family"] == fam]:
                curves = [r["result"]["train_curve"] for r in gp[v].values() if r["result"].get("train_curve")]
                if not curves:
                    continue
                n = min(len(c) for c in curves)
                mean = np.array([c[:n] for c in curves], dtype=float).mean(axis=0)
                w = max(1, n // 50)
                sm = np.convolve(mean, np.ones(w) / w, mode="valid")
                ax.plot(np.arange(w, n + 1), sm, color=COLORS[v], linewidth=2,
                        linestyle="--" if v.endswith("_x2") else "-",
                        label=f"{VARIANTS[v]['label']} (n={len(curves)})")
                drew = True
            if drew:
                ax.legend(frameon=False, fontsize=8)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{title}, {pop}", fontsize=10)
            ax.set_xlabel("Episode")
            if j == 0:
                ax.set_ylabel("System reward")
            _style(ax)
    fig.suptitle("Training curves (mean over seeds, moving average with window = N/50)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_plots(g, d: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = [
        _bar_fig(g, "efficiency_pct", "Matching efficiency η = P2P / E_max (%)",
                 "Input ablation: matching efficiency (mean ± std over seeds; dots = seeds)",
                 os.path.join(d, "fig_ablation_eta.png"), plt),
        _bar_fig(g, "avg_price", "Average P2P clearing price ($/kWh)",
                 "Input ablation: average clearing price (mean ± std over seeds; dots = seeds)",
                 os.path.join(d, "fig_ablation_price.png"), plt),
        _learning_fig(g, os.path.join(d, "fig_ablation_learning.png"), plt),
    ]
    return [p for p in paths if p]


def main(argv: List[str]) -> int:
    d = argv[0] if argv else os.path.join(REPO_ROOT, "ablation", "results")
    if not os.path.isabs(d):
        d = os.path.join(REPO_ROOT, d)
    if not os.path.isdir(d):
        print(f"Directory not found: {d}")
        return 1
    runs = load_runs(d)
    if not runs:
        print(f"No ablation JSON found in {d}")
        return 1
    g = group(runs)
    text, js = build_text(g, d)
    print(text)
    with open(os.path.join(d, "ablation_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(d, "ablation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(js, f, ensure_ascii=False, indent=1)
    for p in make_plots(g, d):
        print(f"Figure saved: {p}")
    print(f"Summary written: {os.path.join(d, 'ablation_summary.txt')}, ablation_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
