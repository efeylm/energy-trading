"""
Şok deneyinin (src/experiment_shock.py) çok seed'li özeti. EĞİTİM YOK; yalnızca
dizindeki per-seed JSON'ları okur.

    python -m src.summarize_shock shock_results_ausgrid
    python -m src.summarize_shock supply_shock_results_ausgrid

Dizindeki bütün "*shock_p*c*_seed*.json" dosyaları şok türü × popülasyona göre
gruplanır. Bir grup içinde profil kaynağı, Ausgrid split_id'si, faz uzunlukları
ve çarpanları aynı olmalıdır (değilse hata); QL hiperparametre farkı uyarıdır.

Her grup için (önek talep şokunda "", arz şokunda "supply_"; dizinde aynı şok
türünden birden çok popülasyon varsa ada ayrıca "_p{P}c{C}" eklenir):
    <önek>shock_summary_seeds.txt   metin özeti (ekrana da basılır)
    <önek>shock_summary_seeds.json  aynı sayılar, makine okunur
    <önek>fig_shock_seeds.png       seed ortalaması eğitim eğrileri (± std bandı)

Metin özeti:
    (a) başlık: şok türü, popülasyon, seed'ler, profil kaynağı, fazlar
    (b) her fazın son SUMMARY_TAIL eğitim episode'u: seed'ler üzerinden ort. ± std (ddof=1)
    (c) Ausgrid held-out değerlendirmesi (hepsinde varsa): aynı biçimde
    (d) eşli farklar (aynı seed): P2 − P1 ve P3 − P1 — Ausgrid'de held-out η ve
        fiyat, sentetikte son-500 fiyat ve P2P hacmi
    (e) seed bazında tablo
n = 1 iken std "-" yazılır.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.experiment import _log, write_json_atomic
from src.experiment_shock import (
    C_BLUE, C_GREY, C_VERM, HELDOUT_ROWS, PHASE_FILL, SUMMARY_TAIL, TRAIN_ROWS,
    _moving_average, _prefix, _shock_of, summarize,
)

NAME_RE = re.compile(r"^(?:(?P<pre>[a-z]+)_)?shock_p(?P<p>\d+)c(?P<c>\d+)_seed(?P<s>-?\d+)\.json$")
CELL = 22          # sütun genişliği ("mean ± std")
LABEL = 34


# ---------------------------------------------------------------------------
# Okuma ve gruplama
# ---------------------------------------------------------------------------

def _load(path: str) -> Dict:
    import json
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _phase_sig(data: Dict) -> List[Tuple]:
    return [(ph["phase"], ph["start"], ph["end"], float(ph["multiplier"])) for ph in data["phases"]]


def load_groups(directory: str) -> Dict[Tuple[str, int, int], List[Dict]]:
    """{(şok, P, C): [veri, ...] (seed'e göre sıralı)}; tutarsızlıkta ValueError."""
    files = sorted(glob.glob(os.path.join(directory, "*shock_p*c*_seed*.json")))
    groups: Dict[Tuple[str, int, int], List[Dict]] = {}
    for path in files:
        if not NAME_RE.match(os.path.basename(path)):
            continue
        data = _load(path)
        data["_path"] = path
        key = (_shock_of(data), int(data["population"]["producers"]),
               int(data["population"]["consumers"]))
        groups.setdefault(key, []).append(data)
    for key, runs in groups.items():
        runs.sort(key=lambda d: int(d["seed"]))
        _check_group(key, runs)
    return groups


def _check_group(key, runs: List[Dict]) -> None:
    ref = runs[0]
    seeds = [int(d["seed"]) for d in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{key}: aynı seed birden fazla JSON'da: {seeds}")
    src0 = ref.get("profile_source", "synthetic")
    split0 = ref.get("ausgrid", {}).get("split_id")
    for d in runs[1:]:
        p = os.path.basename(d["_path"])
        if d.get("profile_source", "synthetic") != src0:
            raise ValueError(f"{key}: karışık profil kaynağı ({src0} / "
                             f"{d.get('profile_source', 'synthetic')} in {p})")
        if d.get("ausgrid", {}).get("split_id") != split0:
            raise ValueError(f"{key}: farklı Ausgrid split_id ({split0} / "
                             f"{d.get('ausgrid', {}).get('split_id')} in {p})")
        if _phase_sig(d) != _phase_sig(ref):
            raise ValueError(f"{key}: faz uzunlukları / çarpanları farklı ({p})")
        if d.get("ql_hyperparameters") != ref.get("ql_hyperparameters"):
            _log(f"UYARI {key}: QL hiperparametreleri farklı ({p})")
        if len(d["series"]["p2p_kwh"]) != len(ref["series"]["p2p_kwh"]):
            raise ValueError(f"{key}: seri uzunlukları farklı ({p})")


# ---------------------------------------------------------------------------
# İstatistik
# ---------------------------------------------------------------------------

def mstd(values) -> Dict:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if len(v) > 1 else None,
            "n": int(len(v)),
            "values": [float(x) for x in v]}


def _cell(st: Dict, fmt: str) -> str:
    s = fmt.format(st["mean"]) + " ± " + ("-" if st["std"] is None else fmt.format(st["std"]))
    return f"{s:>{CELL}}"


def _heldout_val(e: Dict, key: str) -> float:
    # _format_heldout ile aynı: D, S topluluk bloğundan, diğerleri metrics'ten
    return e["community"][key] if key in ("deficit_kwh", "surplus_kwh") else e["metrics"][key]


def aggregate(runs: List[Dict]) -> Dict:
    ref = runs[0]
    phases = ref["phases"]
    seeds = [int(d["seed"]) for d in runs]
    train = [summarize(d["series"], phases) for d in runs]          # son SUMMARY_TAIL
    cols = [c for c in (f"phase{ph['phase']}" for ph in phases) if c in train[0]]

    out: Dict = {
        "format": "energy-trading-shock-seeds/1",
        "shock": _shock_of(ref),
        "population": ref["population"],
        "seeds": seeds,
        "n_seeds": len(seeds),
        "profile_source": ref.get("profile_source", "synthetic"),
        "ausgrid_split_id": ref.get("ausgrid", {}).get("split_id"),
        "phases": phases,
        "summary_tail": SUMMARY_TAIL,
        "std_ddof": 1,
        "source_files": [os.path.basename(d["_path"]) for d in runs],
        "train_last_tail": {
            c: {"episodes": train[0][c]["episodes"],
                **{k: mstd([t[c][k] for t in train]) for k, _, _ in TRAIN_ROWS}}
            for c in cols
        },
    }

    has_he = all(d.get("heldout_eval", {}).get("phases") for d in runs)
    if has_he:
        he_ref = ref["heldout_eval"]["phases"]
        for d in runs[1:]:
            if [(e["phase"], e["multiplier"], e["n_scenarios"]) for e in d["heldout_eval"]["phases"]] != \
               [(e["phase"], e["multiplier"], e["n_scenarios"]) for e in he_ref]:
                raise ValueError(f"held-out fazları farklı ({os.path.basename(d['_path'])})")
        out["heldout"] = {
            f"phase{e['phase']}": {
                "multiplier": float(e["multiplier"]), "n_scenarios": int(e["n_scenarios"]),
                **{k: mstd([_heldout_val(d["heldout_eval"]["phases"][i], k) for d in runs])
                   for k, _, _ in HELDOUT_ROWS},
            }
            for i, e in enumerate(he_ref)
        }
        diff_src = "heldout"
        diff_keys = [("efficiency_pct", "Held-out eta (pp)", "{:.2f}"),
                     ("avg_price", "Held-out price ($/kWh)", "{:.4f}")]
        block = out["heldout"]
    else:
        diff_src = "train_last_tail"
        diff_keys = [("avg_price", f"Last-{SUMMARY_TAIL} price ($/kWh)", "{:.4f}"),
                     ("p2p_kwh", f"Last-{SUMMARY_TAIL} P2P (kWh/day)", "{:.2f}")]
        block = out["train_last_tail"]

    diffs: Dict = {"source": diff_src, "pairs": {}}
    for later in ("phase2", "phase3"):
        if later not in block or "phase1" not in block:
            continue
        pair = f"{later[-1]}-1"
        diffs["pairs"][pair] = {
            k: mstd(np.asarray(block[later][k]["values"]) - np.asarray(block["phase1"][k]["values"]))
            for k, _, _ in diff_keys
        }
    out["paired_differences"] = diffs
    out["_diff_keys"] = diff_keys
    return out


# ---------------------------------------------------------------------------
# Metin
# ---------------------------------------------------------------------------

def format_seeds_summary(agg: Dict) -> str:
    phases = agg["phases"]
    tr = agg["train_last_tail"]
    cols = list(tr.keys())
    width = LABEL + 2 + CELL * max(len(cols), 1)
    n = agg["n_seeds"]
    seeds = ", ".join(str(s) for s in agg["seeds"])
    lines = [
        "=" * width,
        f"{agg['shock'].upper()} SHOCK — Q-Learning {agg['population']['label']}, "
        f"{n} seed{'s' if n != 1 else ''}: {seeds}",
    ]
    if agg["profile_source"] == "ausgrid":
        lines.append(f"  profiles: Ausgrid real data (split {agg['ausgrid_split_id']})")
    else:
        lines.append(f"  profiles: {agg['profile_source']}")
    lines += [
        "  phases: " + " | ".join(
            f"{ph['phase']}: {ph['name']} x{ph['multiplier']:g}, episodes {ph['start'] + 1}-{ph['end']}"
            for ph in phases),
        f"  values: mean ± sample std (ddof=1) over seeds; '-' = undefined (n=1)",
        "=" * width,
        f"(b) TRAINING — mean of the last {SUMMARY_TAIL} training episodes of each phase "
        f"(exploration on)",
        "-" * width,
        f"  {'Metric':<{LABEL}}" + "".join(
            f"{'P' + c[-1] + ' ' + str(tr[c]['episodes'][0]) + '-' + str(tr[c]['episodes'][1]):>{CELL}}"
            for c in cols),
        "-" * width,
    ]
    for key, label, fmt in TRAIN_ROWS:
        lines.append(f"  {label:<{LABEL}}" + "".join(_cell(tr[c][key], fmt) for c in cols))
    lines.append("=" * width)

    if "heldout" in agg:
        he = agg["heldout"]
        hc = list(he.keys())
        ns = he[hc[0]]["n_scenarios"]
        lines += [
            f"(c) HELD-OUT EVALUATION — Q-tables frozen at the end of each phase, greedy (eps=0), "
            f"{ns} Ausgrid test scenarios",
            f"  each column: that phase's multiplier applied to the test scenarios; energies / $ "
            f"summed over the {ns} scenarios, eta = 100*sum(P2P)/sum(E_max)",
            "-" * width,
            f"  {'Metric':<{LABEL}}" + "".join(
                f"{'P' + c[-1] + ' end x' + format(he[c]['multiplier'], 'g'):>{CELL}}" for c in hc),
            "-" * width,
        ]
        for key, label, fmt in HELDOUT_ROWS:
            lines.append(f"  {label:<{LABEL}}" + "".join(_cell(he[c][key], fmt) for c in hc))
        lines.append("=" * width)
    else:
        lines += ["(c) HELD-OUT EVALUATION — not available (no heldout_eval in the JSONs)",
                  "=" * width]

    # (d) eşli farklar
    dk = agg["_diff_keys"]
    pairs = agg["paired_differences"]["pairs"]
    src = ("held-out evaluation" if agg["paired_differences"]["source"] == "heldout"
           else f"last {SUMMARY_TAIL} training episodes")
    pair_cols = list(pairs.keys())
    lines += [
        f"(d) PAIRED DIFFERENCES over seeds (same seed), from the {src}",
        "  P3 - P1 = 0 means recovery returns to the pre-shock level"
        + ("; P2 - P1 compares phase 2 (shocked test set) with phase 1" if src.startswith("held") else ""),
        "-" * width,
        f"  {'Metric':<{LABEL}}" + "".join(f"{'P' + p.replace('-', ' - P'):>{CELL}}" for p in pair_cols),
        "-" * width,
    ]
    for key, label, fmt in dk:
        signed = fmt.replace("{:", "{:+")
        lines.append(f"  {label:<{LABEL}}" + "".join(
            f"{(signed.format(pairs[p][key]['mean']) + ' ± ' + ('-' if pairs[p][key]['std'] is None else fmt.format(pairs[p][key]['std']))):>{CELL}}"
            for p in pair_cols))
    lines.append("=" * width)

    # (e) seed bazında
    block = agg["heldout"] if "heldout" in agg else agg["train_last_tail"]
    bcols = list(block.keys())
    hdr_src = "held-out" if "heldout" in agg else f"last-{SUMMARY_TAIL} training"
    w2 = 10
    lines += [
        f"(e) PER SEED — {hdr_src} values",
        "-" * width,
        f"  {'Seed':<6}" + "".join(
            f"{(short + ' P' + c[-1]):>{w2}}" for _, short, _ in
            [(k, {'efficiency_pct': 'eta%', 'avg_price': 'price', 'p2p_kwh': 'P2P'}[k], f) for k, _, f in dk]
            for c in bcols),
        "-" * width,
    ]
    for i, s in enumerate(agg["seeds"]):
        lines.append(f"  {s:<6}" + "".join(
            f"{fmt.format(block[c][k]['values'][i]):>{w2}}" for k, _, fmt in dk for c in bcols))
    lines.append("=" * width)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Şekil
# ---------------------------------------------------------------------------

def plot_seeds(runs: List[Dict], agg: Dict, path: str) -> Optional[str]:
    try:
        import matplotlib
    except ImportError:
        _log("UYARI: matplotlib yok, şekil atlandı")
        return None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    })

    phases = runs[0]["phases"]
    n = len(runs[0]["series"]["p2p_kwh"])
    x = np.arange(1, n + 1)
    w = max(50, n // 80)                                   # plot_all ile aynı pencere

    def ma_stack(key: str) -> np.ndarray:                  # (seed, episode)
        return np.vstack([_moving_average(np.asarray(d["series"][key], dtype=float), phases, w)
                          for d in runs])

    price, p2p = ma_stack("avg_price"), ma_stack("p2p_kwh")
    panels = [("Price ($/kWh)", price, C_BLUE), ("P2P (kWh/day)", p2p, C_BLUE)]
    has_emax = all("e_max_kwh" in d["series"] for d in runs)
    emax = ma_stack("e_max_kwh") if has_emax else None
    if has_emax:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(emax > 0, 100.0 * p2p / emax, np.nan)
        panels.append(("P2P / $E_{\\max}$ (%)", ratio, C_VERM))

    fig, axes = plt.subplots(len(panels), 1, figsize=(4.6, 1.75 * len(panels) + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    bounds = [ph["end"] for ph in phases[:-1]]
    for ax, (ylabel, Y, col) in zip(axes, panels):
        for i, ph in enumerate(phases):
            if ph["end"] > ph["start"]:
                ax.axvspan(ph["start"] + 0.5, ph["end"] + 0.5, color=PHASE_FILL[i % 3],
                           alpha=0.18, lw=0, zorder=0)
        for b in bounds:
            if 0 < b < n:
                ax.axvline(b + 0.5, color=C_GREY, ls="--", lw=0.7, zorder=1)
        mean = np.nanmean(Y, axis=0)
        if len(runs) > 1:
            sd = np.nanstd(Y, axis=0, ddof=1)
            ax.fill_between(x, mean - sd, mean + sd, color=col, alpha=0.22, lw=0, zorder=2)
        ax.plot(x, mean, color=col, lw=1.2, zorder=3)
        if ylabel.startswith("P2P (") and emax is not None:
            ax.plot(x, emax.mean(axis=0), color="black", lw=0.7, ls=(0, (4, 2)), zorder=3)
        ax.set_ylabel(ylabel, fontsize=7.5)
        ax.grid(True, color="#d9d9d9", lw=0.35, alpha=0.7)
        ax.set_axisbelow(True)
        ax.set_xlim(0.5, n + 0.5)
        ax.tick_params(labelsize=7.0, length=2.5, width=0.6, pad=2)
    axes[-1].set_xlabel("Training episode", fontsize=7.5)
    for ax, tag in zip(axes, "abcd"):
        ax.text(0.012, 0.965, f"({tag})", transform=ax.transAxes, ha="left", va="top",
                fontsize=7.5, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.8))
    fig.align_ylabels(axes)

    short = {1: "Phase 1:", 2: "Phase 2: shock", 3: "Phase 3:"}
    handles = [Patch(facecolor=PHASE_FILL[i % 3], alpha=0.45, edgecolor="none",
                     label=f"{short.get(ph['phase'], 'Phase ' + str(ph['phase']))} ×{ph['multiplier']:.1f}")
               for i, ph in enumerate(phases)]
    handles.append(Line2D([], [], color=C_BLUE, lw=1.2,
                          label=f"mean of {len(runs)} seeds" + (" ± std" if len(runs) > 1 else "")))
    if emax is not None:
        handles.append(Line2D([], [], color="black", lw=0.7, ls=(0, (4, 2)), label="$E_{\\max}$"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=0.4)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Şok deneyinin çok seed'li özeti (eğitim yok)")
    ap.add_argument("dir", help="per-seed *shock_p*c*_seed*.json dosyalarının dizini")
    ap.add_argument("--no-plot", dest="no_plot", action="store_true", help="şekil çizme")
    args = ap.parse_args(argv)

    groups = load_groups(args.dir)
    if not groups:
        sys.exit(f"{args.dir}: *shock_p*c*_seed*.json bulunamadı")
    per_shock: Dict[str, int] = {}
    for shock, _, _ in groups:
        per_shock[shock] = per_shock.get(shock, 0) + 1

    for (shock, p, c), runs in sorted(groups.items()):
        pop = f"_p{p}c{c}" if per_shock[shock] > 1 else ""
        pre = _prefix(shock)
        agg = aggregate(runs)
        text = format_seeds_summary(agg)
        agg.pop("_diff_keys")
        txt = os.path.join(args.dir, f"{pre}shock_summary_seeds{pop}.txt")
        with open(txt, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        write_json_atomic(agg, os.path.join(args.dir, f"{pre}shock_summary_seeds{pop}.json"))
        _log(f"özet: {txt}")
        if not args.no_plot:
            fig = plot_seeds(runs, agg, os.path.join(args.dir, f"{pre}fig_shock_seeds{pop}.png"))
            if fig:
                _log(f"grafik: {fig}")
        print(text)


if __name__ == "__main__":
    main()
