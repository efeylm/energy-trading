"""
Deney sonuçlarını (src/experiment.py JSON'ları) tablolara ve grafiklere döker.

Kullanım:
    python -m src.summarize_experiments results          # veya results_smoke

Çıktılar (aynı dizine):
    summary.txt                 düz metin kutu tabloları (aynı zamanda ekrana basılır)
    summary.csv                 iş × yöntem başına bir satır, tüm skaler alanlar
    fig_eta_by_population.png   η — popülasyon × strateji (hata çubuklu)
    fig_nn_eta_by_depth.png     REINFORCE η — popülasyon × gizli katman sayısı
    fig_training_curves.png     eğitim eğrileri (seed ortalaması, hafif yumuşatılmış)

Her hücre "ortalama ± std" (std: seed'ler arası, ddof=1; tek seed'de ± 0).
Eksik gruplar (ör. yarım kalmış bir çalıştırma) hata vermez, EKSİK olarak yazılır.
Grafik etiketleri İngilizcedir (makaleye girebilir).
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.random_environment import METRIC_ROWS

POP_ORDER = ["4+4", "10+10", "20+20", "10+20"]
METHODS = [("random", "Random"), ("ql", "Q-Learning"), ("nn", "REINFORCE")]

# Ölçekten bağımsız metrikler önce, mutlak değerler sonra.
_LABELS = {k: (lbl, fmt) for k, lbl, fmt in METRIC_ROWS}
ROW_ORDER = (["efficiency_pct", "p2p_ratio_pct", "avg_price"]
             + [k for k, _, _ in METRIC_ROWS
                if k not in ("efficiency_pct", "p2p_ratio_pct", "avg_price", "e_max_kwh")]
             + ["e_max_kwh"])
COMMUNITY_ROWS = [("surplus_kwh", "Topluluk fazlası S (kWh)", "{:.2f}"),
                  ("deficit_kwh", "Topluluk açığı D (kWh)", "{:.2f}")]

# Varsayılan doğrulanmış kategorik palet (dataviz referansı), sabit sıra.
COLORS = {"random": "#2a78d6", "ql": "#eb6834", "nn": "#1baf7a"}
DEPTH_COLORS = {2: "#86b6ef", 3: "#2a78d6", 4: "#104281"}   # tek ton, açık → koyu


# ---------------------------------------------------------------------------
# Yükleme ve gruplama
# ---------------------------------------------------------------------------

def load_runs(d: str) -> List[Dict]:
    runs = []
    for path in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                r = json.load(f)
        except (OSError, ValueError) as e:
            print(f"  [UYARI] okunamadı: {path} ({e})")
            continue
        if r.get("format", "").startswith("energy-trading-experiment"):
            r["_file"] = os.path.basename(path)
            runs.append(r)
    return runs


def _pop(r: Dict) -> str:
    return r["population"]["label"]


def _pops_present(runs: List[Dict]) -> List[str]:
    found = {_pop(r) for r in runs}
    return [p for p in POP_ORDER if p in found] + sorted(found - set(POP_ORDER))


def collect(runs: List[Dict], tag: str, pop: str, method: str,
            layers: Optional[int] = None, seeds: Optional[Sequence[int]] = None) -> Dict[int, Dict]:
    """seed → yöntem sonucu (metrics'e community eklenmiş) sözlüğü."""
    out = {}
    for r in runs:
        if r.get("tag") != tag or _pop(r) != pop or method not in r["results"]:
            continue
        if seeds is not None and r["seed"] not in seeds:
            continue
        res = r["results"][method]
        if layers is not None and res.get("n_layers", 2) != layers:
            continue
        res = dict(res)
        res["metrics"] = dict(res["metrics"], **r["community"])
        out[r["seed"]] = res
    return out


def _vals(group: Dict[int, Dict], key: str) -> np.ndarray:
    return np.array([g["metrics"][key] for _, g in sorted(group.items())], dtype=float)


def _ms(v: np.ndarray) -> Tuple[float, float]:
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)


def _cell(group: Dict[int, Dict], key: str, fmt: str) -> str:
    if not group:
        return "EKSİK"
    m, s = _ms(_vals(group, key))
    return f"{fmt.format(m)} ± {fmt.format(s)}"


def _pct(group: Dict[int, Dict], ref: Dict[int, Dict], key: str) -> str:
    if not group or not ref:
        return "-"
    a, b = float(np.mean(_vals(group, key))), float(np.mean(_vals(ref, key)))
    if abs(b) < 1e-9:
        return "N/A"
    return f"{(a - b) / abs(b) * 100:+.1f}%"


def _seeds(group: Dict[int, Dict]) -> str:
    return ",".join(str(s) for s in sorted(group)) or "-"


# ---------------------------------------------------------------------------
# Kutu tablo (comparison.py loglarıyla aynı stil)
# ---------------------------------------------------------------------------

def box_table(title: str, header: List[str], rows: List[List[str]],
              notes: Sequence[str] = ()) -> str:
    cols = list(zip(*([header] + rows))) if rows else [[h] for h in header]
    widths = [max(len(str(c)) for c in col) for col in cols]
    def fmt_row(cells):
        first = f"{cells[0]:<{widths[0]}}"
        rest = " ".join(f"{c:>{w}}" for c, w in zip(cells[1:], widths[1:]))
        return f"{first} {rest}" if rest else first
    inner = max(len(fmt_row(header)), len(title) + 2)
    width = inner + 4
    lines = ["+" + "=" * width + "+", f"|{('  ' + title):^{width}}|", "+" + "=" * width + "+",
             f"|  {fmt_row(header):<{inner}}  |", f"|  {'-' * inner}  |"]
    lines += [f"|  {fmt_row(r):<{inner}}  |" for r in rows]
    lines.append("+" + "=" * width + "+")
    lines += [f"  {n}" for n in notes]
    return "\n".join(lines)


def _metric_rows(keys=None):
    rows = [(k,) + _LABELS[k] for k in (keys or ROW_ORDER)]
    return rows + COMMUNITY_ROWS if keys is None else rows


# ---------------------------------------------------------------------------
# Tablolar
# ---------------------------------------------------------------------------

def table_a(runs: List[Dict]) -> List[str]:
    out = []
    for pop in _pops_present(runs):
        g = {m: collect(runs, "A", pop, m) for m, _ in METHODS}
        if not any(g.values()):
            continue
        header = ["Metrik"] + [f"{lbl} (n={len(g[m])})" for m, lbl in METHODS] + ["ΔQL", "ΔNN"]
        rows = [[lbl] + [_cell(g[m], k, fmt) for m, _ in METHODS]
                + [_pct(g["ql"], g["random"], k), _pct(g["nn"], g["random"], k)]
                for k, lbl, fmt in _metric_rows()]
        out.append(box_table(
            f"DENEY A — POPÜLASYON {pop} (üretici+tüketici)", header, rows,
            [f"seed'ler: Random {_seeds(g['random'])} | QL {_seeds(g['ql'])} | NN {_seeds(g['nn'])}",
             "ΔQL, ΔNN: ortalamaların Random baseline'a göre % değişimi",
             "Random: her seed'de çekiliş ortalaması; ± seed'ler arası std. "
             "S, D, E_max yalnızca profillere bağlıdır."]))
    return out or ["[EKSİK] Deney A: hiç sonuç yok"]


def table_cross(runs: List[Dict]) -> List[str]:
    pops = [p for p in _pops_present(runs) if any(collect(runs, "A", p, m) for m, _ in METHODS)]
    if not pops:
        return ["[EKSİK] Popülasyonlar arası tablo: Deney A sonucu yok"]
    out = []
    for key in ("efficiency_pct", "p2p_ratio_pct", "avg_price"):
        lbl, fmt = _LABELS[key]
        rows = []
        for pop in pops:
            g = {m: collect(runs, "A", pop, m) for m, _ in METHODS}
            n = max(len(v) for v in g.values())
            rows.append([pop, str(n)] + [_cell(g[m], key, fmt) for m, _ in METHODS])
        out.append(box_table(f"POPÜLASYONLAR ARASI — {lbl}",
                             ["Popülasyon", "n"] + [l for _, l in METHODS], rows))
    return out


def table_b(runs: List[Dict]) -> List[str]:
    out = []
    for pop in _pops_present(runs):
        deep = {L: collect(runs, "B", pop, "nn", layers=L) for L in (3, 4)}
        seeds = sorted(set(deep[3]) | set(deep[4]))
        if not seeds:
            continue
        g = {2: collect(runs, "A", pop, "nn", layers=2, seeds=seeds), **deep}
        def hdr(L):
            p = next((v["n_params_per_net"] for v in g[L].values()), None)
            return f"{L} katman (p={p if p is not None else '?'}, n={len(g[L])})"
        header = ["Metrik"] + [hdr(L) for L in (2, 3, 4)] + ["Δ3L", "Δ4L"]
        rows = [[lbl] + [_cell(g[L], k, fmt) for L in (2, 3, 4)]
                + [_pct(g[3], g[2], k), _pct(g[4], g[2], k)]
                for k, lbl, fmt in _metric_rows(ROW_ORDER)]
        rows.append(["Eğitim süresi (s)"] + [_time_cell(g[L]) for L in (2, 3, 4)]
                    + [_pct_time(g[3], g[2]), _pct_time(g[4], g[2])])
        notes = [f"seed'ler: 2L {_seeds(g[2])} | 3L {_seeds(g[3])} | 4L {_seeds(g[4])}",
                 "2 katman = Deney A'nın aynı seed'li REINFORCE sonucu; p = tek politika ağının "
                 "parametre sayısı; Δ: 2 katmana göre % değişim"]
        missing = [f"{L}L" for L in (2, 3, 4) if not g[L]]
        if missing:
            notes.append(f"[EKSİK] {', '.join(missing)} sonuçları yok")
        out.append(box_table(f"DENEY B — REINFORCE DERİNLİK, POPÜLASYON {pop}", header, rows, notes))
    return out or ["[EKSİK] Deney B: hiç sonuç yok"]


def _time_cell(group: Dict[int, Dict], field: str = "train_seconds") -> str:
    if not group:
        return "EKSİK"
    m, s = _ms(np.array([g[field] for g in group.values()]))
    return f"{m:.1f} ± {s:.1f}"


def _pct_time(a: Dict[int, Dict], b: Dict[int, Dict]) -> str:
    if not a or not b:
        return "-"
    ma = np.mean([g["train_seconds"] for g in a.values()])
    mb = np.mean([g["train_seconds"] for g in b.values()])
    return f"{(ma - mb) / mb * 100:+.1f}%" if mb > 0 else "N/A"


def table_c(runs: List[Dict]) -> List[str]:
    pops = sorted({_pop(r) for r in runs if r.get("tag") == "C"},
                  key=lambda p: POP_ORDER.index(p) if p in POP_ORDER else 99)
    if not pops:
        return ["[EKSİK] Deney C (yakınsama): hiç sonuç yok"]
    out = []
    for pop in pops:
        x2 = {m: collect(runs, "C", pop, m) for m in ("ql", "nn")}
        seeds = sorted(set(x2["ql"]) | set(x2["nn"]))
        x1 = {m: collect(runs, "A", pop, m, seeds=seeds) for m in ("ql", "nn")}
        def ep(g):
            return next((v["episodes"] for v in g.values()), "?")
        header = ["Metrik",
                  f"QL {ep(x1['ql'])} (n={len(x1['ql'])})", f"QL {ep(x2['ql'])} (n={len(x2['ql'])})", "ΔQL",
                  f"NN {ep(x1['nn'])} (n={len(x1['nn'])})", f"NN {ep(x2['nn'])} (n={len(x2['nn'])})", "ΔNN"]
        rows = [[lbl, _cell(x1["ql"], k, fmt), _cell(x2["ql"], k, fmt), _pct(x2["ql"], x1["ql"], k),
                 _cell(x1["nn"], k, fmt), _cell(x2["nn"], k, fmt), _pct(x2["nn"], x1["nn"], k)]
                for k, lbl, fmt in _metric_rows(ROW_ORDER)]
        rows.append(["Eğitim süresi (s)", _time_cell(x1["ql"]), _time_cell(x2["ql"]), "-",
                     _time_cell(x1["nn"]), _time_cell(x2["nn"]), "-"])
        notes = [f"seed'ler: 1× QL {_seeds(x1['ql'])}, NN {_seeds(x1['nn'])} (Deney A) | "
                 f"2× QL {_seeds(x2['ql'])}, NN {_seeds(x2['nn'])} (Deney C)",
                 "Δ: 2× bütçenin 1× bütçeye göre % değişimi (küçükse eğitim yakınsamış demektir)"]
        if not x1["ql"] and not x1["nn"]:
            notes.append(f"[EKSİK] 1× referans (Deney A, {pop}) yok")
        out.append(box_table(f"DENEY C — YAKINSAMA KONTROLÜ, POPÜLASYON {pop} (1× vs 2× bütçe)",
                             header, rows, notes))
    return out


def table_times(runs: List[Dict]) -> List[str]:
    cols = [("A", "random", None, "eval_seconds", "Random değerl."),
            ("A", "ql", None, "train_seconds", "QL eğitim"),
            ("A", "nn", 2, "train_seconds", "NN 2L eğitim"),
            ("B", "nn", 3, "train_seconds", "NN 3L eğitim"),
            ("B", "nn", 4, "train_seconds", "NN 4L eğitim"),
            ("C", "ql", None, "train_seconds", "QL 2× eğitim"),
            ("C", "nn", 2, "train_seconds", "NN 2× eğitim")]
    rows = []
    for pop in _pops_present(runs):
        row = [pop]
        for tag, m, L, field, _ in cols:
            g = collect(runs, tag, pop, m, layers=L)
            row.append(_time_cell(g, field) if g else "-")
        rows.append(row)
    wall = sum(r["total_wall_seconds"] for r in runs)
    cpu = sum(res.get("cpu_seconds", 0.0) for r in runs for res in r["results"].values())
    notes = [f"Toplam iş sayısı: {len(runs)} | toplam iş süresi (duvar saati, her iş tek "
             f"iş parçacığı): {wall:.1f}s = {wall / 3600:.2f} saat",
             f"Toplam süreç CPU zamanı (yöntemler): {cpu:.1f}s = {cpu / 3600:.2f} saat "
             f"(Python başlangıcı / import hariç)",
             "Hücreler: seed'ler arası ortalama ± std, saniye. '-' = sonuç yok."]
    return [box_table("ÇALIŞMA SÜRELERİ (s, seed ortalaması)", ["Popülasyon"] + [c[4] for c in cols],
                      rows, notes)]


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def write_csv(runs: List[Dict], path: str) -> None:
    keys = [k for k, _, _ in METRIC_ROWS]
    fields = (["file", "tag", "producers", "consumers", "seed", "method", "episodes", "n_draws",
               "n_layers", "hidden", "n_params_per_net", "train_seconds", "eval_seconds",
               "cpu_seconds", "job_wall_seconds", "surplus_kwh", "deficit_kwh"]
              + keys + [f"{k}_draw_{s}" for s in ("std", "min", "max") for k in keys])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in runs:
            for m in r["methods"]:
                res = r["results"][m]
                row = {"file": r["_file"], "tag": r.get("tag", ""),
                       "producers": r["population"]["producers"],
                       "consumers": r["population"]["consumers"], "seed": r["seed"], "method": m,
                       "episodes": res.get("episodes", ""), "n_draws": res.get("n_draws", ""),
                       "n_layers": res.get("n_layers", ""), "hidden": res.get("hidden", ""),
                       "n_params_per_net": res.get("n_params_per_net", ""),
                       "train_seconds": res["train_seconds"], "eval_seconds": res["eval_seconds"],
                       "cpu_seconds": res.get("cpu_seconds", ""),
                       "job_wall_seconds": r["total_wall_seconds"],
                       "surplus_kwh": r["community"]["surplus_kwh"],
                       "deficit_kwh": r["community"]["deficit_kwh"]}
                row.update(res["metrics"])
                for s in ("std", "min", "max"):
                    for k, v in res.get(f"draw_{s}", {}).items():
                        row[f"{k}_draw_{s}"] = v
                w.writerow(row)


# ---------------------------------------------------------------------------
# Grafikler
# ---------------------------------------------------------------------------

def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8a8984")
    ax.tick_params(colors="#52514e")
    ax.grid(True, axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)


def _bars(ax, pops, series, colors, labels):
    """series: {anahtar: [(ortalama, std) veya None, ...]} — gruplu çubuk + hata çubuğu."""
    x = np.arange(len(pops))
    n = len(series)
    w = 0.8 / n
    for i, key in enumerate(series):
        vals = series[key]
        xs = [x[j] + (i - (n - 1) / 2) * w for j, v in enumerate(vals) if v is not None]
        ms = [v[0] for v in vals if v is not None]
        ss = [v[1] for v in vals if v is not None]
        if xs:
            ax.bar(xs, ms, width=w * 0.92, color=colors[key], label=labels[key], yerr=ss,
                   capsize=3, error_kw={"elinewidth": 1, "ecolor": "#52514e"})
    ax.set_xticks(x)
    ax.set_xticklabels(pops)


def make_plots(runs: List[Dict], d: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    pops = _pops_present(runs)

    def ms_or_none(g, key="efficiency_pct"):
        return _ms(_vals(g, key)) if g else None

    # 1) η — popülasyon × strateji
    series = {m: [ms_or_none(collect(runs, "A", p, m)) for p in pops] for m, _ in METHODS}
    if any(v is not None for vals in series.values() for v in vals):
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        _bars(ax, pops, series, COLORS, dict(METHODS))
        ax.set_xlabel("Population (producers + consumers)")
        ax.set_ylabel("Matching efficiency η = P2P / E_max (%)")
        ax.set_title("Matching efficiency by population (mean ± std over seeds)")
        ax.legend(frameon=False)
        _style(ax)
        fig.tight_layout()
        p = os.path.join(d, "fig_eta_by_population.png")
        fig.savefig(p, dpi=200)
        plt.close(fig)
        paths.append(p)

    # 2) REINFORCE η — derinlik
    series, labels = {}, {}
    for L in (2, 3, 4):
        vals = []
        for p in pops:
            seeds = sorted(set(collect(runs, "B", p, "nn", layers=3))
                           | set(collect(runs, "B", p, "nn", layers=4)))
            tag = "A" if L == 2 else "B"
            vals.append(ms_or_none(collect(runs, tag, p, "nn", layers=L, seeds=seeds)) if seeds else None)
        series[L] = vals
        labels[L] = f"{L} hidden layers"
    if any(v is not None for vals in series.values() for v in vals):
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        _bars(ax, pops, series, DEPTH_COLORS, labels)
        ax.set_xlabel("Population (producers + consumers)")
        ax.set_ylabel("Matching efficiency η (%)")
        ax.set_title("REINFORCE: effect of network depth (width 64, mean ± std)")
        ax.legend(frameon=False)
        _style(ax)
        fig.tight_layout()
        p = os.path.join(d, "fig_nn_eta_by_depth.png")
        fig.savefig(p, dpi=200)
        plt.close(fig)
        paths.append(p)

    # 3) Eğitim eğrileri — satır: yöntem, sütun: popülasyon
    curve_pops = [p for p in pops if collect(runs, "A", p, "ql") or collect(runs, "A", p, "nn")]
    if curve_pops:
        fig, axes = plt.subplots(2, len(curve_pops), figsize=(3.6 * len(curve_pops), 5.6),
                                 squeeze=False)
        for j, p in enumerate(curve_pops):
            for i, (m, lbl) in enumerate([("ql", "Q-Learning"), ("nn", "REINFORCE")]):
                ax = axes[i, j]
                g = collect(runs, "A", p, m)
                curves = [v["train_curve"] for v in g.values() if v.get("train_curve")]
                if curves:
                    n = min(len(c) for c in curves)
                    arr = np.array([c[:n] for c in curves], dtype=float)
                    mean = arr.mean(axis=0)
                    w = max(1, n // 50)
                    sm = np.convolve(mean, np.ones(w) / w, mode="valid")
                    ax.plot(np.arange(1, n + 1), mean, color=COLORS[m], alpha=0.25, linewidth=0.8)
                    ax.plot(np.arange(w, n + 1), sm, color=COLORS[m], linewidth=2)
                    ax.set_title(f"{lbl}, {p} (n={len(curves)})", fontsize=10)
                else:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(f"{lbl}, {p}", fontsize=10)
                if i == 1:
                    ax.set_xlabel("Episode")
                if j == 0:
                    ax.set_ylabel("System reward")
                _style(ax)
        fig.suptitle("Training curves (mean over seeds; bold = moving average, window = N/50)",
                     fontsize=11)
        fig.tight_layout()
        p = os.path.join(d, "fig_training_curves.png")
        fig.savefig(p, dpi=200)
        plt.close(fig)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------

def main(argv: List[str]) -> int:
    d = argv[0] if argv else "results"
    if not os.path.isdir(d):
        print(f"Dizin bulunamadı: {d}")
        return 1
    runs = load_runs(d)
    sections = [f"DENEY ÖZETİ — {os.path.abspath(d)} ({len(runs)} iş JSON'u)"]
    if runs:
        hosts = sorted({r.get("hostname", "?") for r in runs})
        sections.append(f"  Makine(ler): {', '.join(hosts)}")
        sections += ["", "### 1. DENEY A — POPÜLASYON BOYUTU"] + table_a(runs)
        sections += ["", "### 2. POPÜLASYONLAR ARASI KARŞILAŞTIRMA (Deney A)"] + table_cross(runs)
        sections += ["", "### 3. DENEY B — DERİN AĞ"] + table_b(runs)
        sections += ["", "### 4. DENEY C — YAKINSAMA KONTROLÜ"] + table_c(runs)
        sections += ["", "### 5. ÇALIŞMA SÜRELERİ"] + table_times(runs)
    else:
        sections.append("[EKSİK] Hiç sonuç JSON'u bulunamadı.")
    text = "\n\n".join(sections) + "\n"
    print(text)
    with open(os.path.join(d, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    if runs:
        write_csv(runs, os.path.join(d, "summary.csv"))
        for p in make_plots(runs, d):
            print(f"Grafik kaydedildi: {p}")
        print(f"Özet yazıldı: {os.path.join(d, 'summary.txt')}, {os.path.join(d, 'summary.csv')}")
    else:
        print(f"Özet yazıldı: {os.path.join(d, 'summary.txt')} (CSV / grafik yok)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
