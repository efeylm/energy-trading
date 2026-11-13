"""
Deney planının tamamını paralel çalıştırır (her iş ayrı bir süreç).

Deneyler (popülasyonlar: 4+4, 10+10, 20+20, 10+20 — üretici+tüketici):
  A — popülasyon boyutu : her popülasyon için random (200 çekiliş) + Q-Learning
                          (3000 ep) + REINFORCE (7000 ep, 2×64), seed 1..5   → 20 iş
  B — derin ağ          : her popülasyon için yalnızca REINFORCE (7000 ep),
                          3 ve 4 gizli katman (genişlik 64), seed 1..3      → 24 iş
                          (2 katmanlı referans = A'nın aynı seed'li NN sonucu)
  C — yakınsama kontrolü: 20+20, Q-Learning 6000 + REINFORCE 14000 ep
                          (2× bütçe, varsayılan ağ), seed 1..3              →  3 iş

Kullanım:
    python -m src.run_experiments --workers 16            # tüm plan → results/
    python -m src.run_experiments --only A --dry-run      # yalnızca A'yı listele
    python -m src.run_experiments --smoke --workers 4     # küçük deneme → results_smoke/

Her iş: sys.executable -m src.experiment ... (OMP/MKL/OPENBLAS_NUM_THREADS=1).
Çıktı: <out_dir>/<iş adı>.json ve <out_dir>/logs/<iş adı>.log.
Devam ettirilebilir: JSON'u zaten olan iş atlanır. Başarısız iş diğerlerini
durdurmaz; sonunda başarısızlar listelenir ve çıkış kodu 1 olur.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

POPULATIONS = [(4, 4), (10, 10), (20, 20), (10, 20)]
CONV_POP = (20, 20)

FULL = dict(pops=POPULATIONS, seeds_a=[1, 2, 3, 4, 5], seeds_b=[1, 2, 3], seeds_c=[1, 2, 3],
            ql=3000, nn=7000, draws=200, conv_ql=6000, conv_nn=14000, conv_pop=CONV_POP,
            out_dir="results")
# Smoke: aynı iş yapısı, küçük bütçe. Yalnızca 4+4 ve 10+20 popülasyonları
# kullanıldığından yakınsama kontrolü de bu ikisinin büyüğü (10+20) üzerinde
# çalışır — böylece özetteki 1× / 2× tablosu uçtan uca sınanır.
SMOKE = dict(pops=[(4, 4), (10, 20)], seeds_a=[1], seeds_b=[1], seeds_c=[1],
             ql=4, nn=4, draws=5, conv_ql=8, conv_nn=8, conv_pop=(10, 20),
             out_dir="results_smoke")


@dataclass
class Job:
    group: str          # A | B | C
    name: str
    producers: int
    consumers: int
    seed: int
    methods: str
    episodes: int
    nn_episodes: int
    nn_layers: int
    draws: int

    def cost(self) -> float:
        """Göreli süre tahmini (yalnızca sıralama için): ajan × episode, NN ağırlıklı."""
        n = self.producers + self.consumers
        c = 0.0
        if "random" in self.methods:
            c += 0.3 * self.draws
        if "ql" in self.methods:
            c += self.episodes
        if "nn" in self.methods:
            c += 8.0 * self.nn_episodes * (1.0 + 0.2 * (self.nn_layers - 2))
        return n * c

    def argv(self, out_path: str) -> List[str]:
        return [
            sys.executable, "-m", "src.experiment",
            "--producers", str(self.producers), "--consumers", str(self.consumers),
            "--seed", str(self.seed), "--methods", self.methods,
            "--episodes", str(self.episodes), "--nn_episodes", str(self.nn_episodes),
            "--nn_layers", str(self.nn_layers), "--baseline_draws", str(self.draws),
            "--out", out_path, "--tag", self.group,
        ]


def _pop(p: int, c: int) -> str:
    return f"p{p}c{c}"


def build_jobs(plan: Dict, only: Optional[List[str]] = None) -> List[Job]:
    jobs: List[Job] = []
    for p, c in plan["pops"]:
        for s in plan["seeds_a"]:
            jobs.append(Job("A", f"A_{_pop(p, c)}_seed{s}", p, c, s, "random,ql,nn",
                            plan["ql"], plan["nn"], 2, plan["draws"]))
    for p, c in plan["pops"]:
        for layers in (3, 4):
            for s in plan["seeds_b"]:
                jobs.append(Job("B", f"B_{_pop(p, c)}_L{layers}_seed{s}", p, c, s, "nn",
                                plan["ql"], plan["nn"], layers, plan["draws"]))
    p, c = plan["conv_pop"]
    for s in plan["seeds_c"]:
        jobs.append(Job("C", f"C_{_pop(p, c)}_x2_seed{s}", p, c, s, "ql,nn",
                        plan["conv_ql"], plan["conv_nn"], 2, plan["draws"]))
    if only:
        jobs = [j for j in jobs if j.group in only]
    # En uzun işler önce (büyük popülasyon / büyük bütçe); eşitlikte ada göre.
    jobs.sort(key=lambda j: (-j.cost(), j.name))
    return jobs


def _fmt(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def _job_wall(path: str) -> Optional[float]:
    try:
        with open(path, encoding="utf-8") as f:
            return float(json.load(f)["total_wall_seconds"])
    except (OSError, ValueError, KeyError):
        return None


def _parse_only(values: Optional[List[str]]) -> Optional[List[str]]:
    if not values:
        return None
    out = []
    for v in values:
        out += [x.strip().upper() for x in v.split(",") if x.strip()]
    bad = [x for x in out if x not in ("A", "B", "C")]
    if bad:
        raise SystemExit(f"--only: geçersiz grup(lar) {bad} (A, B, C)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Deney planını paralel çalıştır (A, B, C)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="Aynı anda çalışan iş sayısı (varsayılan: CPU sayısının yarısı)")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Çıktı dizini (varsayılan: results/, --smoke ile results_smoke/)")
    ap.add_argument("--only", action="append", default=None,
                    help="Yalnızca bu gruplar: A, B, C (tekrarlanabilir veya virgüllü)")
    ap.add_argument("--dry-run", action="store_true", help="İşleri listele, çalıştırma")
    ap.add_argument("--smoke", action="store_true", help="Küçük bütçeli deneme planı")
    args = ap.parse_args()

    plan = SMOKE if args.smoke else FULL
    out_dir = os.path.abspath(args.out_dir or os.path.join(REPO_ROOT, plan["out_dir"]))
    log_dir = os.path.join(out_dir, "logs")
    jobs = build_jobs(plan, _parse_only(args.only))
    workers = max(1, args.workers)
    total = len(jobs)

    def out_path(j: Job) -> str:
        return os.path.join(out_dir, j.name + ".json")

    if args.dry_run:
        print(f"{total} iş → {out_dir}  (workers={workers})")
        for i, j in enumerate(jobs, 1):
            state = "VAR (atlanacak)" if os.path.exists(out_path(j)) else "bekliyor"
            print(f"  {i:3d}. {j.name:<26} {j.producers:>2}+{j.consumers:<2} seed={j.seed} "
                  f"methods={j.methods:<12} ql={j.episodes:<5} nn={j.nn_episodes:<5} "
                  f"L={j.nn_layers} draws={j.draws:<4} [{state}]")
        return 0

    os.makedirs(log_dir, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONUNBUFFERED="1")
    t_start = time.perf_counter()

    def stamp() -> str:
        return f"geçen {_fmt(time.perf_counter() - t_start)}"

    pending: List[Job] = []
    done = 0
    for j in jobs:
        if os.path.exists(out_path(j)):
            done += 1
            print(f"[{done:3d}/{total}] ATLA   {j.name} (JSON zaten var)", flush=True)
        else:
            pending.append(j)
    print(f"{total} iş, {done} tanesi zaten tamam, {len(pending)} çalışacak "
          f"| workers={workers} | çıktı: {out_dir}", flush=True)

    running: Dict[str, tuple] = {}   # name → (job, Popen, log file, t0)
    failed: List[str] = []
    while pending or running:
        while pending and len(running) < workers:
            j = pending.pop(0)
            log = open(os.path.join(log_dir, j.name + ".log"), "w", encoding="utf-8")
            proc = subprocess.Popen(j.argv(out_path(j)), cwd=REPO_ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
            running[j.name] = (j, proc, log, time.perf_counter())
            print(f"[{done:3d}/{total}] BAŞLA  {j.name} | {stamp()}", flush=True)
        time.sleep(0.5)
        for name in list(running):
            j, proc, log, t0 = running[name]
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            del running[name]
            done += 1
            wall = time.perf_counter() - t0
            if rc == 0 and os.path.exists(out_path(j)):
                print(f"[{done:3d}/{total}] BİTTİ  {name} {wall:8.1f}s | {stamp()}", flush=True)
            else:
                failed.append(name)
                print(f"[{done:3d}/{total}] HATA   {name} (çıkış kodu {rc}, {wall:.1f}s) "
                      f"→ {os.path.join(log_dir, name + '.log')} | {stamp()}", flush=True)

    # ── Grup bazında süre tablosu (JSON'lardaki toplam iş süresinden) ──────
    groups: Dict[str, List[float]] = {}
    for j in jobs:
        w = _job_wall(out_path(j))
        if w is not None:
            key = f"{j.group} {j.producers}+{j.consumers}" + (f" L{j.nn_layers}" if j.group == "B" else "")
            groups.setdefault(key, []).append(w)
    print()
    print("+" + "=" * 66 + "+")
    print(f"|{'İŞ GRUBU BAZINDA SÜRE (s)':^66}|")
    print("+" + "=" * 66 + "+")
    print(f"|  {'Grup':<20} {'İş':>4} {'Ortalama':>11} {'Maks':>11} {'Toplam':>12}  |")
    print(f"|  {'-' * 62}  |")
    for key in sorted(groups):
        v = groups[key]
        print(f"|  {key:<20} {len(v):>4} {sum(v) / len(v):>11.1f} {max(v):>11.1f} {sum(v):>12.1f}  |")
    print("+" + "=" * 66 + "+")
    all_w = [w for v in groups.values() for w in v]
    print(f"  Tamamlanan iş: {len(all_w)}/{total} | toplam iş süresi (≈ tek çekirdek CPU) "
          f"{_fmt(sum(all_w))} | bu çalıştırmanın duvar saati {_fmt(time.perf_counter() - t_start)}")

    if failed:
        print(f"\nBAŞARISIZ İŞLER ({len(failed)}):")
        for n in failed:
            print(f"  - {n}  (log: {os.path.join(log_dir, n + '.log')})")
        return 1
    print("\nTüm işler başarılı.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
