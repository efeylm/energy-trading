"""
Tüm eğitim işlerini paralel alt süreçler olarak çalıştırır, sonra evaluate.py'yi çağırır.

    python -m shock_lab.run_all --workers 16              # tam plan → shock_lab/runs, shock_lab/eval
    python -m shock_lab.run_all --dry-run                 # yalnızca listele
    python -m shock_lab.run_all --smoke --workers 4       # küçük deneme → runs_smoke / eval_smoke

Her iş: sys.executable -m shock_lab.train ... (OMP/MKL/OPENBLAS_NUM_THREADS=1, torch 1 thread).
Log: <runs>/logs/<iş>.log. Devam ettirilebilir: run.json'u olan iş atlanır.
Başarısız iş diğerlerini durdurmaz; varsa sonunda listelenir, değerlendirme yine de
mevcut koşularla yapılır ve çıkış kodu 1 olur.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Dict, List

from shock_lab.common import REPO_ROOT, VARIANTS, VARIANT_ORDER, fmt_hms
from shock_lab.train import run_name

# Göreli maliyet (sıralama için): NN episode'u ≈ 8× Q episode'u (10+10, M1 ölçümü)
NN_COST = 8.0

SMOKE = dict(episodes_q=60, episodes_nn=60, test_days=2, timeline_days="3,3,3",
             specialist_episodes=40)


def main() -> int:
    ap = argparse.ArgumentParser(description="shock_lab: tüm varyant × seed işleri + değerlendirme")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--variants", default=",".join(VARIANT_ORDER))
    ap.add_argument("--episodes-q", type=int, default=40000)
    ap.add_argument("--episodes-nn", type=int, default=40000)
    ap.add_argument("--producers", type=int, default=10)
    ap.add_argument("--consumers", type=int, default=10)
    ap.add_argument("--runs", default=None, help="varsayılan shock_lab/runs (smoke: runs_smoke)")
    ap.add_argument("--eval-out", default=None, help="varsayılan shock_lab/eval (smoke: eval_smoke)")
    ap.add_argument("--test-days", type=int, default=30)
    ap.add_argument("--timeline-days", default="20,20,20")
    ap.add_argument("--specialist-episodes", type=int, default=5000)
    ap.add_argument("--no-specialists", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help=f"küçük bütçe: {SMOKE}")
    ap.add_argument("--train-args", default="",
                    help="train.py'ye aynen geçirilecek ek argümanlar (ör. \"--normal-prob 0.5\")")
    args = ap.parse_args()

    if args.smoke:
        args.episodes_q, args.episodes_nn = SMOKE["episodes_q"], SMOKE["episodes_nn"]
        args.test_days, args.timeline_days = SMOKE["test_days"], SMOKE["timeline_days"]
        args.specialist_episodes = SMOKE["specialist_episodes"]
    here = os.path.join(REPO_ROOT, "shock_lab")
    runs_dir = os.path.abspath(args.runs or os.path.join(here, "runs_smoke" if args.smoke else "runs"))
    eval_dir = os.path.abspath(args.eval_out or os.path.join(here, "eval_smoke" if args.smoke else "eval"))
    log_dir = os.path.join(runs_dir, "logs")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        raise SystemExit(f"bilinmeyen varyant(lar): {bad}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    jobs = []
    for v in variants:
        kind = VARIANTS[v]["kind"]
        n = args.episodes_nn if kind == "nn" else args.episodes_q
        for s in seeds:
            name = run_name(v, args.producers, args.consumers, s)
            argv = [sys.executable, "-m", "shock_lab.train", "--variant", v, "--seed", str(s),
                    "--producers", str(args.producers), "--consumers", str(args.consumers),
                    "--episodes", str(n), "--out", runs_dir] + args.train_args.split()
            cost = n * (NN_COST if kind == "nn" else 1.0) * (1.25 if VARIANTS[v].get("features") else 1.0)
            jobs.append({"name": name, "argv": argv, "cost": cost, "variant": v, "seed": s, "episodes": n})
    jobs.sort(key=lambda j: (-j["cost"], j["name"]))

    def done_path(j: Dict) -> str:
        return os.path.join(runs_dir, j["name"], "run.json")

    eval_argv = [sys.executable, "-m", "shock_lab.evaluate", "--runs", runs_dir, "--out", eval_dir,
                 "--test-days", str(args.test_days), "--timeline-days", args.timeline_days,
                 "--workers", str(args.workers)]
    if not args.no_specialists:
        eval_argv += ["--specialists", "--specialist-episodes", str(args.specialist_episodes),
                      "--specialist-seeds", ",".join(map(str, seeds))]

    if args.dry_run:
        print(f"{len(jobs)} iş → {runs_dir} (workers={args.workers})")
        for i, j in enumerate(jobs, 1):
            st = "VAR (atlanacak)" if os.path.exists(done_path(j)) else "bekliyor"
            print(f"  {i:3d}. {j['name']:<26} episodes={j['episodes']:<6} [{st}]")
        print("değerlendirme:" if not args.no_eval else "değerlendirme: (--no-eval)",
              " ".join(eval_argv[1:]) if not args.no_eval else "")
        return 0

    os.makedirs(log_dir, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONUNBUFFERED="1")
    t_start = time.perf_counter()

    def stamp() -> str:
        return f"geçen {fmt_hms(time.perf_counter() - t_start)}"

    pending, ndone = [], 0
    for j in jobs:
        if os.path.exists(done_path(j)):
            ndone += 1
            print(f"[{ndone:3d}/{len(jobs)}] ATLA   {j['name']} (run.json var)", flush=True)
        else:
            pending.append(j)
    print(f"{len(jobs)} iş, {ndone} tamam, {len(pending)} çalışacak | workers={args.workers} "
          f"| çıktı: {runs_dir}", flush=True)

    running: Dict[str, tuple] = {}
    failed: List[str] = []
    while pending or running:
        while pending and len(running) < args.workers:
            j = pending.pop(0)
            lf = open(os.path.join(log_dir, j["name"] + ".log"), "w", encoding="utf-8")
            proc = subprocess.Popen(j["argv"], cwd=REPO_ROOT, env=env, stdout=lf,
                                    stderr=subprocess.STDOUT)
            running[j["name"]] = (j, proc, lf, time.perf_counter())
            print(f"[{ndone:3d}/{len(jobs)}] BAŞLA  {j['name']} | {stamp()}", flush=True)
        time.sleep(0.5)
        for name in list(running):
            j, proc, lf, t0 = running[name]
            rc = proc.poll()
            if rc is None:
                continue
            lf.close()
            del running[name]
            ndone += 1
            wall = time.perf_counter() - t0
            if rc == 0 and os.path.exists(done_path(j)):
                print(f"[{ndone:3d}/{len(jobs)}] BİTTİ  {name} {fmt_hms(wall)} | {stamp()}", flush=True)
            else:
                failed.append(name)
                print(f"[{ndone:3d}/{len(jobs)}] HATA   {name} (çıkış kodu {rc}) → "
                      f"{os.path.join(log_dir, name + '.log')} | {stamp()}", flush=True)

    rc_eval = 0
    if not args.no_eval:
        print(f"\nDeğerlendirme başlıyor | {stamp()}\n  {' '.join(eval_argv[1:])}", flush=True)
        rc_eval = subprocess.call(eval_argv, cwd=REPO_ROOT, env=env)
        print(f"Değerlendirme bitti (çıkış kodu {rc_eval}) | {stamp()}", flush=True)

    if failed:
        print(f"\nBAŞARISIZ İŞLER ({len(failed)}):")
        for n in failed:
            print(f"  - {n} (log: {os.path.join(log_dir, n + '.log')})")
        return 1
    return 1 if rc_eval else 0


if __name__ == "__main__":
    sys.exit(main())
