"""
Ablasyon planının tamamını paralel çalıştırır (her iş ayrı bir süreç), sonra özeti üretir.

Varsayılan plan: popülasyon 10+10, seed 1..5, tüm varyantlar (8) → 40 iş.
    Q varyantları 3000 episode (×2 varyantlar 6000), NN varyantları 7000 episode.

Kullanım (repo kökünden):
    python -m ablation.run_all --dry-run                       # plan + süre tahmini
    nohup python -m ablation.run_all --workers 16 > ablation/run_all.out 2>&1 &
    python -m ablation.run_all --smoke --workers 4             # küçük deneme → ablation/results_smoke
    python -m ablation.run_all --populations 10+10,4+4 --seeds 1,2,3 --variants q_base,nn_noprice

Her iş: sys.executable -m ablation.run_job ... (OMP/MKL/OPENBLAS_NUM_THREADS=1).
Çıktı: <out>/<iş>.json, <out>/logs/<iş>.log. Devam ettirilebilir: JSON'u olan iş atlanır.
Sonuç yalnızca (popülasyon, seed, varyant, bütçe)'ye bağlıdır; işçi sayısı / sıra etkisizdir
(her iş kendi sürecinde seed_everything ile başlar).
"""

from __future__ import annotations

import argparse
import heapq
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ablation.variants import (VARIANTS, VARIANT_ORDER, REPO_ROOT, episodes_for, job_name,
                               parse_variants, fmt_hms)

FULL = dict(populations=[(10, 10)], seeds=[1, 2, 3, 4, 5], episodes_q=3000, episodes_nn=7000,
            out="ablation/results")
SMOKE = dict(populations=[(4, 4)], seeds=[1, 2], episodes_q=20, episodes_nn=20,
             out="ablation/results_smoke")

# Süre tahmini (yalnızca --dry-run ve sıralama için): ajan × episode başına ms.
# Kaynak: yayımlanmış results/A_p10c10_* (aynı sunucu: QL ≈ 20 ms/ep, NN ≈ 136 ms/ep, 20 ajan)
# ve M1 dizüstünde ölçülen göreli oranlar (nohist / noprice ≈ 0.67 × full).
MS_PER_AGENT_EP = {"q": 1.0, "nn_full": 6.8, "nn_small": 4.6}


@dataclass
class Job:
    variant: str
    producers: int
    consumers: int
    seed: int
    episodes_q: int
    episodes_nn: int

    @property
    def name(self) -> str:
        return job_name(self.variant, self.producers, self.consumers, self.seed)

    @property
    def episodes(self) -> int:
        return episodes_for(self.variant, self.episodes_q, self.episodes_nn)

    def est_seconds(self) -> float:
        v = VARIANTS[self.variant]
        if v["family"] == "q":
            k = MS_PER_AGENT_EP["q"]
        else:
            k = MS_PER_AGENT_EP["nn_full" if v["history"] else "nn_small"]
        return k * (self.producers + self.consumers) * self.episodes / 1000.0 + 3.0

    def argv(self, out_dir: str) -> List[str]:
        return [sys.executable, "-m", "ablation.run_job",
                "--producers", str(self.producers), "--consumers", str(self.consumers),
                "--seed", str(self.seed), "--variant", self.variant,
                "--episodes-q", str(self.episodes_q), "--episodes-nn", str(self.episodes_nn),
                "--out", out_dir]


def parse_pops(text: str) -> List[Tuple[int, int]]:
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok:
            a, b = tok.split("+")
            out.append((int(a), int(b)))
    return out


def build_jobs(pops, seeds, variants, episodes_q, episodes_nn) -> List[Job]:
    jobs = [Job(v, p, c, s, episodes_q, episodes_nn)
            for (p, c) in pops for s in seeds for v in VARIANT_ORDER if v in variants]
    # En uzun (NN) işler önce; eşitlikte ada göre (deterministik sıra)
    jobs.sort(key=lambda j: (-j.est_seconds(), j.name))
    return jobs


def makespan(costs: List[float], workers: int) -> float:
    """Verilen sırayla açgözlü zamanlama (run_all'ın yaptığı gibi) → tahmini duvar süresi."""
    heap = [0.0] * max(1, workers)
    for c in costs:
        t = heapq.heappop(heap)
        heapq.heappush(heap, t + c)
    return max(heap)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ablasyon planını paralel çalıştır")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--populations", default=None, help="ör. 10+10,4+4 (varsayılan 10+10)")
    ap.add_argument("--seeds", default=None, help="ör. 1,2,3,4,5")
    ap.add_argument("--variants", default=",".join(VARIANT_ORDER))
    ap.add_argument("--episodes-q", type=int, default=None, help="Q temel bütçesi (varsayılan 3000)")
    ap.add_argument("--episodes-nn", type=int, default=None, help="NN bütçesi (varsayılan 7000)")
    ap.add_argument("--out", default=None, help="çıktı dizini (varsayılan ablation/results)")
    ap.add_argument("--dry-run", action="store_true", help="planı ve süre tahminini yaz, çalıştırma")
    ap.add_argument("--smoke", action="store_true", help="küçük bütçe (4+4, seed 1-2, 20 ep) → results_smoke")
    ap.add_argument("--no-summary", action="store_true", help="sonda özeti çalıştırma")
    args = ap.parse_args(argv)

    plan = dict(SMOKE if args.smoke else FULL)
    pops = parse_pops(args.populations) if args.populations else plan["populations"]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else plan["seeds"]
    variants = parse_variants(args.variants)
    eq = args.episodes_q if args.episodes_q is not None else plan["episodes_q"]
    enn = args.episodes_nn if args.episodes_nn is not None else plan["episodes_nn"]
    out = args.out or plan["out"]
    out_dir = out if os.path.isabs(out) else os.path.join(REPO_ROOT, out)
    log_dir = os.path.join(out_dir, "logs")
    jobs = build_jobs(pops, seeds, variants, eq, enn)
    workers = max(1, args.workers)
    total = len(jobs)

    def out_path(j: Job) -> str:
        return os.path.join(out_dir, j.name + ".json")

    todo = [j for j in jobs if not os.path.exists(out_path(j))]
    est = [j.est_seconds() for j in todo]
    print(f"{total} iş ({len(todo)} çalışacak) → {out_dir} | workers={workers} | "
          f"popülasyonlar={','.join(f'{p}+{c}' for p, c in pops)} seed'ler={seeds} | Q {eq} ep, NN {enn} ep",
          flush=True)
    print(f"Tahmin (yayımlanmış sonuçların sunucusuna göre; çekirdek ≥ işçi varsayımıyla): "
          f"toplam iş süresi {fmt_hms(sum(est))}, duvar saati ≈ {fmt_hms(makespan(est, workers))}",
          flush=True)
    if args.dry_run:
        for i, j in enumerate(jobs, 1):
            state = "VAR (atlanacak)" if os.path.exists(out_path(j)) else "bekliyor"
            print(f"  {i:3d}. {j.name:<28} {j.producers:>2}+{j.consumers:<2} seed={j.seed} "
                  f"ep={j.episodes:<6} tahmin≈{fmt_hms(j.est_seconds())} [{state}]")
        return 0

    os.makedirs(log_dir, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONUNBUFFERED="1")
    t_start = time.perf_counter()

    def stamp() -> str:
        return f"geçen {fmt_hms(time.perf_counter() - t_start)}"

    done = total - len(todo)
    pending = list(todo)
    running: Dict[str, tuple] = {}
    failed: List[str] = []
    while pending or running:
        while pending and len(running) < workers:
            j = pending.pop(0)
            log = open(os.path.join(log_dir, j.name + ".log"), "w", encoding="utf-8")
            proc = subprocess.Popen(j.argv(out_dir), cwd=REPO_ROOT, env=env,
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
                print(f"[{done:3d}/{total}] HATA   {name} (çıkış kodu {rc}) → "
                      f"{os.path.join(log_dir, name + '.log')} | {stamp()}", flush=True)

    print(f"\nEğitim bitti: duvar saati {fmt_hms(time.perf_counter() - t_start)}", flush=True)
    rc = 0
    if failed:
        print(f"BAŞARISIZ İŞLER ({len(failed)}):")
        for n in failed:
            print(f"  - {n}  (log: {os.path.join(log_dir, n + '.log')})")
        rc = 1
    if not args.no_summary:
        print("\nÖzet üretiliyor...", flush=True)
        r = subprocess.run([sys.executable, "-m", "ablation.summarize", out_dir], cwd=REPO_ROOT, env=env)
        rc = rc or r.returncode
    return rc


if __name__ == "__main__":
    sys.exit(main())
