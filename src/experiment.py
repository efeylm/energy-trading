"""
Tek bir deney işi: bir popülasyon + bir seed için seçilen yöntemleri çalıştırır
ve sonucu tek bir JSON dosyasına yazar.

Kullanım:
    python -m src.experiment --producers 10 --consumers 20 --seed 1 \\
        --methods random,ql,nn --episodes 3000 --nn_episodes 7000 \\
        --out results/A_p10c20_seed1.json --tag A

Yöntemler (her zaman bu sırayla çalışır, --methods sırası önemsizdir):
    random : rastgele ZI baseline, aynı gün üzerinde --baseline_draws çekiliş ortalaması
    ql     : tabular Q-Learning, --episodes eğitim + greedy (ε=0) değerlendirme
    nn     : REINFORCE politika ağı, --nn_episodes eğitim + deterministik (σ=0)
             değerlendirme; ağ şekli --nn_layers × --nn_hidden

TEKRARLANABİLİRLİK: Her yöntemden ÖNCE seed_everything(seed) çağrılır. Böylece
bir yöntemin sonucu aynı süreçte başka hangi yöntemlerin çalıştığına bağlı
değildir (ör. "--methods nn" ile "--methods random,ql,nn" aynı NN sonucunu verir).
Seed'in anlamı repo genelindekiyle aynıdır: eğitim günleri seed + episode,
değerlendirme günü seed (SimConfig.seed).

Ortam her adımda çok fazla çıktı basar; bu çıktı atılır, yalnızca kısa ilerleme
satırları yazdırılır. torch yalnızca "nn" istendiğinde içe aktarılır ve tek
iş parçacığına (CPU) sabitlenir.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from src.config import SimConfig
from src.profiles import ProfileManager
from src.random_environment import community_balance, run_random_baseline, _draw_values, METRIC_ROWS
from src.seeding import seed_everything

METHOD_ORDER = ["random", "ql", "nn"]
METRIC_KEYS = [key for key, _, _ in METRIC_ROWS]


class _NullWriter:
    """stdout çöpü — StringIO'dan farklı olarak bellekte birikmez."""

    def write(self, s: str) -> int:
        return len(s)

    def flush(self) -> None:
        pass


# Gerçek stdout'u yönlendirmeden ÖNCE sakla; ilerleme satırları buraya yazılır.
_OUT = sys.stdout


def _log(msg: str) -> None:
    _OUT.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    _OUT.flush()


def _silenced():
    return contextlib.redirect_stdout(_NullWriter())


def _attach_progress(env, name: str, n_episodes: int) -> None:
    """env.end_episode'u sarar: eğitimin ~her %10'unda bir ilerleme satırı.

    end_episode RNG tüketmez; sarmak sonuçları değiştirmez. Eğitim döngüsünün
    kendisine (train()) dokunulmaz.
    """
    orig = env.end_episode
    every = max(1, n_episodes // 10)
    state = {"ep": 0, "t0": time.perf_counter()}

    def end_episode_with_progress():
        orig()
        state["ep"] += 1
        ep = state["ep"]
        if ep % every == 0 or ep == n_episodes:
            el = time.perf_counter() - state["t0"]
            eta = el / ep * (n_episodes - ep)
            _log(f"  {name}: episode {ep}/{n_episodes} ({ep / n_episodes * 100:3.0f}%) "
                 f"| {el:8.1f}s geçti | kalan ~{eta:8.1f}s")

    env.end_episode = end_episode_with_progress


def _metrics_dict(m, config: SimConfig, e_max: float) -> Dict[str, float]:
    """Makale tablolarındaki skaler metrikler + E_max + η (config.n_producers ile)."""
    return {k: float(v) for k, v in _draw_values(m, config.n_producers, e_max).items()}


# ---------------------------------------------------------------------------
# Yöntemler
# ---------------------------------------------------------------------------

def run_random(config: SimConfig, n_draws: int) -> Dict:
    _log(f"random: başlıyor ({n_draws} fiyat çekilişi)")
    t0, c0 = time.perf_counter(), time.process_time()
    with _silenced():
        res = run_random_baseline(config, n_draws=n_draws, seed=config.seed, verbose=False)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    stats = {k: res.stats(k) for k in METRIC_KEYS}
    _log(f"random: bitti ({wall:.1f}s) | η={stats['efficiency_pct']['mean']:.2f}%")
    return {
        "n_draws": n_draws,
        "metrics": {k: stats[k]["mean"] for k in METRIC_KEYS},
        "draw_std": {k: stats[k]["std"] for k in METRIC_KEYS},
        "draw_min": {k: stats[k]["min"] for k in METRIC_KEYS},
        "draw_max": {k: stats[k]["max"] for k in METRIC_KEYS},
        "train_seconds": 0.0,
        "eval_seconds": wall,
        "cpu_seconds": cpu,
        "train_curve": None,
    }


def _run_learner(name: str, env, config: SimConfig, n_episodes: int, e_max: float) -> Dict:
    _attach_progress(env, name, n_episodes)
    t0, c0 = time.perf_counter(), time.process_time()
    with _silenced():
        curve = env.train(n_episodes=n_episodes, verbose=False)
    t1 = time.perf_counter()
    with _silenced():
        m = env.run_evaluation(seed=config.seed, verbose=False)
    t2, c2 = time.perf_counter(), time.process_time()
    metrics = _metrics_dict(m, config, e_max)
    _log(f"{name}: bitti (eğitim {t1 - t0:.1f}s, değerlendirme {t2 - t1:.2f}s) "
         f"| η={metrics['efficiency_pct']:.2f}%")
    return {
        "episodes": n_episodes,
        "metrics": metrics,
        "train_seconds": t1 - t0,
        "eval_seconds": t2 - t1,
        "cpu_seconds": c2 - c0,
        "train_curve": [float(x) for x in curve],
    }


def run_ql(config: SimConfig, n_episodes: int, e_max: float) -> Dict:
    from src.ql_environment import QLearningEnv
    _log(f"ql: başlıyor ({n_episodes} episode)")
    return _run_learner("ql", QLearningEnv(config), config, n_episodes, e_max)


def run_nn(config: SimConfig, n_episodes: int, e_max: float,
           n_layers: int, hidden: int) -> Dict:
    from src.reinforce_environment import REINFORCEEnv
    _log(f"nn: başlıyor ({n_episodes} episode, {n_layers}×{hidden} gizli katman)")
    env = REINFORCEEnv(config, hidden=hidden, n_layers=n_layers)
    out = _run_learner("nn", env, config, n_episodes, e_max)
    # Tek bir politika ağının (ör. ajan 0'ın satıcı ağı) parametre sayısı;
    # her ajanın biri satıcı biri alıcı olmak üzere 2 ağı vardır.
    net = env._rf_agents[0].policy_net
    out["n_layers"] = n_layers
    out["hidden"] = hidden
    out["n_params_per_net"] = int(sum(p.numel() for p in net.parameters()))
    return out


# ---------------------------------------------------------------------------
# İş
# ---------------------------------------------------------------------------

def _versions(with_torch: bool) -> Dict[str, Optional[str]]:
    v = {"python": platform.python_version(), "numpy": np.__version__, "torch": None}
    if with_torch:
        import torch
        v["torch"] = torch.__version__
    return v


def run_job(args) -> Dict:
    methods = [m for m in METHOD_ORDER if m in args.methods]
    config = SimConfig(n_producers=args.producers, n_consumers=args.consumers, seed=args.seed)

    if "nn" in methods:
        import torch
        torch.set_num_threads(1)   # tek iş parçacığı; ağ çok küçük, CPU yeterli

    # Değerlendirme günü (gün seed'i = config.seed) için profil tabanlı denge
    day = community_balance(
        ProfileManager(config.n_producers, config.n_consumers, seed=config.seed), config
    )
    e_max = day["e_max_kwh"]

    _log(f"İŞ başlıyor: {args.tag or '-'} | {config.n_producers}+{config.n_consumers} "
         f"| seed={config.seed} | yöntemler={','.join(methods)} | E_max={e_max:.2f} kWh")
    t_job = time.perf_counter()
    results: Dict[str, Dict] = {}
    for m in methods:
        seed_everything(config.seed)   # her yöntem aynı global RNG durumundan başlar
        if m == "random":
            results[m] = run_random(config, args.baseline_draws)
        elif m == "ql":
            results[m] = run_ql(config, args.episodes, e_max)
        elif m == "nn":
            results[m] = run_nn(config, args.nn_episodes, e_max, args.nn_layers, args.nn_hidden)
    total = time.perf_counter() - t_job
    _log(f"İŞ bitti: toplam {total:.1f}s")

    return {
        "format": "energy-trading-experiment/1",
        "tag": args.tag,
        "population": {"producers": config.n_producers, "consumers": config.n_consumers,
                       "label": f"{config.n_producers}+{config.n_consumers}"},
        "seed": config.seed,
        "methods": methods,
        "budget": {"ql_episodes": args.episodes, "nn_episodes": args.nn_episodes,
                   "baseline_draws": args.baseline_draws},
        "network": {"n_layers": args.nn_layers, "hidden": args.nn_hidden},
        "tariffs": {"fit_price": config.fit_price, "tou_price": config.tou_price},
        "sim_config": dataclasses.asdict(config),
        "community": day,
        "results": results,
        "total_wall_seconds": total,
        "versions": _versions("nn" in methods),
        "hostname": socket.gethostname(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv[1:],
    }


def write_json_atomic(data: Dict, path: str) -> None:
    """Önce geçici dosyaya yaz, sonra yeniden adlandır (yarım JSON kalmaz)."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _parse_methods(s: str) -> List[str]:
    ms = [m.strip() for m in s.split(",") if m.strip()]
    bad = [m for m in ms if m not in METHOD_ORDER]
    if bad or not ms:
        raise argparse.ArgumentTypeError(f"geçersiz yöntem(ler): {bad} (random, ql, nn)")
    return ms


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tek deney işi (popülasyon × seed × yöntemler)")
    p.add_argument("--producers", type=int, default=4, help="Üretici sayısı (varsayılan: 4)")
    p.add_argument("--consumers", type=int, default=4, help="Tüketici sayısı (varsayılan: 4)")
    p.add_argument("--seed", type=int, default=1, help="Gün/profil + global RNG seed'i (varsayılan: 1)")
    p.add_argument("--methods", type=_parse_methods, default=list(METHOD_ORDER),
                   help="Virgülle ayrılmış alt küme: random,ql,nn (varsayılan: hepsi)")
    p.add_argument("--episodes", type=int, default=3000, help="Q-Learning episode sayısı (varsayılan: 3000)")
    p.add_argument("--nn_episodes", type=int, default=7000, help="REINFORCE episode sayısı (varsayılan: 7000)")
    p.add_argument("--nn_layers", type=int, default=2, help="Politika ağı gizli katman sayısı (varsayılan: 2)")
    p.add_argument("--nn_hidden", type=int, default=64, help="Gizli katman genişliği (varsayılan: 64)")
    p.add_argument("--baseline_draws", type=int, default=200,
                   help="Random baseline fiyat çekilişi sayısı (varsayılan: 200)")
    p.add_argument("--out", type=str, required=True, help="Sonuç JSON dosyası")
    p.add_argument("--tag", type=str, default="", help="Serbest etiket (ör. deney adı A/B/C)")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    data = run_job(args)
    write_json_atomic(data, args.out)
    _log(f"JSON yazıldı: {args.out}")
