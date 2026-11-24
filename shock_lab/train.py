"""
Tek bir ajan setini (bir varyant × bir seed) eğitir ve politikaları kaydeder.

    python -m shock_lab.train --variant q_price --seed 1 --producers 10 --consumers 10 \\
        --episodes 40000 --out shock_lab/runs

Çıktı: <out>/<variant>_p<P>c<C>_seed<S>/
    run.json           yapılandırma, argv, makine adı, süreler, özet, enerji dengesi
    policy.npz | .pt   Q-tabloları (sell_i / buy_i) veya NN state_dict'leri
    train_metrics.npz  episode başına eğitim metrikleri (d, s, fiyat, P2P, E_max, ...)
run.json EN SON yazılır; varsa iş tamamlanmış sayılır (run_all atlar).

Gün yapısı src ile aynıdır: episode k'nin gün seed'i = seed + k; her episode
reset → 48 adım → end_episode; ödül, piyasa, α/γ src varsayılanları. Global RNG'ler
başta bir kez seed_everything(seed) ile sabitlenir. Rejim örneklemesi AYRI bir
RNG (default_rng([seed, REGIME_SALT])) ile yapılır; ε-greedy / ağ gürültüsü
çekilişlerini etkilemez.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import platform
import socket
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from shock_lab.common import (
    chdir_repo_root, VARIANTS, VARIANT_ORDER, DEFAULT_TRAIN_REGIMES, DEFAULT_NORMAL_PROB,
    DEFAULT_PRICE_THRESHOLDS, NORMAL, MAX_TRAIN_DAY_SEED, Regime, BalanceTracker,
    parse_regimes, regime_probs, derived_decay, run_day, log, write_json_atomic, rkey,
    ShockQLEnv, fmt_hms,
)
from src.config import SimConfig
from src.seeding import seed_everything

REGIME_SALT = 0x5EED5
DEFAULT_EPISODES = {"q": 40000, "nn": 40000}
SERIES = ["regime_idx", "d", "s", "avg_price", "p2p_kwh", "e_max_kwh", "eta_pct",
          "grid_buy_kwh", "grid_sell_kwh", "seller_revenue", "buyer_cost",
          "demand_kwh", "supply_kwh", "system_reward", "exploration"]

# src varsayılanları (QLearningEnv / REINFORCEEnv)
Q_DEFAULTS = dict(alpha_lr=0.1, gamma=0.95, eps_start=1.0, eps_min=0.05)
NN_DEFAULTS = dict(lr=5e-4, gamma=0.95, sigma_start=0.25, sigma_min=0.05,
                   hidden=64, seq_len=12, d_model=16, n_layers=2)


def run_name(variant: str, p: int, c: int, seed: int) -> str:
    return f"{variant}_p{p}c{c}_seed{seed}"


def make_meta(variant: str, seed: int, producers: int, consumers: int, episodes: int,
              train_regimes: List[Regime], probs, explore_frac: float = 0.3,
              eps_decay: Optional[float] = None, sigma_decay: Optional[float] = None,
              price_thresholds=DEFAULT_PRICE_THRESHOLDS) -> Dict:
    v = VARIANTS[variant]
    kind = v["kind"]
    d_levels = sorted({r[0] for r in train_regimes})
    s_levels = sorted({r[1] for r in train_regimes})
    meta = {
        "variant": variant, "kind": kind, "seed": seed,
        "producers": producers, "consumers": consumers, "episodes": episodes,
        "randomized": v["randomized"],
        "train_regimes": [list(r) for r in train_regimes],
        "train_probs": [float(p) for p in probs],
        "explore_frac": explore_frac,
    }
    if kind == "q":
        meta["features"] = list(v["features"])
        meta["price_thresholds"] = [float(x) for x in price_thresholds]
        meta["d_levels"], meta["s_levels"] = d_levels, s_levels
        dec = eps_decay if eps_decay is not None else derived_decay(
            Q_DEFAULTS["eps_start"], Q_DEFAULTS["eps_min"], episodes, explore_frac)
        meta["hyper"] = dict(Q_DEFAULTS, eps_decay=dec,
                             eps_decay_source="explicit" if eps_decay is not None else "derived")
    else:
        meta["oracle"] = bool(v["oracle"])
        dec = sigma_decay if sigma_decay is not None else derived_decay(
            NN_DEFAULTS["sigma_start"], NN_DEFAULTS["sigma_min"], episodes, explore_frac)
        meta["hyper"] = dict(NN_DEFAULTS, sigma_decay=dec,
                             sigma_decay_source="explicit" if sigma_decay is not None else "derived")
    return meta


def build_env(meta: Dict):
    """run.json'daki meta'dan ortamı (ve ajanları) kurar. NN için torch içe aktarılır."""
    config = SimConfig(n_producers=meta["producers"], n_consumers=meta["consumers"],
                       seed=meta["seed"])
    h = meta["hyper"]
    if meta["kind"] == "q":
        return ShockQLEnv(config, features=meta["features"],
                          price_thresholds=meta["price_thresholds"],
                          d_levels=meta["d_levels"], s_levels=meta["s_levels"],
                          alpha_lr=h["alpha_lr"], gamma=h["gamma"], epsilon=h["eps_start"],
                          epsilon_min=h["eps_min"], epsilon_decay=h["eps_decay"])
    from shock_lab.nn_agents import ShockNNEnv
    return ShockNNEnv(config, oracle=meta["oracle"], lr=h["lr"], gamma=h["gamma"],
                      sigma=h["sigma_start"], sigma_min=h["sigma_min"],
                      sigma_decay=h["sigma_decay"], hidden=h["hidden"], seq_len=h["seq_len"],
                      d_model=h["d_model"], n_layers=h["n_layers"])


def prepare_torch() -> None:
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def train(meta: Dict, progress: bool = True, label: str = ""):
    """Eğitimi çalıştırır. Döner: (env, series dict, balance dict, timings dict)."""
    if meta["kind"] == "nn":
        prepare_torch()            # seed_everything torch'u yalnızca yüklüyse tohumlar
    seed = meta["seed"]
    n = meta["episodes"]
    if seed + n >= MAX_TRAIN_DAY_SEED:
        raise ValueError("eğitim gün seed'leri test seed'leriyle çakışabilir")
    seed_everything(seed)
    env = build_env(meta)

    regimes = [tuple(r) for r in meta["train_regimes"]]
    probs = np.asarray(meta["train_probs"], dtype=float)
    rng = np.random.default_rng([seed, REGIME_SALT])
    if len(regimes) == 1:
        idx_seq = np.zeros(n, dtype=np.int64)
    else:
        idx_seq = rng.choice(len(regimes), size=n, p=probs)

    series = {k: np.zeros(n, dtype=np.float64) for k in SERIES}
    bal = BalanceTracker()
    every = max(1, n // 100)
    t0, c0 = time.perf_counter(), time.process_time()
    for ep in range(n):
        r = regimes[int(idx_seq[ep])]
        explore = env.exploration()          # bu episode'da kullanılan ε / σ
        day, viol = run_day(env, seed + ep, r, learn=True)
        bal.add(day, viol)
        sysrew = float(sum(a.episode_rewards[-1] for a in env.agents))
        series["regime_idx"][ep] = idx_seq[ep]
        series["d"][ep], series["s"][ep] = r
        for k in ("avg_price", "p2p_kwh", "e_max_kwh", "eta_pct", "grid_buy_kwh",
                  "grid_sell_kwh", "seller_revenue", "buyer_cost", "demand_kwh", "supply_kwh"):
            series[k][ep] = day[k]
        series["system_reward"][ep] = sysrew
        series["exploration"][ep] = explore
        done_eps = ep + 1
        if progress and (done_eps % every == 0 or done_eps == n):
            el = time.perf_counter() - t0
            eta = el / done_eps * (n - done_eps)
            lo = max(0, done_eps - every)
            log(f"{label} ep {done_eps}/{n} ({done_eps / n * 100:5.1f}%) | geçen {fmt_hms(el)} "
                f"| kalan ~{fmt_hms(eta)} | keşif {explore:.3f} | son blok: fiyat "
                f"{series['avg_price'][lo:done_eps].mean():.4f} η "
                f"{series['eta_pct'][lo:done_eps].mean():5.1f}% ödül "
                f"{series['system_reward'][lo:done_eps].mean():8.2f} | denge ihlali {bal.n_viol}")
    timings = {"train_wall_seconds": time.perf_counter() - t0,
               "train_cpu_seconds": time.process_time() - c0}
    timings["seconds_per_episode"] = timings["train_wall_seconds"] / max(1, n)
    return env, series, bal.as_dict(), timings


def summarize_series(series: Dict[str, np.ndarray], regimes: List[Regime]) -> Dict:
    n = len(series["avg_price"])
    nb = min(100, n)
    edges = np.linspace(0, n, nb + 1).astype(int)
    blocks = {k: [float(series[k][a:b].mean()) for a, b in zip(edges[:-1], edges[1:]) if b > a]
              for k in ("avg_price", "eta_pct", "p2p_kwh", "system_reward", "exploration")}
    tail = slice(int(n * 0.9), n)
    per_regime = {}
    for i, r in enumerate(regimes):
        mask = series["regime_idx"][tail] == i
        cnt_all = int((series["regime_idx"] == i).sum())
        if mask.sum() == 0:
            per_regime[rkey(r)] = {"n_total": cnt_all, "n_last10pct": 0}
            continue
        per_regime[rkey(r)] = {
            "n_total": cnt_all, "n_last10pct": int(mask.sum()),
            **{k: float(series[k][tail][mask].mean())
               for k in ("avg_price", "eta_pct", "p2p_kwh", "buyer_cost", "seller_revenue")},
        }
    return {"block_edges": [int(x) for x in edges], "block_means": blocks,
            "last10pct_by_regime": per_regime}


def save_policy(env, meta: Dict, run_dir: str) -> str:
    if meta["kind"] == "q":
        path = os.path.join(run_dir, "policy.npz")
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **env.policy_arrays())
        os.replace(tmp, path)
    else:
        import torch
        path = os.path.join(run_dir, "policy.pt")
        tmp = path + ".tmp"
        torch.save(env.policy_state(), tmp)
        os.replace(tmp, path)
    return path


def load_policy(env, meta: Dict, run_dir: str) -> None:
    if meta["kind"] == "q":
        with np.load(os.path.join(run_dir, "policy.npz")) as arrs:
            env.load_policy_arrays(arrs)
    else:
        import torch
        state = torch.load(os.path.join(run_dir, "policy.pt"), map_location="cpu",
                           weights_only=False)
        env.load_policy_state(state)


def _versions(kind: str) -> Dict:
    v = {"python": platform.python_version(), "numpy": np.__version__, "torch": None}
    if kind == "nn":
        import torch
        v["torch"] = torch.__version__
    return v


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="shock_lab: bir varyantı rejim rastgeleleştirmesiyle eğit")
    p.add_argument("--variant", required=True, choices=VARIANT_ORDER)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--producers", type=int, default=10)
    p.add_argument("--consumers", type=int, default=10)
    p.add_argument("--episodes", type=int, default=None,
                   help="Episode sayısı (varsayılan: Q 40000, NN 40000)")
    p.add_argument("--out", default="shock_lab/runs")
    p.add_argument("--train-regimes", default=None,
                   help="Rastgeleleştirilmiş varyantlar için 'd:s,d:s,...' "
                        "(varsayılan: d,s ∈ {0.5,1,2} → 9 rejim). *_normal varyantları daima 1:1.")
    p.add_argument("--normal-prob", type=float, default=DEFAULT_NORMAL_PROB,
                   help="(1,1) rejiminin olasılığı; kalan kütle diğerlerine eşit (varsayılan 1/3)")
    p.add_argument("--regime-weights", default=None,
                   help="Rejim başına açık ağırlıklar 'w1,w2,...' (--normal-prob'u geçersiz kılar)")
    p.add_argument("--explore-frac", type=float, default=0.3,
                   help="ε / σ tabanına eğitimin bu kesrinde ulaşılır (varsayılan 0.3)")
    p.add_argument("--eps-decay", type=float, default=None,
                   help="Q: açık ε çarpanı (src: 0.985). Verilmezse --explore-frac'tan türetilir")
    p.add_argument("--sigma-decay", type=float, default=None,
                   help="NN: açık σ çarpanı (src: 0.993). Verilmezse --explore-frac'tan türetilir")
    p.add_argument("--price-thresholds", default="0.135,0.185",
                   help="q_price fiyat kovası eşikleri $/kWh (varsayılan 0.135,0.185)")
    p.add_argument("--force", action="store_true", help="run.json varsa bile yeniden eğit")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_root = os.path.abspath(args.out)
    chdir_repo_root()
    v = VARIANTS[args.variant]
    episodes = args.episodes if args.episodes is not None else DEFAULT_EPISODES[v["kind"]]
    if v["randomized"]:
        regimes = parse_regimes(args.train_regimes) if args.train_regimes else list(DEFAULT_TRAIN_REGIMES)
        weights = [float(x) for x in args.regime_weights.split(",")] if args.regime_weights else None
        probs = regime_probs(regimes, args.normal_prob, weights)
    else:
        regimes, probs = [NORMAL], np.ones(1)
    thr = [float(x) for x in args.price_thresholds.split(",")]
    meta = make_meta(args.variant, args.seed, args.producers, args.consumers, episodes,
                     regimes, probs, args.explore_frac, args.eps_decay, args.sigma_decay, thr)

    name = run_name(args.variant, args.producers, args.consumers, args.seed)
    run_dir = os.path.join(out_root, name)
    if os.path.exists(os.path.join(run_dir, "run.json")) and not args.force:
        log(f"{name}: run.json zaten var, atlanıyor (--force ile yeniden eğit)")
        return 0
    os.makedirs(run_dir, exist_ok=True)

    hp = meta["hyper"]
    dec = hp.get("eps_decay", hp.get("sigma_decay"))
    log(f"EĞİTİM başlıyor: {name} | {episodes} episode | rejimler "
        f"{', '.join(f'{rkey(r)}({p:.3f})' for r, p in zip(regimes, probs))} | keşif çarpanı {dec:.6f}")
    t_job = time.perf_counter()
    env, series, bal, timings = train(meta, progress=True, label=name)
    policy_path = save_policy(env, meta, run_dir)
    np.savez_compressed(os.path.join(run_dir, "train_metrics.npz"),
                        **{k: (series[k].astype(np.int16) if k == "regime_idx" else series[k])
                           for k in SERIES})
    timings["job_wall_seconds"] = time.perf_counter() - t_job
    if bal["n_violations"]:
        log(f"UYARI: {bal['n_violations']} eğitim gününde enerji dengesi tutmadı")
    data = {
        "format": "shock-lab-run/1",
        "name": name,
        "meta": meta,
        "regime_salt": REGIME_SALT,
        "day_seeds": [args.seed, args.seed + episodes - 1],
        "sim_config": dataclasses.asdict(env.config),
        "policy_file": os.path.basename(policy_path),
        "metrics_file": "train_metrics.npz",
        "train_summary": summarize_series(series, [tuple(r) for r in meta["train_regimes"]]),
        "balance_check": bal,
        "timings": timings,
        "versions": _versions(meta["kind"]),
        "hostname": socket.gethostname(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv[1:] if argv is None else list(argv),
    }
    write_json_atomic(data, os.path.join(run_dir, "run.json"))
    log(f"EĞİTİM bitti: {name} | {fmt_hms(timings['train_wall_seconds'])} "
        f"({timings['seconds_per_episode'] * 1000:.1f} ms/episode) | denge ihlali {bal['n_violations']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
