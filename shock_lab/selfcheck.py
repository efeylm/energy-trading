"""
Hızlı doğruluk kontrolleri (birkaç dakika):

  1. Rejim uygulaması: (d,s) = (2,1) → tüketici yükleri ×2, başka hiçbir şey değişmez;
     (1,0.5) → üretici PV ×0.5, başka hiçbir şey değişmez; ajan kimlik/türleri ve
     RNG çekiliş sırası aynı.
  2. src eşdeğerliği: q_normal (ε çarpanı 0.985) ↔ src QLearningEnv.train ve
     nn_normal (σ çarpanı 0.993) ↔ src REINFORCEEnv.train, aynı seed, ilk N episode
     sistem ödülleri birebir aynı olmalı.
  3. Süre ölçümü: Q ve NN için ms/episode (verilen popülasyonda).

    python -m shock_lab.selfcheck --producers 10 --consumers 10 --q-episodes 30 --nn-episodes 5
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from shock_lab.common import (chdir_repo_root, ShockQLEnv, NORMAL, silenced, log,
                              DEFAULT_TRAIN_REGIMES, regime_probs, DEFAULT_NORMAL_PROB)
from src.config import SimConfig
from src.seeding import seed_everything


def check_regime(p: int, c: int) -> bool:
    ok = True
    cfg = SimConfig(n_producers=p, n_consumers=c, seed=7)
    env = ShockQLEnv(cfg)
    snaps = {}
    for r in [(1.0, 1.0), (2.0, 1.0), (1.0, 0.5), (0.5, 2.0)]:
        env.set_regime(r)
        with silenced():
            obs = env.reset(seed=123)
        snaps[r] = (
            np.array([env.profiles.load_profiles[i] for i in range(cfg.n_agents)]),
            np.array([env.profiles.pv_profiles[i] for i in range(cfg.n_agents)]),
            [(a.agent_id, a.agent_type) for a in env.agents],
            np.array([obs[i].inflexible_load for i in range(cfg.n_agents)]),
        )
    L0, P0, A0, O0 = snaps[(1.0, 1.0)]
    for r, (L, P, A, O) in snaps.items():
        d, s = r
        expL = L0.copy(); expL[p:] *= d
        expP = P0.copy(); expP[:p] *= s
        good = (np.array_equal(L, expL) and np.array_equal(P, expP) and A == A0
                and np.allclose(O, expL[:, 0] - expP[:, 0]))
        cons_ratio = L[p:].sum() / L0[p:].sum()
        prod_load_same = np.array_equal(L[:p], L0[:p])
        pv_ratio = P[:p].sum() / P0[:p].sum() if P0[:p].sum() > 0 else float("nan")
        cons_pv_zero = not P[p:].any()
        log(f"  rejim {r}: tüketici yük oranı {cons_ratio:.6f} | üretici yükü aynı {prod_load_same} "
            f"| üretici PV oranı {pv_ratio:.6f} | tüketici PV=0 {cons_pv_zero} "
            f"| ajan kimlik/türleri aynı {A == A0} | ilk gözlem ölçekli {np.allclose(O, expL[:, 0] - expP[:, 0])} "
            f"→ {'OK' if good else 'HATA'}")
        ok &= bool(good)
    return ok


def check_q_equivalence(p: int, c: int, n: int) -> bool:
    from src.ql_environment import QLearningEnv
    from src.ql_agent import QLearningAgent
    cfg = SimConfig(n_producers=p, n_consumers=c, seed=3)
    seed_everything(cfg.seed)
    with silenced():
        ref = QLearningEnv(cfg).train(n_episodes=n, verbose=False)
    from shock_lab.train import make_meta, train
    meta = make_meta("q_normal", 3, p, c, n, [NORMAL], np.ones(1), eps_decay=0.985)
    t0 = time.perf_counter()
    env, series, bal, _ = train(meta, progress=False)
    mine = series["system_reward"]
    diff = float(np.max(np.abs(np.asarray(ref) - mine)))
    same_q = True
    log(f"  Q: {n} episode | src ilk 3 ödül {np.round(ref[:3], 6).tolist()} | shock_lab "
        f"{np.round(mine[:3], 6).tolist()} | maks |fark| {diff:.3e} | denge ihlali {bal['n_violations']}")
    return diff == 0.0 and same_q


def check_nn_equivalence(p: int, c: int, n: int) -> bool:
    from shock_lab.train import prepare_torch, make_meta, train
    prepare_torch()
    from src.reinforce_environment import REINFORCEEnv
    cfg = SimConfig(n_producers=p, n_consumers=c, seed=3)
    seed_everything(cfg.seed)
    with silenced():
        ref = REINFORCEEnv(cfg).train(n_episodes=n, verbose=False)
    meta = make_meta("nn_normal", 3, p, c, n, [NORMAL], np.ones(1), sigma_decay=0.993)
    env, series, bal, _ = train(meta, progress=False)
    mine = series["system_reward"]
    diff = float(np.max(np.abs(np.asarray(ref) - mine)))
    log(f"  NN: {n} episode | src ilk 3 ödül {np.round(ref[:3], 6).tolist()} | shock_lab "
        f"{np.round(mine[:3], 6).tolist()} | maks |fark| {diff:.3e} | denge ihlali {bal['n_violations']}")
    return diff == 0.0


def timing(p: int, c: int, nq: int, nnn: int) -> None:
    from shock_lab.train import make_meta, train
    regs = list(DEFAULT_TRAIN_REGIMES)
    probs = regime_probs(regs, DEFAULT_NORMAL_PROB)
    for variant, n in (("q_blind", nq), ("q_price", nq), ("nn_price", nnn), ("nn_oracle", nnn)):
        meta = make_meta(variant, 1, p, c, n, regs, probs)
        _, _, _, t = train(meta, progress=False)
        log(f"  süre {variant:<9} {p}+{c}: {t['seconds_per_episode'] * 1000:7.1f} ms/episode ({n} ep)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--producers", type=int, default=10)
    ap.add_argument("--consumers", type=int, default=10)
    ap.add_argument("--q-episodes", type=int, default=30)
    ap.add_argument("--nn-episodes", type=int, default=5)
    ap.add_argument("--no-timing", action="store_true")
    args = ap.parse_args()
    chdir_repo_root()
    p, c = args.producers, args.consumers
    results = {}
    log("1) Rejim uygulaması")
    results["regime"] = check_regime(p, c)
    log("2) src eşdeğerliği")
    results["q_equiv"] = check_q_equivalence(p, c, args.q_episodes)
    results["nn_equiv"] = check_nn_equivalence(p, c, args.nn_episodes)
    if not args.no_timing:
        log("3) Süre")
        timing(p, c, args.q_episodes, args.nn_episodes)
    log("SONUÇ: " + ", ".join(f"{k}={'OK' if v else 'HATA'}" for k, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
