"""
Tek ablasyon işi: bir (popülasyon, seed, varyant) → tek JSON.

Kullanım (repo kökünden):
    python -m ablation.run_job --producers 10 --consumers 10 --seed 1 --variant nn_noprice \\
        --episodes-q 3000 --episodes-nn 7000 --out ablation/results

Çıktı: <out>/<varyant>_p<P>c<C>_seed<S>.json (atomik yazım). İlerleme satırları
stdout'a yazılır (run_all bunları <out>/logs/<iş>.log dosyasına yönlendirir).

PROTOKOL src/experiment.py ile AYNIDIR ve oradaki kod yeniden kullanılır:
    seed_everything(seed)  →  ortam kur  →  src.experiment._run_learner
        (env.train(n) + env.run_evaluation(seed=seed) + _metrics_dict + süreler)
E_max, değerlendirme gününün (gün seed'i = seed) profillerinden src run_job'daki
gibi hesaplanır. Tek fark: ilerleme satırı ~%2'de bir (src'de %10) — bunun için
src.experiment._attach_progress bu iş süresince ~%2'lik eşiyle değiştirilir;
sarmalayıcı RNG tüketmez, sonuçları etkilemez.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import socket
import sys
import time
from datetime import datetime
from typing import Dict, Optional, Tuple
from unittest import mock

from ablation.variants import (VARIANTS, chdir_repo_root, episodes_for, job_name, REPO_ROOT)

chdir_repo_root()

import src.experiment as sx                                    # noqa: E402
from src.config import SimConfig                               # noqa: E402
from src.profiles import make_profile_manager                  # noqa: E402
from src.random_environment import community_balance           # noqa: E402
from src.seeding import seed_everything                        # noqa: E402

FORMAT = "energy-trading-ablation/1"
_OUT = sys.stdout
BALANCE_REL_TOL = 1e-6


def _log(msg: str) -> None:
    _OUT.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    _OUT.flush()


def _make_progress(every_frac: float):
    """src.experiment._attach_progress'in daha sık (ör. %2) yazan eşi."""

    def attach(env, name: str, n_episodes: int) -> None:
        orig = env.end_episode
        every = max(1, int(round(n_episodes * every_frac)))
        state = {"ep": 0, "t0": time.perf_counter()}

        def end_episode_with_progress():
            orig()
            state["ep"] += 1
            ep = state["ep"]
            if ep % every == 0 or ep == n_episodes:
                el = time.perf_counter() - state["t0"]
                eta = el / ep * (n_episodes - ep)
                _log(f"  {name}: episode {ep}/{n_episodes} ({ep / n_episodes * 100:5.1f}%) "
                     f"| {el:8.1f}s geçti | {el / ep * 1000:7.1f} ms/ep | kalan ~{eta:8.1f}s")

        env.end_episode = end_episode_with_progress

    return attach


def community_for_eval_day(config: SimConfig) -> Dict[str, float]:
    """src run_job ile aynı: değerlendirme günü (gün seed'i = config.seed) dengesi."""
    return community_balance(make_profile_manager(config, config.seed), config)


def build_env(variant: str, config: SimConfig, hidden: int = 64, n_layers: int = 2):
    v = VARIANTS[variant]
    if v["family"] == "q":
        from ablation.q_agents import AblationQLEnv
        return AblationQLEnv(config, price_buckets=v["price_buckets"])
    from ablation.nn_agents import AblationNNEnv
    state_dim = None if v["history"] else v["state_dim"]
    return AblationNNEnv(config, state_dim=state_dim, hidden=hidden, n_layers=n_layers)


def run_variant(variant: str, config: SimConfig, n_episodes: int, e_max: float,
                hidden: int = 64, n_layers: int = 2,
                progress_frac: Optional[float] = 0.02) -> Tuple[Dict, object]:
    """seed_everything → ortam → src _run_learner. (sonuç sözlüğü, ortam) döndürür.

    NN varyantlarında torch, seed_everything'den ÖNCE içe aktarılır (src run_job gibi),
    böylece torch RNG'si de tohumlanır.
    """
    if VARIANTS[variant]["family"] == "nn":
        import ablation.nn_agents  # noqa: F401  (torch'u yükler)
    seed_everything(config.seed)
    env = build_env(variant, config, hidden=hidden, n_layers=n_layers)
    if progress_frac:
        with mock.patch.object(sx, "_attach_progress", _make_progress(progress_frac)):
            res = sx._run_learner(variant, env, config, n_episodes, e_max)
    else:
        res = sx._run_learner(variant, env, config, n_episodes, e_max)
    return res, env


def balance_check(env, config: SimConfig) -> Dict:
    """Değerlendirme günü enerji dengesi: D = P2P + şebeke alımı, S = P2P + şebeke satışı."""
    bal = community_balance(env.profiles, config)
    m = env.metrics
    p2p = float(m.total_internal_trading())
    gb, gs = float(m.total_grid_purchased()), float(m.total_grid_sold())
    dev_d = abs(bal["deficit_kwh"] - p2p - gb)
    dev_s = abs(bal["surplus_kwh"] - p2p - gs)
    ok = (dev_d <= BALANCE_REL_TOL * max(1.0, bal["deficit_kwh"])
          and dev_s <= BALANCE_REL_TOL * max(1.0, bal["surplus_kwh"]))
    return {"ok": bool(ok), "abs_D_minus_p2p_minus_gridbuy": dev_d,
            "abs_S_minus_p2p_minus_gridsell": dev_s, "rel_tol": BALANCE_REL_TOL}


def architecture(variant: str, env) -> Dict:
    v = VARIANTS[variant]
    if v["family"] == "q":
        from ablation.q_agents import n_q_states
        from src.ql_agent import N_ACTIONS
        n = n_q_states(v["price_buckets"])
        return {"kind": "tabular", "state_components": 3 if v["price_buckets"] else 2,
                "price_buckets": v["price_buckets"], "n_states_per_table": n,
                "n_actions": N_ACTIONS, "input_dim": n,
                "tables_per_agent": 2}
    from ablation.nn_agents import head_input_dim, n_params
    a0 = env._rf_agents[0]
    net = a0.policy_net
    return {"kind": "mlp", "state_dim": v["state_dim"], "history_encoder": bool(v["history"]),
            "head_input_dim": head_input_dim(net), "input_dim": head_input_dim(net),
            "n_params_per_net": n_params(net), "nets_per_agent": 2,
            "hidden": env.nn_hidden, "n_layers": env.nn_n_layers,
            "seq_len": env.nn_seq_len if v["history"] else 0,
            "d_model": env.nn_d_model if v["history"] else 0}


def run_job(args) -> Dict:
    v = VARIANTS[args.variant]
    config = SimConfig(n_producers=args.producers, n_consumers=args.consumers, seed=args.seed)
    if v["family"] == "nn":
        import torch
        torch.set_num_threads(1)   # tek iş parçacığı; ağ çok küçük
    n_ep = episodes_for(args.variant, args.episodes_q, args.episodes_nn)
    day = community_for_eval_day(config)
    e_max = day["e_max_kwh"]

    _log(f"İŞ başlıyor: {job_name(args.variant, args.producers, args.consumers, args.seed)} "
         f"| {v['label']} | {n_ep} episode | E_max={e_max:.2f} kWh")
    t_job = time.perf_counter()
    res, env = run_variant(args.variant, config, n_ep, e_max, hidden=args.nn_hidden,
                           n_layers=args.nn_layers, progress_frac=args.progress)
    arch = architecture(args.variant, env)
    diag = env.diagnostics() if hasattr(env, "diagnostics") else {}
    if v["family"] == "nn":
        diag = {"final_sigma": float(env._rf_agents[0].sigma)}
    bal = balance_check(env, config)
    total = time.perf_counter() - t_job
    _log(f"İŞ bitti: toplam {total:.1f}s | η={res['metrics']['efficiency_pct']:.2f}% "
         f"| denge {'OK' if bal['ok'] else 'İHLAL'}")

    return {
        "format": FORMAT,
        "variant": args.variant,
        "family": v["family"],
        "reference": v["ref"],
        "label": v["label"],
        "description": v["description"],
        "inputs": v["inputs"],
        "population": {"producers": config.n_producers, "consumers": config.n_consumers,
                       "label": f"{config.n_producers}+{config.n_consumers}"},
        "seed": config.seed,
        "episodes": n_ep,
        "budget": {"episodes_q": args.episodes_q, "episodes_nn": args.episodes_nn,
                   "ep_mult": v.get("ep_mult", 1)},
        "architecture": arch,
        "result": res,
        "diagnostics": diag,
        "balance_check": bal,
        "tariffs": {"fit_price": config.fit_price, "tou_price": config.tou_price},
        "sim_config": dataclasses.asdict(config),
        "community": day,
        "total_wall_seconds": total,
        "versions": sx._versions(v["family"] == "nn"),
        "hostname": socket.gethostname(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv[1:],
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tek ablasyon işi (popülasyon × seed × varyant)")
    p.add_argument("--producers", type=int, default=10)
    p.add_argument("--consumers", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--variant", required=True, choices=list(VARIANTS))
    p.add_argument("--episodes-q", type=int, default=3000, help="Q temel bütçesi (×ep_mult)")
    p.add_argument("--episodes-nn", type=int, default=7000, help="NN bütçesi")
    p.add_argument("--nn-hidden", type=int, default=64)
    p.add_argument("--nn-layers", type=int, default=2)
    p.add_argument("--progress", type=float, default=0.02, help="ilerleme satırı sıklığı (kesir)")
    p.add_argument("--out", default="ablation/results", help="çıktı dizini")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(REPO_ROOT, args.out)
    path = os.path.join(out_dir, job_name(args.variant, args.producers, args.consumers, args.seed) + ".json")
    data = run_job(args)
    sx.write_json_atomic(data, path)
    _log(f"JSON yazıldı: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
