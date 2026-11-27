"""
Talep / arz şoku deneyi (demand / supply shock) — tabular Q-Learning, tek
kesintisiz eğitim.

--shock demand (varsayılan): Faz 2'de TÜKETİCİ yük profilleri ×--mult
    (config.demand_multiplier). Dosya adları: shock_p{P}c{C}_seed{S}.json,
    shock_summary.txt, fig_shock_*.png.
--shock supply: Faz 2'de ÜRETİCİ PV profilleri ×--mult
    (config.supply_multiplier); talep çarpanı 1.0 kalır. Dosya adları
    (aynı dizinde talep şoku sonuçlarının üzerine yazmamak için):
    supply_shock_p{P}c{C}_seed{S}.json, supply_shock_summary.txt,
    supply_fig_shock_*.png.
Aşağıdaki açıklama talep şoku içindir; arz şokunda tek fark çarpanın hangi
profile uygulandığıdır. JSON'daki "shock" alanı türü kaydeder (eski
JSON'larda yoksa "demand" kabul edilir).

Üç faz, Q-tabloları fazlar arasında KORUNUR (aynı QLearningEnv nesnesi):
    Faz 1 (denge)      : --p1 episode, tüketici yük çarpanı 1.0
    Faz 2 (talep şoku) : --p2 episode, tüketici yük çarpanı --mult (varsayılan 2.0)
    Faz 3 (toparlanma) : --p3 episode, çarpan 1.0
Her faz değişiminde tüm Q-ajanlarının keşif oranı ε, --eps-reset değerine
(varsayılan 0.5) çekilir; sonra her zamanki gibi azalır.

Eğitim döngüsü QLearningEnv.train() ile BİREBİR aynıdır (ql_environment.py
değiştirilmedi): global RNG'ler başta bir kez seed_everything(seed) ile
sabitlenir (experiment.py ile aynı), episode k'nin gün seed'i
config.seed + k'dir (k = fazlar boyunca sürekli global indeks), her episode
reset → step… → end_episode. Faz 1, aynı seed ile QLearningEnv.train()'in
ilk --p1 episode'u ile birebir aynı sonucu verir.

Kayıt edilen metrikler (episode başına, keşifli EĞİTİM günleri, gün toplamı):
    avg_price    : miktar ağırlıklı ortalama P2P takas fiyatı ($/kWh)
    p2p_kwh      : P2P ticaret hacmi
    demand_kwh   : topluluk açığı  D = Σ_t Σ_i max(net, 0)   (community_balance)
    supply_kwh   : topluluk fazlası S = Σ_t Σ_i max(−net, 0) (community_balance)
    e_max_kwh    : Σ_t min(S_t, D_t)
    unmet_kwh    : P2P ile karşılanmayan talep = şebekeden (ToU) alınan enerji
    grid_sold_kwh: şebekeye (FiT) satılan enerji
    epsilon, multiplier, phase, system_reward

AUSGRID MODU (--profiles ausgrid): eğitim günleri yine config.seed + k'dır; her
biri bir Ausgrid EĞİTİM senaryosudur (eğitim evleri × eğitim günü) ve fazın
çarpanı make_profile_manager üzerinden aynen uygulanır. Yukarıdaki episode
serileri (D, S, E_max dahil) her episode'un KENDİ eğitim senaryosundan hesaplanır
(community_balance(env.profiles, ...)); sabit bir güne bağlı değildir.
Ek olarak (yalnızca Ausgrid): her fazın SONUNDAKİ Q-tabloları kopyalanır ve
eğitim tamamen bittikten sonra her kopya 73 değerlendirme senaryosunun hepsinde
(test evleri × test günleri, fazın çarpanı uygulanmış) greedy (ε=0) olarak
değerlendirilir; toplama src/experiment.py ile aynıdır (_evaluate_ausgrid,
aggregate_scenarios: η = 100·ΣP2P/ΣE_max). Değerlendirme eğitimden SONRA yapıldığı
için eğitim yörüngesini (RNG sırası, Q-tabloları) etkilemez. JSON'da
"heldout_eval" altında, özet metninde ayrı bir blok olarak yazılır. Varsayılan
çıktı dizinleri: shock_results_ausgrid/ (talep), supply_shock_results_ausgrid/ (arz).
Sentetik (varsayılan) yol değişmedi.

Kullanım:
    python -m src.experiment_shock --producers 10 --consumers 10 --seed 1 \\
        --p1 6000 --p2 5000 --p3 5000 --mult 2.0 --eps-reset 0.5 --out shock_results
    python -m src.experiment_shock --shock supply --producers 10 --consumers 10 --seed 1 \\
        --p1 6000 --p2 5000 --p3 5000 --mult 2.0 --eps-reset 0.5 --out supply_shock_results
    python -m src.experiment_shock --plot-only --out shock_results   # yalnızca grafikler
    python -m src.experiment_shock --profiles ausgrid                # → shock_results_ausgrid/
    python -m src.experiment_shock --profiles ausgrid --shock supply # → supply_shock_results_ausgrid/

torch İÇE AKTARILMAZ; matplotlib yalnızca grafik çizilirken yüklenir.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime
from typing import Dict, List

import numpy as np

from src.config import SimConfig
from src.experiment import _log, _silenced, write_json_atomic
from src.random_environment import community_balance
from src.seeding import seed_everything

SERIES_KEYS = [
    "avg_price", "p2p_kwh", "demand_kwh", "supply_kwh", "e_max_kwh",
    "unmet_kwh", "grid_sold_kwh", "epsilon", "multiplier", "phase", "system_reward",
]
SUMMARY_KEYS = [
    "avg_price", "p2p_kwh", "demand_kwh", "supply_kwh", "e_max_kwh",
    "unmet_kwh", "grid_sold_kwh", "system_reward", "epsilon",
]
SUMMARY_TAIL = 500        # her fazın son kaç episode'unun ortalaması
PROGRESS_EVERY = 500
# Enerji dengesi denetimi: D = P2P + şebeke alımı, S = P2P + şebeke satımı
BALANCE_TOL = 1e-6


SHOCK_TYPES = ("demand", "supply")
# Faz 2'de hangi SimConfig alanının çarpan olarak kullanıldığı
SHOCK_FIELD = {"demand": "demand_multiplier", "supply": "supply_multiplier"}


def _prefix(shock: str) -> str:
    """Dosya adı öneki: talep şoku için "" (eski adlar aynen), arz için "supply_"."""
    return "" if shock == "demand" else f"{shock}_"


def _json_name(p: int, c: int, seed: int, shock: str = "demand") -> str:
    return f"{_prefix(shock)}shock_p{p}c{c}_seed{seed}.json"


def _shock_of(data: Dict) -> str:
    """JSON'daki şok türü; eski (alan içermeyen) JSON'lar talep şokudur."""
    return data.get("shock", "demand")


# ---------------------------------------------------------------------------
# Eğitim
# ---------------------------------------------------------------------------

def run_shock(args) -> Dict:
    from src.ql_environment import QLearningEnv
    from src.ql_agent import QLearningAgent

    profiles = getattr(args, "profiles", "synthetic")
    config = SimConfig(n_producers=args.producers, n_consumers=args.consumers, seed=args.seed,
                       profile_source=profiles)
    ausgrid = config.profile_source == "ausgrid"
    phases = [
        {"phase": 1, "name": "equilibrium", "episodes": args.p1, "multiplier": 1.0},
        {"phase": 2, "name": f"{args.shock} shock", "episodes": args.p2,
         "multiplier": float(args.mult)},
        {"phase": 3, "name": "recovery", "episodes": args.p3, "multiplier": 1.0},
    ]
    n_total = sum(ph["episodes"] for ph in phases)
    start = 0
    for ph in phases:
        ph["start"], ph["end"] = start, start + ph["episodes"]   # [start, end)
        start = ph["end"]
    if ausgrid:
        from src.ausgrid_profiles import assert_train_seeds_ok
        assert_train_seeds_ok(config.seed, n_total)   # eğitim seed'leri test aralığına girmez

    # experiment.py ile aynı: global RNG'ler (ε-greedy) yöntemden önce bir kez.
    seed_everything(config.seed)
    # experiment.run_ql ile aynı: QLearningEnv(config) varsayılan hiperparametreleri.
    env = QLearningEnv(config)
    hyper = {
        "alpha_lr": env.ql_alpha_lr, "gamma": env.ql_gamma, "epsilon": env.ql_epsilon,
        "epsilon_min": env.ql_epsilon_min, "epsilon_decay": env.ql_epsilon_decay,
        "epsilon_reset": float(args.eps_reset),
    }

    series: Dict[str, List[float]] = {k: [] for k in SERIES_KEYS}
    max_dev_demand = 0.0
    max_dev_supply = 0.0
    n_balance_viol = 0

    _log(f"{'Talep' if args.shock == 'demand' else 'Arz'} şoku başlıyor: {config.n_producers}+{config.n_consumers} | seed={config.seed} "
         f"| fazlar {args.p1}/{args.p2}/{args.p3} | çarpan ×{args.mult} | ε reset {args.eps_reset}")
    t0, c0 = time.perf_counter(), time.process_time()
    phase_seconds = []
    # Ausgrid: her fazın sonundaki Q-tablolarının kopyaları (eğitim bittikten sonra
    # 73 test senaryosunda değerlendirilir). Kopyalamak RNG tüketmez.
    phase_tables: List[Dict] = []
    ep = 0
    for ph in phases:
        t_ph = time.perf_counter()
        # env.config ile aynı nesne. Şoklanmayan çarpan 1.0'da kalır.
        config.demand_multiplier = 1.0
        config.supply_multiplier = 1.0
        setattr(config, SHOCK_FIELD[args.shock], ph["multiplier"])
        for k in range(ph["episodes"]):
            with _silenced():
                env.reset(seed=config.seed + ep)      # QLearningEnv.train ile aynı kural
                if k == 0 and ph["phase"] > 1:
                    # Faz değişimi: ε'u sıfırla (reset RNG/ε kullanmaz; ilk aksiyon
                    # seçimi step()'te olduğundan sıra train() ile aynı kalır).
                    for a in env._ql_agents.values():
                        a.epsilon = float(args.eps_reset)
                eps_used = float(env.agents[0].epsilon) if env.agents else 0.0
                bal = community_balance(env.profiles, config)   # RNG tüketmez
                done = False
                while not done:
                    _, _, done, _ = env.step()
                env.end_episode()

            m = env.metrics
            p2p = float(m.total_internal_trading())
            grid_buy = float(m.total_grid_purchased())
            grid_sell = float(m.total_grid_sold())
            dev_d = abs(bal["deficit_kwh"] - (p2p + grid_buy))
            dev_s = abs(bal["surplus_kwh"] - (p2p + grid_sell))
            max_dev_demand = max(max_dev_demand, dev_d)
            max_dev_supply = max(max_dev_supply, dev_s)
            if dev_d > BALANCE_TOL * max(1.0, bal["deficit_kwh"]) or \
               dev_s > BALANCE_TOL * max(1.0, bal["surplus_kwh"]):
                n_balance_viol += 1

            series["avg_price"].append(float(m.average_clearing_price()))
            series["p2p_kwh"].append(p2p)
            series["demand_kwh"].append(float(bal["deficit_kwh"]))
            series["supply_kwh"].append(float(bal["surplus_kwh"]))
            series["e_max_kwh"].append(float(bal["e_max_kwh"]))
            series["unmet_kwh"].append(grid_buy)
            series["grid_sold_kwh"].append(grid_sell)
            series["epsilon"].append(eps_used)
            series["multiplier"].append(float(ph["multiplier"]))
            series["phase"].append(ph["phase"])
            series["system_reward"].append(float(sum(
                a.episode_rewards[-1] for a in env.agents if isinstance(a, QLearningAgent)
            )))

            ep += 1
            if ep % PROGRESS_EVERY == 0 or ep == n_total:
                el = time.perf_counter() - t0
                eta = el / ep * (n_total - ep)
                w = min(ep, PROGRESS_EVERY)
                _log(f"  episode {ep:6d}/{n_total} faz {ph['phase']} | ε={eps_used:.3f} "
                     f"| son {w}: fiyat {np.mean(series['avg_price'][-w:]):.4f} "
                     f"P2P {np.mean(series['p2p_kwh'][-w:]):7.1f} "
                     f"D {np.mean(series['demand_kwh'][-w:]):7.1f} "
                     f"şebeke {np.mean(series['unmet_kwh'][-w:]):7.1f} kWh "
                     f"| {el:8.1f}s geçti | kalan ~{eta:8.1f}s")
        phase_seconds.append(time.perf_counter() - t_ph)
        if ausgrid:
            phase_tables.append({aid: (a.q_table.copy(), a.buy_q_table.copy())
                                 for aid, a in env._ql_agents.items()})
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    config.demand_multiplier = 1.0
    config.supply_multiplier = 1.0

    if n_balance_viol:
        _log(f"UYARI: {n_balance_viol} episode'da enerji dengesi tutmadı "
             f"(maks |D−P2P−şebeke| = {max_dev_demand:.3e}, maks |S−P2P−FiT| = {max_dev_supply:.3e})")

    extra: Dict = {}
    if ausgrid:
        from src.ausgrid_profiles import split_identity
        t_ev = time.perf_counter()
        heldout, scen_info = evaluate_phases_ausgrid(env, config, phases, phase_tables, args.shock)
        extra = {
            "profile_source": config.profile_source,
            "ausgrid": split_identity(),
            "heldout_eval": heldout,
            "eval_scenarios": scen_info,
        }
        _log(f"73 senaryolu değerlendirme bitti ({time.perf_counter() - t_ev:.1f}s)")

    return {
        "format": "energy-trading-shock/1",
        "shock": args.shock,
        "population": {"producers": config.n_producers, "consumers": config.n_consumers,
                       "label": f"{config.n_producers}+{config.n_consumers}"},
        "seed": config.seed,
        "phases": phases,
        "phase_boundaries": [ph["end"] for ph in phases[:-1]],
        "n_episodes": n_total,
        "ql_hyperparameters": hyper,
        "tariffs": {"fit_price": config.fit_price, "tou_price": config.tou_price},
        "sim_config": dataclasses.asdict(config),
        "series": series,
        "summary": summarize(series, phases),
        "balance_check": {
            "max_abs_demand_minus_p2p_minus_grid_kwh": max_dev_demand,
            "max_abs_supply_minus_p2p_minus_gridsold_kwh": max_dev_supply,
            "n_violations": n_balance_viol, "rel_tol": BALANCE_TOL,
        },
        "timings": {"total_wall_seconds": wall, "cpu_seconds": cpu,
                    "phase_wall_seconds": phase_seconds},
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "hostname": socket.gethostname(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv[1:],
        **extra,
    }


# Ausgrid test değerlendirmesinde özet satırları (aggregate_scenarios anahtarları)
HELDOUT_ROWS = [
    ("deficit_kwh", "Community demand D (kWh, sum)", "{:.2f}"),
    ("surplus_kwh", "Community supply S (kWh, sum)", "{:.2f}"),
    ("e_max_kwh", "Feasible maximum E_max (kWh, sum)", "{:.2f}"),
    ("p2p_kwh", "P2P volume (kWh, sum)", "{:.2f}"),
    ("efficiency_pct", "Efficiency eta = P2P/E_max (%)", "{:.2f}"),
    ("avg_price", "Avg. clearing price ($/kWh)", "{:.4f}"),
    ("grid_purchased_kwh", "Unmet by P2P = grid buy (kWh, sum)", "{:.2f}"),
    ("grid_sold_kwh", "Grid sale at FiT (kWh, sum)", "{:.2f}"),
    ("buyer_cost", "Buyer cost ($, sum)", "{:.2f}"),
    ("seller_revenue", "Seller revenue ($, sum)", "{:.2f}"),
]


def evaluate_phases_ausgrid(env, config: SimConfig, phases: List[Dict],
                            phase_tables: List[Dict], shock: str):
    """Ausgrid: her fazın sonundaki Q-tablolarını 73 test senaryosunda değerlendirir.

    Faz i'nin kopyası, fazın çarpanı config'e konarak (öbür çarpan 1.0)
    src.experiment.ausgrid_eval_community + _evaluate_ausgrid ile değerlendirilir
    (topluluk dengesi / E_max da aynı çarpanla, senaryo başına). Eğitim BİTTİKTEN
    sonra çağrılır; sonunda son Q-tabloları ve çarpanlar (1.0) geri konur.
    """
    from src.experiment import ausgrid_eval_community, _evaluate_ausgrid, _per_scenario_block
    final = {aid: (a.q_table, a.buy_q_table) for aid, a in env._ql_agents.items()}
    out: List[Dict] = []
    scen_info: Dict = {}
    try:
        for ph, tables in zip(phases, phase_tables):
            if ph["end"] <= ph["start"]:
                continue                       # boş faz: önceki fazla aynı tablolar
            for aid, (qs, qb) in tables.items():
                env._ql_agents[aid].q_table = qs
                env._ql_agents[aid].buy_q_table = qb
            config.demand_multiplier = 1.0
            config.supply_multiplier = 1.0
            setattr(config, SHOCK_FIELD[shock], ph["multiplier"])
            scens = ausgrid_eval_community(config)       # çarpanlı S, D, E_max (senaryo başına)
            with _silenced():
                metrics, per = _evaluate_ausgrid(env, config, scens)
            community = {k: float(sum(sc[k] for sc in scens))
                         for k in ("surplus_kwh", "deficit_kwh", "e_max_kwh")}
            if abs(community["e_max_kwh"] - metrics["e_max_kwh"]) > 1e-9 * max(1.0, community["e_max_kwh"]):
                raise RuntimeError("E_max uyuşmazlığı (değerlendirme)")
            per_block = _per_scenario_block(scens, per)
            per_block["surplus_kwh"] = [float(sc["surplus_kwh"]) for sc in scens]
            per_block["deficit_kwh"] = [float(sc["deficit_kwh"]) for sc in scens]
            out.append({"phase": ph["phase"], "multiplier": float(ph["multiplier"]),
                        "after_episode": ph["end"], "n_scenarios": len(scens),
                        "community": community, "metrics": metrics, "per_scenario": per_block})
            _log(f"  test (73 senaryo) faz {ph['phase']} sonu, ×{ph['multiplier']:g}: "
                 f"η={metrics['efficiency_pct']:.2f}% fiyat {metrics['avg_price']:.4f} "
                 f"P2P {metrics['p2p_kwh']:.1f} / E_max {metrics['e_max_kwh']:.1f} kWh")
            if not scen_info:
                scen_info = {"eval_seed": [sc["seed"] for sc in scens],
                             "day_index": [sc["day"] for sc in scens],
                             "date": [sc["date"] for sc in scens],
                             "producers": [sc["producers"] for sc in scens],
                             "consumers": [sc["consumers"] for sc in scens]}
    finally:
        for aid, (qs, qb) in final.items():
            env._ql_agents[aid].q_table = qs
            env._ql_agents[aid].buy_q_table = qb
        config.demand_multiplier = 1.0
        config.supply_multiplier = 1.0
    return {
        "note": ("Q-tables frozen at the end of each phase, evaluated greedily (epsilon=0, no "
                 "learning) on all 73 Ausgrid test scenarios with that phase's multiplier; "
                 "metrics summed over scenarios, eta = 100*sum(P2P)/sum(E_max)."),
        "phases": out,
    }, scen_info


def summarize(series: Dict[str, List[float]], phases: List[Dict]) -> Dict:
    """Her fazın son SUMMARY_TAIL episode'u üzerinden ortalamalar."""
    out = {}
    for ph in phases:
        a, b = ph["start"], ph["end"]
        if b <= a:
            continue
        lo = max(a, b - SUMMARY_TAIL)
        out[f"phase{ph['phase']}"] = {
            "episodes": [lo + 1, b],   # 1 tabanlı, kapalı aralık
            "n": b - lo,
            **{k: float(np.mean(series[k][lo:b])) for k in SUMMARY_KEYS},
        }
    return out


def format_summary(data: Dict) -> str:
    phases = data["phases"]
    s = data["summary"]
    cols = [f"phase{ph['phase']}" for ph in phases if f"phase{ph['phase']}" in s]
    rows = [
        ("avg_price", "Avg. clearing price ($/kWh)", "{:.4f}"),
        ("demand_kwh", "Community demand D (kWh/day)", "{:.2f}"),
        ("supply_kwh", "Community supply S (kWh/day)", "{:.2f}"),
        ("e_max_kwh", "Feasible maximum E_max (kWh/day)", "{:.2f}"),
        ("p2p_kwh", "P2P volume (kWh/day)", "{:.2f}"),
        ("unmet_kwh", "Unmet by P2P = grid buy (kWh/day)", "{:.2f}"),
        ("grid_sold_kwh", "Grid sale at FiT (kWh/day)", "{:.2f}"),
        ("system_reward", "System reward", "{:.2f}"),
        ("epsilon", "Epsilon", "{:.3f}"),
    ]
    hp = data["ql_hyperparameters"]
    width = 36 + 16 * len(cols)
    lines = [
        "=" * width,
        f"{_shock_of(data).upper()} SHOCK — Q-Learning {data['population']['label']}, seed {data['seed']}",
        "  phases: " + " | ".join(
            f"{ph['phase']}: {ph['name']} x{ph['multiplier']:g}, episodes {ph['start'] + 1}-{ph['end']}"
            for ph in phases),
        f"  QL: alpha={hp['alpha_lr']} gamma={hp['gamma']} eps={hp['epsilon']}->{hp['epsilon_min']} "
        f"decay={hp['epsilon_decay']} eps_reset={hp['epsilon_reset']}",
        f"  values: mean of the last {SUMMARY_TAIL} training episodes of each phase "
        f"(exploration on)",
        "-" * width,
        f"  {'Metric':<34}" + "".join(
            f"{'P' + c[-1] + ' ' + str(s[c]['episodes'][0]) + '-' + str(s[c]['episodes'][1]):>16}"
            for c in cols),
        "-" * width,
    ]
    for key, label, fmt in rows:
        lines.append(f"  {label:<34}" + "".join(f"{fmt.format(s[c][key]):>16}" for c in cols))
    bc = data["balance_check"]
    lines += [
        "-" * width,
        f"  energy balance check: max |D - P2P - grid buy| = {bc['max_abs_demand_minus_p2p_minus_grid_kwh']:.2e} kWh, "
        f"violations = {bc['n_violations']}",
        "=" * width,
    ]
    if data.get("profile_source", "synthetic") == "ausgrid":
        # Başlığa profil kaynağı; eğitim satırları eğitim senaryolarından gelir.
        lines.insert(2, "  profiles: Ausgrid real data; training episodes = Ausgrid TRAIN scenarios "
                        f"(split {data.get('ausgrid', {}).get('split_id', '?')}); D, S, E_max per "
                        "episode from that episode's own scenario")
        lines += _format_heldout(data, width)
    return "\n".join(lines)


def _format_heldout(data: Dict, width: int) -> List[str]:
    """Ausgrid: faz sonu Q-tablolarının 73 test senaryosundaki değerlendirmesi (ayrı blok)."""
    he = data.get("heldout_eval", {}).get("phases", [])
    if not he:
        return []
    n = he[0]["n_scenarios"]
    out = [
        f"HELD-OUT EVALUATION — Q-tables frozen at the end of each phase, greedy (eps=0), "
        f"{n} Ausgrid test scenarios",
        f"  each column: that phase's multiplier applied to the test scenarios; energies / $ summed "
        f"over the {n} scenarios, eta = 100*sum(P2P)/sum(E_max), price volume-weighted",
        "-" * width,
        f"  {'Metric':<34}" + "".join(f"{'P' + str(e['phase']) + ' end x' + format(e['multiplier'], 'g'):>16}"
                                      for e in he),
        "-" * width,
    ]
    for key, label, fmt in HELDOUT_ROWS:
        vals = [e["community"][key] if key in ("deficit_kwh", "surplus_kwh") else e["metrics"][key]
                for e in he]
        out.append(f"  {label:<34}" + "".join(f"{fmt.format(v):>16}" for v in vals))
    out.append("=" * width)
    return out


# ---------------------------------------------------------------------------
# Grafikler
# ---------------------------------------------------------------------------

# Okabe–Ito (renk körlüğüne dayanıklı)
C_BLUE, C_VERM, C_GREEN, C_GREY = "#0072B2", "#D55E00", "#009E73", "#555555"
PHASE_FILL = ["#9ecae1", "#fdae6b", "#a1d99b"]


def _moving_average(y: np.ndarray, phases: List[Dict], w: int) -> np.ndarray:
    """Faz içinde geriye dönük (trailing) hareketli ortalama.

    Faz sınırını aşmaz (şok anı ortalamayla bulanıklaşmaz); fazın ilk w−1
    episode'unda pencere o ana kadarki episode'larla sınırlıdır.
    """
    out = np.empty_like(y, dtype=float)
    for ph in phases:
        a, b = ph["start"], ph["end"]
        if b <= a:
            continue
        seg = y[a:b].astype(float)
        cs = np.concatenate([[0.0], np.cumsum(seg)])
        idx = np.arange(1, len(seg) + 1)
        lo = np.maximum(0, idx - w)
        out[a:b] = (cs[idx] - cs[lo]) / (idx - lo)
    return out


def plot_all(data: Dict, out_dir: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.legend import Legend

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
    })

    phases = data["phases"]
    S = {k: np.asarray(v, dtype=float) for k, v in data["series"].items()}
    n = len(S["p2p_kwh"])
    x = np.arange(1, n + 1)
    w = max(50, n // 80)
    ma = lambda key: _moving_average(S[key], phases, w)

    shock = _shock_of(data)
    pre = _prefix(shock)
    phase_names = {1: "equilibrium", 2: f"{shock} shock", 3: "recovery"}

    def new_ax():
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        for i, ph in enumerate(phases):
            if ph["end"] > ph["start"]:
                ax.axvspan(ph["start"] + 0.5, ph["end"] + 0.5, color=PHASE_FILL[i % 3],
                           alpha=0.18, lw=0, zorder=0)
        for b in data["phase_boundaries"]:
            if 0 < b < n:
                ax.axvline(b + 0.5, color=C_GREY, ls="--", lw=0.8, zorder=1)
        ax.grid(True, color="#d9d9d9", lw=0.4, alpha=0.7)
        ax.set_axisbelow(True)
        ax.set_xlim(0.5, n + 0.5)
        ax.set_xlabel("Training episode")
        return fig, ax

    def finish(fig, ax, series_handles, fname):
        phase_handles = [
            Patch(facecolor=PHASE_FILL[i % 3], alpha=0.45, edgecolor="none",
                  label=f"Phase {ph['phase']}: {phase_names.get(ph['phase'], '')} "
                        f"(×{ph['multiplier']:.1f})")
            for i, ph in enumerate(phases)
        ]
        # Faz göstergesi: ayrı bir Legend sanatçısı (ax.legend ikinci çağrıda
        # ilkinin yerine geçtiği için) — eksenlerin üstünde, 1. satır.
        leg_ph = Legend(ax, phase_handles, [h.get_label() for h in phase_handles],
                        loc="lower center", bbox_to_anchor=(0.5, 1.075), ncol=3,
                        frameon=False, handlelength=1.4, columnspacing=1.2,
                        borderaxespad=0.0)
        ax.add_artist(leg_ph)
        leg_ph.set_clip_on(False)   # add_artist eksen alanına kırpar; dışarıda kalsın
        # Seri göstergesi: eksenlerin üstünde, 2. satır (veriyi örtmez).
        ax.legend(handles=series_handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
                  ncol=len(series_handles), frameon=False, handlelength=2.0,
                  columnspacing=1.4, borderaxespad=0.0)
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        return path

    raw_kw = dict(lw=0.35, alpha=0.30, zorder=2)
    ma_kw = dict(lw=1.6, zorder=3)
    paths = []

    # 1) Ortalama takas fiyatı
    fig, ax = new_ax()
    ax.plot(x, S["avg_price"], color=C_BLUE, **raw_kw)
    ax.plot(x, ma("avg_price"), color=C_BLUE, **ma_kw)
    ax.set_ylabel("Average clearing price ($/kWh)")
    paths.append(finish(fig, ax, [
        Line2D([], [], color=C_BLUE, lw=0.8, alpha=0.45, label="per episode"),
        Line2D([], [], color=C_BLUE, lw=1.6, label="moving average"),
    ], pre + "fig_shock_price.png"))

    # 2) Topluluk talebi D ve arzı S
    fig, ax = new_ax()
    ax.plot(x, S["demand_kwh"], color=C_VERM, **raw_kw)
    ax.plot(x, S["supply_kwh"], color=C_GREEN, **raw_kw)
    ax.plot(x, ma("demand_kwh"), color=C_VERM, **ma_kw)
    ax.plot(x, ma("supply_kwh"), color=C_GREEN, **ma_kw)
    ax.set_ylabel("Energy (kWh/day)")
    ax.set_ylim(bottom=0)
    paths.append(finish(fig, ax, [
        Line2D([], [], color=C_VERM, lw=1.6, label="Demand $D$ (moving average)"),
        Line2D([], [], color=C_GREEN, lw=1.6, label="Supply $S$ (moving average)"),
        Line2D([], [], color=C_GREY, lw=0.8, alpha=0.45, label="per episode"),
    ], pre + "fig_shock_demand_supply.png"))

    # 3) P2P ticaret hacmi + E_max
    fig, ax = new_ax()
    ax.plot(x, S["p2p_kwh"], color=C_BLUE, **raw_kw)
    ax.plot(x, ma("p2p_kwh"), color=C_BLUE, **ma_kw)
    ax.plot(x, ma("e_max_kwh"), color="black", lw=0.8, ls=(0, (4, 2)), zorder=3)
    ax.set_ylabel("P2P trading volume (kWh/day)")
    ax.set_ylim(bottom=0)
    paths.append(finish(fig, ax, [
        Line2D([], [], color=C_BLUE, lw=0.8, alpha=0.45, label="per episode"),
        Line2D([], [], color=C_BLUE, lw=1.6, label="moving average"),
        Line2D([], [], color="black", lw=0.8, ls=(0, (4, 2)),
               label="Feasible maximum $E_{\\max}$"),
    ], pre + "fig_shock_p2p_volume.png"))

    # 4) P2P ile karşılanmayan talep (şebekeden alınan)
    fig, ax = new_ax()
    ax.plot(x, S["unmet_kwh"], color=C_VERM, **raw_kw)
    ax.plot(x, ma("unmet_kwh"), color=C_VERM, **ma_kw)
    ax.set_ylabel("Demand bought from grid (kWh/day)")
    # y ekseni 0'dan başlatılmaz: arz şokunda değişim (~%15) 0 tabanında ezilir.
    paths.append(finish(fig, ax, [
        Line2D([], [], color=C_VERM, lw=0.8, alpha=0.45, label="per episode"),
        Line2D([], [], color=C_VERM, lw=1.6, label="moving average"),
    ], pre + "fig_shock_unmet_demand.png"))

    # 5) Tek sütunluk birleşik şekil: (a) D,S  (b) fiyat  (c) P2P + E_max  (d) şebeke alımı
    paths.append(plot_panels(data, out_dir, pre + "fig_shock_panels.png",
                             S, x, n, ma, phases, raw_kw))

    return paths


def plot_panels(data, out_dir, fname, S, x, n, ma, phases, raw_kw) -> str:
    """Dört paneli ortak x ekseninde alt alta dizen tek sütun (≈4.6×6.6 in) şekli.

    plot_all() içinden, aynı rcParams ve aynı hareketli ortalama ile çağrılır.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    tick_fs, lab_fs, leg_fs, tag_fs = 7.0, 7.5, 6.5, 7.5
    ma_kw = dict(lw=1.2, zorder=3)
    fig, axes = plt.subplots(4, 1, figsize=(4.6, 6.6), sharex=True)
    fig.subplots_adjust(left=0.135, right=0.955, top=0.952, bottom=0.068, hspace=0.13)

    for ax in axes:
        for i, ph in enumerate(phases):
            if ph["end"] > ph["start"]:
                ax.axvspan(ph["start"] + 0.5, ph["end"] + 0.5, color=PHASE_FILL[i % 3],
                           alpha=0.18, lw=0, zorder=0)
        for b in data["phase_boundaries"]:
            if 0 < b < n:
                ax.axvline(b + 0.5, color=C_GREY, ls="--", lw=0.7, zorder=1)
        ax.grid(True, color="#d9d9d9", lw=0.35, alpha=0.7)
        ax.set_axisbelow(True)
        ax.set_xlim(0.5, n + 0.5)
        ax.tick_params(labelsize=tick_fs, length=2.5, width=0.6, pad=2)

    def headroom(ax, frac, bottom=None):
        """Üstte etiket / gösterge için boşluk bırak (veriyi örtmesinler)."""
        lo, hi = ax.get_ylim()
        if bottom is not None:
            lo = bottom
        ax.set_ylim(lo, hi + frac * (hi - lo))

    ax_a, ax_b, ax_c, ax_d = axes
    line_leg = dict(loc="upper right", frameon=False, fontsize=leg_fs, ncol=2,
                    handlelength=1.6, columnspacing=1.0, borderaxespad=0.25,
                    handletextpad=0.5)

    # (a) Talep D ve arz S
    ax_a.plot(x, S["demand_kwh"], color=C_VERM, **raw_kw)
    ax_a.plot(x, S["supply_kwh"], color=C_GREEN, **raw_kw)
    ax_a.plot(x, ma("demand_kwh"), color=C_VERM, **ma_kw)
    ax_a.plot(x, ma("supply_kwh"), color=C_GREEN, **ma_kw)
    ax_a.set_ylabel("$D$, $S$ (kWh/day)", fontsize=lab_fs)
    headroom(ax_a, 0.22, bottom=0)
    ax_a.legend(handles=[Line2D([], [], color=C_VERM, lw=1.2, label="$D$"),
                         Line2D([], [], color=C_GREEN, lw=1.2, label="$S$")], **line_leg)

    # (b) Ortalama takas fiyatı
    ax_b.plot(x, S["avg_price"], color=C_BLUE, **raw_kw)
    ax_b.plot(x, ma("avg_price"), color=C_BLUE, **ma_kw)
    ax_b.set_ylabel("Price ($/kWh)", fontsize=lab_fs)
    headroom(ax_b, 0.12)

    # (c) P2P hacmi + E_max
    ax_c.plot(x, S["p2p_kwh"], color=C_BLUE, **raw_kw)
    ax_c.plot(x, ma("p2p_kwh"), color=C_BLUE, **ma_kw)
    ax_c.plot(x, ma("e_max_kwh"), color="black", lw=0.7, ls=(0, (4, 2)), zorder=3)
    ax_c.set_ylabel("P2P (kWh/day)", fontsize=lab_fs)
    headroom(ax_c, 0.22, bottom=0)
    ax_c.legend(handles=[Line2D([], [], color=C_BLUE, lw=1.2, label="P2P"),
                         Line2D([], [], color="black", lw=0.7, ls=(0, (4, 2)),
                                label="$E_{\\max}$")], **line_leg)

    # (d) Şebekeden alınan (P2P ile karşılanmayan talep)
    ax_d.plot(x, S["unmet_kwh"], color=C_VERM, **raw_kw)
    ax_d.plot(x, ma("unmet_kwh"), color=C_VERM, **ma_kw)
    ax_d.set_ylabel("Grid buy (kWh/day)", fontsize=lab_fs)
    headroom(ax_d, 0.12)
    ax_d.set_xlabel("Training episode", fontsize=lab_fs)

    for ax, tag in zip(axes, "abcd"):
        ax.text(0.012, 0.965, f"({tag})", transform=ax.transAxes, ha="left", va="top",
                fontsize=tag_fs, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.8))
    fig.align_ylabels(axes)

    # Ortak faz göstergesi (kısa etiketler: 4.6 in genişliğe okunur boyutta sığsın)
    short = {1: "Phase 1:", 2: "Phase 2: shock", 3: "Phase 3:"}
    handles = [Patch(facecolor=PHASE_FILL[i % 3], alpha=0.45, edgecolor="none",
                     label=f"{short.get(ph['phase'], 'Phase ' + str(ph['phase']))} "
                           f"×{ph['multiplier']:.1f}")
               for i, ph in enumerate(phases)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=leg_fs, bbox_to_anchor=(0.545, 0.958), handlelength=1.4,
               columnspacing=1.6, borderaxespad=0.0)

    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Talep şoku deneyi (Q-Learning, 3 faz)")
    p.add_argument("--producers", type=int, default=10)
    p.add_argument("--consumers", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--p1", type=int, default=6000, help="Faz 1 (denge) episode sayısı")
    p.add_argument("--p2", type=int, default=5000, help="Faz 2 (şok) episode sayısı")
    p.add_argument("--p3", type=int, default=5000, help="Faz 3 (toparlanma) episode sayısı")
    p.add_argument("--shock", choices=SHOCK_TYPES, default="demand",
                   help="Şok türü: demand = tüketici yükü ×mult, supply = üretici PV ×mult")
    p.add_argument("--mult", type=float, default=2.0,
                   help="Faz 2 çarpanı (demand: tüketici yükü, supply: üretici PV)")
    p.add_argument("--eps-reset", dest="eps_reset", type=float, default=0.5,
                   help="Faz değişiminde ε'un çekileceği değer")
    p.add_argument("--profiles", choices=["synthetic", "ausgrid"], default="synthetic",
                   help="Profil kaynağı (varsayılan: synthetic = eski davranış)")
    p.add_argument("--out", type=str, default=None,
                   help="Çıktı dizini (varsayılan: shock_results; --profiles ausgrid ile "
                        "shock_results_ausgrid / arz şokunda supply_shock_results_ausgrid)")
    p.add_argument("--plot-only", dest="plot_only", action="store_true",
                   help="Eğitme; kayıtlı JSON'dan grafikleri yeniden üret")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="Eğitme; planı, çıktı yollarını (Ausgrid'de 73 test senaryosunun "
                        "çarpanlı S / D / E_max toplamlarını) yaz ve çık")
    return p



def dry_run(args, json_path: str) -> None:
    """Plan özeti; ajan / eğitim yok (yalnızca profil tabanlı denge hesapları)."""
    n_total = args.p1 + args.p2 + args.p3
    print(f"{args.shock} şoku | {args.producers}+{args.consumers} | seed={args.seed} | "
          f"profiller={args.profiles} | fazlar {args.p1}/{args.p2}/{args.p3} "
          f"(toplam {n_total} episode) | çarpan ×{args.mult} | ε reset {args.eps_reset}")
    print(f"  JSON  : {json_path}")
    print(f"  özet  : {os.path.join(args.out, _prefix(args.shock) + 'shock_summary.txt')}")
    print(f"  JSON {'VAR (üzerine yazılır)' if os.path.exists(json_path) else 'yok'}")
    if args.profiles == "ausgrid":
        from src.ausgrid_profiles import assert_train_seeds_ok, split_identity
        from src.experiment import ausgrid_eval_community
        assert_train_seeds_ok(args.seed, n_total)
        print(f"  Ausgrid split_id={split_identity()['split_id']} | eğitim seed'leri "
              f"{args.seed}..{args.seed + n_total - 1}")
        for mult in sorted({1.0, float(args.mult)}):
            cfg = SimConfig(n_producers=args.producers, n_consumers=args.consumers, seed=args.seed,
                            profile_source="ausgrid")
            setattr(cfg, SHOCK_FIELD[args.shock], mult)
            scens = ausgrid_eval_community(cfg)
            tot = {k: sum(sc[k] for sc in scens) for k in ("surplus_kwh", "deficit_kwh", "e_max_kwh")}
            print(f"  test ({len(scens)} senaryo) ×{mult:g}: S={tot['surplus_kwh']:.2f} "
                  f"D={tot['deficit_kwh']:.2f} E_max={tot['e_max_kwh']:.2f} kWh")


def default_out(profiles: str, shock: str) -> str:
    """Varsayılan çıktı dizini. Sentetik: eski "shock_results" (iki şok türü için de)."""
    if profiles == "ausgrid":
        return "shock_results_ausgrid" if shock == "demand" else f"{shock}_shock_results_ausgrid"
    return "shock_results"


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.out is None:
        args.out = default_out(args.profiles, args.shock)
    json_path = os.path.join(args.out, _json_name(args.producers, args.consumers, args.seed,
                                                      args.shock))
    if args.dry_run:
        dry_run(args, json_path)
        return
    os.makedirs(args.out, exist_ok=True)

    if args.plot_only:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        _log(f"JSON okundu: {json_path}")
        src_ = data.get("profile_source", "synthetic")
        if src_ != args.profiles:
            _log(f"UYARI: JSON profil kaynağı '{src_}', --profiles '{args.profiles}' "
                 f"(özet / grafikler JSON'a göre üretilir)")
    else:
        data = run_shock(args)
        write_json_atomic(data, json_path)
        _log(f"JSON yazıldı: {json_path}")

    table = format_summary(data)
    with open(os.path.join(args.out, _prefix(_shock_of(data)) + "shock_summary.txt"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    for path in plot_all(data, args.out):
        _log(f"grafik: {path}")
    print(table)


if __name__ == "__main__":
    main()
