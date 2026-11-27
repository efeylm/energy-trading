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

AUSGRID MODU (--profiles ausgrid): profiller gerçek Ausgrid verisinden gelir
(src/ausgrid_profiles.py). Eğitim günleri yine seed + episode'dur (her biri bir
eğitim senaryosu: eğitim evleri × eğitim günü). Değerlendirme TEK gün yerine 73
sabit değerlendirme senaryosunun (test evleri × test günleri) HEPSİ üzerinde
yapılır; liste deney seed'inden bağımsızdır, her yöntem ve seed aynı senaryolarda
değerlendirilir. Toplama: enerji / maliyet / gelir senaryolar üzerinden toplanır,
efficiency_pct = 100·ΣP2P/ΣE_max, avg_price = hacim ağırlıklı (ΣP2P değeri / ΣP2P),
p2p_ratio_pct = 100·ΣP2P/(ΣP2P + Σşebeke alım + Σşebeke satım). Senaryo başına
diziler results.<yöntem>.per_scenario altında, senaryo listesi eval_scenarios
altında saklanır. Random baseline: her senaryoda --baseline_draws fiyat çekilişi
(Ausgrid varsayılanı 20); çekiliş j'nin 73 senaryodaki toplamı bir "çekiliş"
sayılır, metrikler bu çekilişlerin ortalaması / std / min / max'ıdır.

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
from src.profiles import make_profile_manager
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
# Ausgrid: çok senaryolu değerlendirme yardımcıları
# ---------------------------------------------------------------------------

# Senaryolar üzerinden TOPLANAN (doğrusal) metrikler; geri kalan üçü oranlardır.
SUM_KEYS = ["p2p_kwh", "grid_purchased_kwh", "grid_sold_kwh",
            "seller_p2p_income", "seller_grid_income", "seller_revenue",
            "buyer_p2p_cost", "buyer_grid_cost", "buyer_cost", "e_max_kwh"]
RATIO_KEYS = ["p2p_ratio_pct", "avg_price", "efficiency_pct"]
assert set(SUM_KEYS) | set(RATIO_KEYS) == set(METRIC_KEYS)


def aggregate_scenarios(per: List[Dict[str, float]]) -> Dict[str, float]:
    """Senaryo başına metrik sözlüklerini tek bir metrik sözlüğüne toplar.

    Doğrusal metrikler toplanır; oranlar toplamlardan yeniden hesaplanır (sıfıra
    bölme yok): η = 100·ΣP2P/ΣE_max, avg_price = Σ(fiyat_k·P2P_k)/ΣP2P (tek günlük
    MetricsCollector.average_clearing_price ile aynı hacim ağırlıklı tanım),
    p2p_ratio = 100·ΣP2P/(ΣP2P + Σşebeke).
    """
    out = {k: float(sum(d[k] for d in per)) for k in SUM_KEYS}
    p2p = out["p2p_kwh"]
    flow = p2p + out["grid_purchased_kwh"] + out["grid_sold_kwh"]
    value = float(sum(d["avg_price"] * d["p2p_kwh"] for d in per))
    out["p2p_ratio_pct"] = p2p / flow * 100 if flow > 0 else 0.0
    out["avg_price"] = value / p2p if p2p > 0 else 0.0
    out["efficiency_pct"] = p2p / out["e_max_kwh"] * 100 if out["e_max_kwh"] > 0 else 0.0
    return {k: out[k] for k in METRIC_KEYS}


def ausgrid_eval_community(config: SimConfig) -> List[Dict]:
    """73 değerlendirme senaryosu + her birinin profil tabanlı dengesi (S, D, E_max)."""
    from src.ausgrid_profiles import eval_scenarios
    out = []
    for sc in eval_scenarios(config.n_producers, config.n_consumers):
        bal = community_balance(make_profile_manager(config, sc.seed), config)
        out.append({"seed": sc.seed, "day": sc.day, "date": sc.date,
                    "producers": list(sc.producers), "consumers": list(sc.consumers), **bal})
    return out


def _per_scenario_block(scens: List[Dict], per: List[Dict[str, float]]) -> Dict[str, list]:
    """Senaryo başına diziler (JSON). E_max = 0 olan günde η = null."""
    return {
        "day_index": [sc["day"] for sc in scens],
        "eval_seed": [sc["seed"] for sc in scens],
        "e_max_kwh": [float(d["e_max_kwh"]) for d in per],
        "p2p_kwh": [float(d["p2p_kwh"]) for d in per],
        "efficiency_pct": [(float(d["p2p_kwh"] / d["e_max_kwh"] * 100) if d["e_max_kwh"] > 0 else None)
                           for d in per],
        "avg_price": [float(d["avg_price"]) for d in per],
        "seller_revenue": [float(d["seller_revenue"]) for d in per],
        "buyer_cost": [float(d["buyer_cost"]) for d in per],
    }


def _learned_state_fingerprint(env) -> str:
    """Q-tabloları / ağ ağırlıkları özeti — değerlendirmenin öğrenmediğini doğrulamak için."""
    import hashlib
    h = hashlib.sha256()
    for aid, a in sorted(getattr(env, "_ql_agents", {}).items()):
        h.update(np.ascontiguousarray(a.q_table).tobytes())
        h.update(np.ascontiguousarray(a.buy_q_table).tobytes())
    for aid, a in sorted(getattr(env, "_rf_agents", {}).items()):
        for net in (a.policy_net, a.buy_policy_net):
            for t in net.state_dict().values():
                h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _eval_snapshot(env) -> Dict[int, list]:
    """Günler arasında taşınan değerlendirme durumu: NN ajanlarının fiyat geçmişi.

    REINFORCEAgent._price_history reset()'te temizlenmez (episode'lar arası taşınır).
    Tek günlük değerlendirmede bu geçmiş son eğitim gününden gelir; çok senaryolu
    değerlendirmede her senaryo AYNI başlangıç geçmişiyle (eğitim sonu) başlatılır,
    böylece senaryolar birbirine sızmaz ve sıralarından bağımsızdır.
    """
    snap = {}
    for aid, a in getattr(env, "_rf_agents", {}).items():
        if hasattr(a, "_price_history"):
            snap[aid] = list(a._price_history)
    return snap


def _eval_restore(env, snap: Dict[int, list]) -> None:
    from collections import deque
    for aid, hist in snap.items():
        a = env._rf_agents[aid]
        a._price_history = deque(hist, maxlen=a._price_history.maxlen)


def _evaluate_ausgrid(env, config: SimConfig, scens: List[Dict]):
    """Eğitilmiş ortamı her senaryoda run_evaluation ile (öğrenme / keşif kapalı) koşturur."""
    snap = _eval_snapshot(env)
    fp0 = _learned_state_fingerprint(env)
    per = []
    for sc in scens:
        _eval_restore(env, snap)
        m = env.run_evaluation(seed=sc["seed"], verbose=False)
        per.append(_metrics_dict(m, config, sc["e_max_kwh"]))
    _eval_restore(env, snap)
    if _learned_state_fingerprint(env) != fp0:
        raise RuntimeError("Değerlendirme sırasında Q-tablosu / ağ ağırlıkları değişti")
    return aggregate_scenarios(per), per


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


def run_random_ausgrid(config: SimConfig, n_draws: int, scens: List[Dict]) -> Dict:
    """Ausgrid: her değerlendirme senaryosunda n_draws fiyat çekilişi.

    Senaryo k'da price_seed = k·n_draws + j (j = 0..n_draws-1; her senaryo-çekiliş
    çifti farklı fiyat tohumu). "Çekiliş j" = j'nin 73 senaryodaki sonuçlarının
    aggregate_scenarios toplamı; metrikler çekilişler üzerinden ortalama / std /
    min / max (sentetikteki tek gün × 200 çekilişle aynı yapı). Senaryo başına
    diziler çekiliş ortalamasıdır.
    """
    _log(f"random: başlıyor ({len(scens)} senaryo × {n_draws} fiyat çekilişi)")
    t0, c0 = time.perf_counter(), time.process_time()
    by_draw: List[List[Dict[str, float]]] = [[] for _ in range(n_draws)]
    per_scen: List[Dict[str, float]] = []
    for k, sc in enumerate(scens):
        with _silenced():
            res = run_random_baseline(config, n_draws=n_draws, seed=sc["seed"],
                                      base_price_seed=k * n_draws, verbose=False)
        if abs(res.e_max - sc["e_max_kwh"]) > 1e-9 * max(1.0, sc["e_max_kwh"]):
            raise RuntimeError(f"E_max uyuşmazlığı (senaryo {k})")
        draws_k = [{key: float(res.values[key][j]) for key in METRIC_KEYS} for j in range(n_draws)]
        for j in range(n_draws):
            by_draw[j].append(draws_k[j])
        # senaryo k'nın çekiliş ortalaması (doğrusal metrikler / n_draws, oranlar toplamdan)
        agg_k = aggregate_scenarios(draws_k)
        for key in SUM_KEYS:
            agg_k[key] /= n_draws
        per_scen.append(agg_k)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    agg = [aggregate_scenarios(lst) for lst in by_draw]
    vals = {k: np.array([a[k] for a in agg], dtype=float) for k in METRIC_KEYS}
    std = {k: (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0) for k, v in vals.items()}
    metrics = {k: float(np.mean(v)) for k, v in vals.items()}
    _log(f"random: bitti ({wall:.1f}s) | η={metrics['efficiency_pct']:.2f}%")
    return {
        "n_draws": n_draws,
        "metrics": metrics,
        "draw_std": std,
        "draw_min": {k: float(np.min(v)) for k, v in vals.items()},
        "draw_max": {k: float(np.max(v)) for k, v in vals.items()},
        "train_seconds": 0.0,
        "eval_seconds": wall,
        "cpu_seconds": cpu,
        "train_curve": None,
        "per_scenario": _per_scenario_block(scens, per_scen),
    }


def _run_learner(name: str, env, config: SimConfig, n_episodes: int, e_max: float,
                 scenarios: Optional[List[Dict]] = None) -> Dict:
    """Eğit + değerlendir. Sentetikte tek gün (seed = config.seed, E_max = e_max);
    Ausgrid modunda 73 değerlendirme senaryosu (scenarios; verilmezse burada kurulur,
    e_max kullanılmaz)."""
    ausgrid = getattr(config, "profile_source", "synthetic") == "ausgrid"
    if ausgrid:
        from src.ausgrid_profiles import assert_train_seeds_ok
        assert_train_seeds_ok(config.seed, n_episodes)
        if scenarios is None:
            scenarios = ausgrid_eval_community(config)
    _attach_progress(env, name, n_episodes)
    t0, c0 = time.perf_counter(), time.process_time()
    with _silenced():
        curve = env.train(n_episodes=n_episodes, verbose=False)
    t1 = time.perf_counter()
    per = None
    with _silenced():
        if ausgrid:
            metrics, per = _evaluate_ausgrid(env, config, scenarios)
        else:
            m = env.run_evaluation(seed=config.seed, verbose=False)
    t2, c2 = time.perf_counter(), time.process_time()
    if not ausgrid:
        metrics = _metrics_dict(m, config, e_max)
    _log(f"{name}: bitti (eğitim {t1 - t0:.1f}s, değerlendirme {t2 - t1:.2f}s) "
         f"| η={metrics['efficiency_pct']:.2f}%")
    out = {
        "episodes": n_episodes,
        "metrics": metrics,
        "train_seconds": t1 - t0,
        "eval_seconds": t2 - t1,
        "cpu_seconds": c2 - c0,
        "train_curve": [float(x) for x in curve],
    }
    if per is not None:
        out["per_scenario"] = _per_scenario_block(scenarios, per)
    return out


def run_ql(config: SimConfig, n_episodes: int, e_max: float,
           scenarios: Optional[List[Dict]] = None) -> Dict:
    from src.ql_environment import QLearningEnv
    _log(f"ql: başlıyor ({n_episodes} episode)")
    return _run_learner("ql", QLearningEnv(config), config, n_episodes, e_max, scenarios)


def run_nn(config: SimConfig, n_episodes: int, e_max: float,
           n_layers: int, hidden: int, scenarios: Optional[List[Dict]] = None) -> Dict:
    from src.reinforce_environment import REINFORCEEnv
    _log(f"nn: başlıyor ({n_episodes} episode, {n_layers}×{hidden} gizli katman)")
    env = REINFORCEEnv(config, hidden=hidden, n_layers=n_layers)
    out = _run_learner("nn", env, config, n_episodes, e_max, scenarios)
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
    config = SimConfig(n_producers=args.producers, n_consumers=args.consumers, seed=args.seed,
                       profile_source=args.profiles)
    ausgrid = config.profile_source == "ausgrid"
    draws = args.baseline_draws
    if draws is None:
        draws = 20 if ausgrid else 200

    if "nn" in methods:
        import torch
        torch.set_num_threads(1)   # tek iş parçacığı; ağ çok küçük, CPU yeterli

    scens: Optional[List[Dict]] = None
    if ausgrid:
        from src.ausgrid_profiles import assert_train_seeds_ok, split_identity
        assert_train_seeds_ok(config.seed, max(args.episodes, args.nn_episodes))
        # 73 değerlendirme senaryosu; topluluk dengesi senaryolar üzerinden toplanır
        scens = ausgrid_eval_community(config)
        day = {k: float(sum(sc[k] for sc in scens)) for k in ("surplus_kwh", "deficit_kwh", "e_max_kwh")}
    else:
        # Değerlendirme günü (gün seed'i = config.seed) için profil tabanlı denge
        day = community_balance(make_profile_manager(config, config.seed), config)
    e_max = day["e_max_kwh"]

    _log(f"İŞ başlıyor: {args.tag or '-'} | {config.n_producers}+{config.n_consumers} "
         f"| seed={config.seed} | profiller={config.profile_source} "
         f"| yöntemler={','.join(methods)} | E_max={e_max:.2f} kWh"
         + (f" ({len(scens)} senaryo toplamı)" if ausgrid else ""))
    t_job = time.perf_counter()
    results: Dict[str, Dict] = {}
    for m in methods:
        seed_everything(config.seed)   # her yöntem aynı global RNG durumundan başlar
        if m == "random":
            results[m] = (run_random_ausgrid(config, draws, scens) if ausgrid
                          else run_random(config, draws))
        elif m == "ql":
            results[m] = run_ql(config, args.episodes, e_max, scens)
        elif m == "nn":
            results[m] = run_nn(config, args.nn_episodes, e_max, args.nn_layers, args.nn_hidden,
                                scens)
    total = time.perf_counter() - t_job
    _log(f"İŞ bitti: toplam {total:.1f}s")

    extra: Dict = {}
    if ausgrid:
        extra["ausgrid"] = split_identity()
        extra["eval_scenarios"] = {
            "eval_seed": [sc["seed"] for sc in scens],
            "day_index": [sc["day"] for sc in scens],
            "date": [sc["date"] for sc in scens],
            "producers": [sc["producers"] for sc in scens],
            "consumers": [sc["consumers"] for sc in scens],
            "surplus_kwh": [sc["surplus_kwh"] for sc in scens],
            "deficit_kwh": [sc["deficit_kwh"] for sc in scens],
            "e_max_kwh": [sc["e_max_kwh"] for sc in scens],
        }

    return {
        "format": "energy-trading-experiment/1",
        "tag": args.tag,
        "population": {"producers": config.n_producers, "consumers": config.n_consumers,
                       "label": f"{config.n_producers}+{config.n_consumers}"},
        "seed": config.seed,
        "methods": methods,
        "profile_source": config.profile_source,
        "budget": {"ql_episodes": args.episodes, "nn_episodes": args.nn_episodes,
                   "baseline_draws": draws},
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
        **extra,
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
    p.add_argument("--baseline_draws", type=int, default=None,
                   help="Random baseline fiyat çekilişi sayısı (varsayılan: sentetikte 200, "
                        "Ausgrid'de senaryo başına 20)")
    p.add_argument("--profiles", choices=["synthetic", "ausgrid"], default="synthetic",
                   help="Profil kaynağı (varsayılan: synthetic = eski davranış)")
    p.add_argument("--out", type=str, required=True, help="Sonuç JSON dosyası")
    p.add_argument("--tag", type=str, default="", help="Serbest etiket (ör. deney adı A/B/C)")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    data = run_job(args)
    write_json_atomic(data, args.out)
    _log(f"JSON yazıldı: {args.out}")
