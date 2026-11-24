"""
Hızlı öz-denetim (< ~1 dk). Repo kökünden:  python -m ablation.selfcheck

  1. q_base (ablasyon yolu)  ≡ src.experiment.run_ql   (metrikler + eğitim eğrisi, birebir)
  2. nn_full (ablasyon yolu) ≡ src.experiment.run_nn   (metrikler + eğitim eğrisi, birebir)
  3. nn_nohist / nn_noprice ağları: girdi boyutu, kodlayıcı parametresi yok,
     çıktı fiyat geçmişine / son fiyata (olması gerektiği gibi) duyarsız
  4. q_price*: tablo şekilleri, kova sınırları, kısa koşuda fiyat kovası ziyaretleri
  5. Enerji dengesi (değerlendirme günü): D = P2P + şebeke alımı, S = P2P + şebeke satışı

Çıkış kodu: tümü geçerse 0, aksi halde 1.
"""

from __future__ import annotations

import contextlib
import io
import sys
import time

from ablation.variants import VARIANTS, Q_VARIANTS, NN_VARIANTS, chdir_repo_root

chdir_repo_root()

import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402

import src.experiment as sx                                    # noqa: E402
from src.agent import Observation                              # noqa: E402
from src.config import SimConfig                               # noqa: E402
from src.seeding import seed_everything                        # noqa: E402

from ablation.run_job import run_variant, community_for_eval_day, balance_check  # noqa: E402
from ablation.q_agents import price_bucket, bucket_edges, n_q_states  # noqa: E402
from ablation.nn_agents import AblationNNEnv, head_input_dim, n_params  # noqa: E402

P, C, SEED = 4, 4, 3
EP_Q, EP_NN = 40, 12

RESULTS = []


def report(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n       {detail}" if detail else ""), flush=True)


@contextlib.contextmanager
def _quiet():
    """src ortamının ve ilerleme satırlarının çıktısını atar (_OUT içe aktarımda yakalanır)."""
    import ablation.run_job as rj
    saved = (sx._OUT, rj._OUT)
    sink = io.StringIO()
    sx._OUT = rj._OUT = sink
    try:
        with contextlib.redirect_stdout(sink):
            yield
    finally:
        sx._OUT, rj._OUT = saved


def _compare(a: dict, b: dict):
    keys = sorted(a["metrics"])
    diffs = {k: abs(a["metrics"][k] - b["metrics"][k]) for k in keys}
    same_metrics = all(a["metrics"][k] == b["metrics"][k] for k in keys) and set(a["metrics"]) == set(b["metrics"])
    same_curve = a["train_curve"] == b["train_curve"]
    return same_metrics, same_curve, max(diffs.values()), diffs


def check_reproduction(config: SimConfig, e_max: float) -> None:
    # (1) Q
    with _quiet():
        seed_everything(config.seed)
        ref = sx.run_ql(config, EP_Q, e_max)
        abl, _ = run_variant("q_base", config, EP_Q, e_max, progress_frac=None)
    sm, sc, mx, _ = _compare(ref, abl)
    report("1. q_base ≡ src.experiment.run_ql", sm and sc,
           f"{P}+{C}, seed {SEED}, {EP_Q} ep | metrikler birebir: {sm} | eğitim eğrisi birebir: {sc} "
           f"| maks |Δ| = {mx:.3e} | η src={ref['metrics']['efficiency_pct']:.6f}% "
           f"abl={abl['metrics']['efficiency_pct']:.6f}% | eğri toplamı {sum(ref['train_curve']):.6f} / "
           f"{sum(abl['train_curve']):.6f}")
    # (2) NN
    with _quiet():
        seed_everything(config.seed)
        ref = sx.run_nn(config, EP_NN, e_max, n_layers=2, hidden=64)
        abl, env = run_variant("nn_full", config, EP_NN, e_max, progress_frac=None)
    sm, sc, mx, _ = _compare(ref, abl)
    same_params = ref["n_params_per_net"] == n_params(env._rf_agents[0].policy_net)
    report("2. nn_full ≡ src.experiment.run_nn", sm and sc and same_params,
           f"{P}+{C}, seed {SEED}, {EP_NN} ep | metrikler birebir: {sm} | eğitim eğrisi birebir: {sc} "
           f"| maks |Δ| = {mx:.3e} | η src={ref['metrics']['efficiency_pct']:.6f}% "
           f"abl={abl['metrics']['efficiency_pct']:.6f}% | parametre/ağ {ref['n_params_per_net']}")


def _obs(load: float, price: float, hour: int = 20) -> Observation:
    return Observation(inflexible_load=load, last_avg_price=price, hour=hour)


def check_nn_nets(config: SimConfig) -> None:
    seed_everything(SEED)
    lines, ok = [], True
    for variant in NN_VARIANTS:
        v = VARIANTS[variant]
        env = AblationNNEnv(config, state_dim=None if v["history"] else v["state_dim"])
        a = env._rf_agents[0]
        for role, net, enc in (("sell", a.policy_net, a._encode_state),
                               ("buy", a.buy_policy_net, a._encode_buyer_state)):
            in_dim = head_input_dim(net)
            has_enc = any(n.startswith("ts_encoder") for n, _ in net.named_parameters())
            exp_in = 19 if v["history"] else v["state_dim"]
            load = -4.0 if role == "sell" else 4.0
            with torch.no_grad():
                h1 = torch.linspace(0.0, 1.0, 12)
                h2 = torch.linspace(1.0, 0.0, 12) * 0.7
                s_lo, s_hi = enc(_obs(load, 0.10)), enc(_obs(load, 0.25))
                dim_ok = s_lo.numel() == v["state_dim"]
                out_h1 = float(net(s_lo, h1)); out_h2 = float(net(s_lo, h2))
                out_p_hi = float(net(s_hi, h1))
            d_hist, d_price = abs(out_h1 - out_h2), abs(out_h1 - out_p_hi)
            if variant == "nn_full":
                good = in_dim == 19 and has_enc and d_hist > 0 and d_price > 0
            elif variant == "nn_nohist":
                good = in_dim == 3 and not has_enc and d_hist == 0.0 and d_price > 0
            else:
                good = in_dim == 2 and not has_enc and d_hist == 0.0 and d_price == 0.0
            good = good and dim_ok and in_dim == exp_in
            ok &= good
            lines.append(f"{variant:<10} {role:<4} girdi={in_dim:>2} durum={s_lo.numel()} "
                         f"kodlayıcı={'var' if has_enc else 'yok'} param={n_params(net):>5} "
                         f"|Δçıktı| geçmiş={d_hist:.3e} sonfiyat={d_price:.3e} → {'OK' if good else 'HATA'}")
        # Uçtan uca: nn_noprice karar fiyatı, aynı gözlem/geçmiş farklı fiyatlarla aynı mı?
        if variant != "nn_full":
            a.training_mode = False
            with _quiet():
                p1 = a.decide_action(_obs(-4.0, 0.10)).order.price
                a._price_history.extend([0.27] * 12)
                p2 = a.decide_action(_obs(-4.0, 0.10)).order.price
                p3 = a.decide_action(_obs(-4.0, 0.25)).order.price
            e2e = (p1 == p2) and ((p1 == p3) if variant == "nn_noprice" else (p1 != p3))
            ok &= e2e
            lines.append(f"{variant:<10} decide_action: geçmiş değişince fiyat aynı={p1 == p2}, "
                         f"son fiyat değişince aynı={p1 == p3} → {'OK' if e2e else 'HATA'}")
    report("3. nn_nohist / nn_noprice ağları (boyut, kodlayıcı yok, girdi duyarsızlığı)", ok,
           "\n       ".join(lines))


def check_q_price(config: SimConfig, e_max: float) -> None:
    fit, tou = config.fit_price, config.tou_price
    lines, ok = [], True
    # Kova fonksiyonu sınırları
    for K in (3, 5):
        edges = bucket_edges(fit, tou, K)
        probes = [0.0, -1.0, fit - 0.01, fit] + [float(e) for e in edges[1:-1]] + [tou, tou + 0.05]
        got = [price_bucket(p, fit, tou, K) for p in probes]
        exp = [0, 0, 1, 1] + list(range(2, K + 1)) + [K, K]
        just_below = [price_bucket(float(e) - 1e-9, fit, tou, K) for e in edges[1:-1]]
        good = got == exp and just_below == list(range(1, K))
        ok &= good
        lines.append(f"K={K} sınırlar={np.round(edges, 4).tolist()} | yoklama {np.round(probes, 4).tolist()} "
                     f"→ {got} (beklenen {exp}); sınırın hemen altı → {just_below} → {'OK' if good else 'HATA'}")
    for variant in [v for v in Q_VARIANTS if VARIANTS[v]["price_buckets"] and VARIANTS[v]["ep_mult"] == 1]:
        K = VARIANTS[variant]["price_buckets"]
        with _quiet():
            res, env = run_variant(variant, config, EP_Q, e_max, progress_frac=None)
        d = env.diagnostics()
        a0 = env._ql_agents[0]
        shape_ok = (list(a0.q_table.shape) == [3, 4, K + 1, 15] and list(a0.buy_q_table.shape) == [3, 4, K + 1, 15]
                    and d["states_per_table"] == n_q_states(K) == 12 * (K + 1))
        tv = d["price_bucket_visits_train"]
        visited_price = sum(1 for x in tv[1:] if x > 0)
        good = shape_ok and visited_price >= 2 and tv[0] > 0
        ok &= good
        lines.append(f"{variant}: tablo {a0.q_table.shape}, durum/tablo {d['states_per_table']} | "
                     f"eğitim ziyaretleri [takas yok, kova1..{K}] = {tv} | değerlendirme = "
                     f"{d['price_bucket_visits_eval']} | ziyaret edilen durum (ort.) satıcı "
                     f"{d['visited_states_sell_mean']:.1f} alıcı {d['visited_states_buy_mean']:.1f} "
                     f"| η={res['metrics']['efficiency_pct']:.2f}% → {'OK' if good else 'HATA'}")
    report("4. q_price* (tablo şekli, kova sınırları, fiyat kovası ziyaretleri)", ok, "\n       ".join(lines))


def check_balance(config: SimConfig, e_max: float) -> None:
    lines, ok = [], True
    for variant in VARIANTS:
        if VARIANTS[variant].get("ep_mult", 1) != 1:
            continue
        n = 6
        with _quiet():
            res, env = run_variant(variant, config, n, e_max, progress_frac=None)
        b = balance_check(env, config)
        ok &= b["ok"]
        lines.append(f"{variant:<11} |D−P2P−grid_buy|={b['abs_D_minus_p2p_minus_gridbuy']:.2e} "
                     f"|S−P2P−grid_sell|={b['abs_S_minus_p2p_minus_gridsell']:.2e} → {'OK' if b['ok'] else 'HATA'}")
    report("5. Enerji dengesi (src.random_environment.community_balance ile)", ok, "\n       ".join(lines))


def main() -> int:
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    config = SimConfig(n_producers=P, n_consumers=C, seed=SEED)
    e_max = community_for_eval_day(config)["e_max_kwh"]
    print(f"ablation.selfcheck | {P}+{C}, seed {SEED}, Q {EP_Q} ep, NN {EP_NN} ep | "
          f"E_max={e_max:.3f} kWh | torch {torch.__version__}, numpy {np.__version__}\n", flush=True)
    check_reproduction(config, e_max)
    check_nn_nets(config)
    check_q_price(config, e_max)
    check_balance(config, e_max)
    n_ok = sum(ok for _, ok in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} kontrol geçti ({time.perf_counter() - t0:.1f}s)")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
