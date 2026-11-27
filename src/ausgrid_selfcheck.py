"""
Ausgrid veri katmanı doğruluk kontrolü — EĞİTİM YOK, öğrenen bölüm koşulmaz.

    python -m src.ausgrid_selfcheck

Kontroller: havuz boyutları ve ayrıklık, gün bölünmesi, senaryoların determinizmi ve
eğitim/test ayrımı, değerlendirme listesinin deney seed'inden bağımsızlığı, profil
içeriği (üretici PV = GG, tüketici PV = 0, yük = GC + CL), talep/arz çarpanları,
split.json'daki ölçek sabitlerinin yeniden hesapla birebir aynı olması, sentetik
varsayılanların eski sabitlerle aynı olması ve sentetik fabrika yolunun eski
doğrudan ProfileManager çağrısıyla bit düzeyinde aynı olması.
Hata varsa çıkış kodu 1.
"""

from __future__ import annotations

import contextlib
import io
import sys

import numpy as np

from src import ausgrid_profiles as ap
from src.config import SimConfig
from src.profiles import ProfileManager, make_profile_manager

POPS = [(4, 4), (10, 10), (20, 20), (10, 20)]
_FAILS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if ok else 'HATA'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        _FAILS.append(name)


def raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def main() -> int:
    print("1. Veri ve bölünme")
    data = ap.load_data()
    split = ap.load_split()
    check("veri boyutları [300, 366, 48]", data["gc"].shape == data["gg"].shape == data["cl"].shape
          == (300, 366, 48), str(data["gc"].shape))
    check("CL'li ev sayısı", True, str(int(data["has_cl"].sum())))
    pt, pe = split["producers"]["train"], split["producers"]["test"]
    ct, ce = split["consumers"]["train"], split["consumers"]["test"]
    check("havuz boyutları 80/20/170/30", (len(pt), len(pe), len(ct), len(ce)) == (80, 20, 170, 30),
          f"{len(pt)}/{len(pe)}/{len(ct)}/{len(ce)}")
    allh = set(pt) | set(pe) | set(ct) | set(ce)
    check("havuzlar ayrık ve 300 evi kapsıyor",
          len(allh) == 300 and len(pt) + len(pe) + len(ct) + len(ce) == 300 and allh == set(range(1, 301)))
    cap = data["capacity"]
    prod = set(pt) | set(pe)
    cmin_p = min(cap[c - 1] for c in prod)
    cmax_c = max(cap[c - 1] for c in set(ct) | set(ce))
    ties = sorted(c for c in range(1, 301) if cap[c - 1] == cmin_p)
    expected = set(sorted(range(1, 301), key=lambda c: (-cap[c - 1], c))[:100])
    check("üretici havuzu = en büyük 100 kapasite (eşitlikte küçük id)",
          prod == expected and cmin_p >= cmax_c,
          f"üretici kapasite min {cmin_p} kW, tüketici max {cmax_c} kW; "
          f"{cmin_p} kW'lık evler {ties}, bunlardan üretici: {[c for c in ties if c in prod]}")
    trd, ted = ap.train_days(), ap.test_days()
    check("gün bölünmesi 293/73, ayrık", len(trd) == 293 and len(ted) == 73
          and not set(trd) & set(ted) and all(d % 5 == 4 for d in ted))

    print("2. Ölçek sabitleri (yalnızca eğitim verisi)")
    re = ap.compute_split(data)
    check("split.json yeniden hesapla birebir aynı", re == split,
          f"split_id={split['split_id']} (yeniden: {re['split_id']})")
    sc = split["scales"]
    print(f"       q_surplus_edges = {sc['q_surplus_edges']}")
    print(f"       q_deficit_edges = {sc['q_deficit_edges']}")
    print(f"       nn_surplus_norm = {sc['nn_surplus_norm']}, nn_deficit_norm = {sc['nn_deficit_norm']}")
    print(f"       örnek sayısı: fazla {sc['n_surplus_samples']}, açık {sc['n_deficit_samples']}")
    cfg_a = SimConfig(profile_source="ausgrid")
    check("SimConfig(profile_source='ausgrid') ölçekleri split.json'dan",
          list(cfg_a.q_surplus_edges) == sc["q_surplus_edges"]
          and list(cfg_a.q_deficit_edges) == sc["q_deficit_edges"]
          and cfg_a.nn_surplus_norm == sc["nn_surplus_norm"]
          and cfg_a.nn_deficit_norm == sc["nn_deficit_norm"])
    check("kenarlar artan", all(np.diff(sc["q_surplus_edges"]) > 0) and all(np.diff(sc["q_deficit_edges"]) > 0))

    print("3. Sentetik varsayılanlar (bit düzeyinde eski davranış)")
    cfg = SimConfig()
    check("SimConfig() varsayılanı synthetic", cfg.profile_source == "synthetic")
    check("varsayılan Q kenarları = [0,1,3,6,20], NN norm = 10.0",
          cfg.q_surplus_edges == (0.0, 1.0, 3.0, 6.0, 20.0) and cfg.q_deficit_edges == (0.0, 1.0, 3.0, 6.0, 20.0)
          and cfg.nn_surplus_norm == 10.0 and cfg.nn_deficit_norm == 10.0)
    from src.ql_agent import QLearningAgent, PV_EDGES, DEMAND_EDGES, _pv_bucket, _demand_bucket
    from src.agent import Observation
    qa = QLearningAgent(0, "producer", cfg)
    probes = [0.0, 1e-12, 0.0099, 0.5, 0.999999, 1.0, 1.0000001, 2.9, 3.0, 5.99, 6.0, 6.01, 19.99,
              20.0, 25.0, 100.0] + list(np.linspace(0, 30, 3001))
    same = all(qa._encode_state(Observation(-x, 0.1, 20)) == (1, _pv_bucket(x)) and
               qa._encode_buyer_state(Observation(x, 0.1, 20)) == (1, _demand_bucket(x)) and
               _pv_bucket(x) == int(np.clip(np.searchsorted(PV_EDGES, x, side="right") - 1, 0, 3))
               for x in probes)
    check("QL kovaları eski sabit dizilerle aynı", same and np.array_equal(qa._surplus_edges, PV_EDGES)
          and np.array_equal(qa._deficit_edges, DEMAND_EDGES))
    qa_a = QLearningAgent(0, "producer", cfg_a)
    check("Ausgrid modunda QL ajanı veri kenarlarını kullanır",
          list(qa_a._surplus_edges) == sc["q_surplus_edges"] and list(qa_a._deficit_edges) == sc["q_deficit_edges"])
    try:
        import torch  # noqa: F401
        from src.reinforce_agent import REINFORCEAgent
        ra = REINFORCEAgent(0, "producer", cfg)
        ok_nn = True
        for x in probes[:200]:
            s1 = ra._encode_state(Observation(-x, 0.1, 20))
            s2 = ra._encode_buyer_state(Observation(x, 0.1, 20))
            ok_nn &= float(s1[1]) == float(torch.tensor(min(x / 10.0, 1.0), dtype=torch.float32))
            ok_nn &= float(s2[1]) == float(torch.tensor(min(x / 10.0, 1.0), dtype=torch.float32))
        ra_a = REINFORCEAgent(0, "producer", cfg_a)
        x = 0.5
        ok_nn &= float(ra_a._encode_state(Observation(-x, 0.1, 20))[1]) == float(
            torch.tensor(min(x / sc["nn_surplus_norm"], 1.0), dtype=torch.float32))
        check("NN normalizasyonu: varsayılan = eski '/ 10.0', Ausgrid = split.json", ok_nn)
    except ImportError:
        print("  [ATLA] torch yok; NN normalizasyon kontrolü atlandı")
    same_pm = True
    for (p, c) in POPS:
        for seed in (1, 2, 1009):
            for dm, sm in ((1.0, 1.0), (1.5, 1.0), (1.0, 0.5)):
                cfg_s = SimConfig(n_producers=p, n_consumers=c, demand_multiplier=dm, supply_multiplier=sm)
                a = make_profile_manager(cfg_s, seed)
                b = ProfileManager(p, c, seed=seed, demand_multiplier=dm, supply_multiplier=sm)
                same_pm &= type(a) is ProfileManager and all(
                    np.array_equal(x, y) for x, y in zip(a.pv_profiles + a.load_profiles,
                                                         b.pv_profiles + b.load_profiles))
    check("sentetik fabrika = doğrudan ProfileManager (bit düzeyinde)", same_pm)

    print("4. Senaryolar")
    trset, teset = set(int(d) for d in trd), set(int(d) for d in ted)
    ok_train = ok_det = True
    for (p, c) in POPS:
        for s in range(1, 1501):
            a = ap.train_scenario(s, p, c)
            ok_train &= (a.day in trset and set(a.producers) <= set(pt) and set(a.consumers) <= set(ct)
                         and len(set(a.producers)) == p and len(set(a.consumers)) == c)
            if s <= 50:
                ok_det &= a == ap.train_scenario(s, p, c) == ap.scenario_for_seed(s, p, c)
    check("eğitim senaryoları yalnızca eğitim evleri + günleri, farklı evler", ok_train)
    days_used = {ap.train_scenario(s, 10, 10).day for s in range(1, 7001)}
    check("eğitim senaryoları günleri kapsıyor (10+10, seed 1..7000)", len(days_used) == 293,
          f"{len(days_used)} / 293 gün")
    ok_eval = True
    for (p, c) in POPS:
        ev = ap.eval_scenarios(p, c)
        ok_eval &= (len(ev) == 73 and [e.day for e in ev] == [int(d) for d in ted]
                    and all(set(e.producers) <= set(pe) and set(e.consumers) <= set(ce)
                            and len(set(e.producers)) == p and len(set(e.consumers)) == c for e in ev)
                    and [e.seed for e in ev] == [ap.EVAL_SEED_BASE + k for k in range(73)])
        ap._EVAL_CACHE.clear()
        ok_det &= ev == ap.eval_scenarios(p, c)
    check("değerlendirme senaryoları yalnızca test evleri + test günleri (73, sırayla)", ok_eval)
    check("senaryolar deterministik (önbellek temizlenince de aynı)", ok_det)
    ev20 = ap.eval_scenarios(20, 20)
    check("20+20 değerlendirmesi 20 test üreticisinin hepsini kullanır",
          all(sorted(e.producers) == sorted(pe) for e in ev20))
    check("havuzdan büyük popülasyon açık hatayla reddedilir",
          raises(lambda: ap.eval_scenarios(21, 4), ValueError)
          and raises(lambda: ap.train_scenario(1, 81, 4), ValueError)
          and raises(lambda: ap.eval_scenarios(4, 31), ValueError))
    check("ayrılmış seed aralığı korunur",
          raises(lambda: ap.scenario_for_seed(ap.EVAL_SEED_BASE + 73, 4, 4), ValueError)
          and raises(lambda: ap.train_scenario(ap.EVAL_SEED_BASE, 4, 4), ValueError)
          and raises(lambda: ap.assert_train_seeds_ok(ap.EVAL_SEED_BASE - 10, 11), AssertionError)
          and not raises(lambda: ap.assert_train_seeds_ok(5, 14000)))

    print("5. Profil içeriği, çarpanlar, deney seed'inden bağımsızlık")
    ok_prof = True
    for (p, c) in POPS:
        cfg_p = SimConfig(n_producers=p, n_consumers=c, profile_source="ausgrid")
        for seed in (1, 2, 3, ap.EVAL_SEED_BASE, ap.EVAL_SEED_BASE + 72):
            pm = make_profile_manager(cfg_p, seed)
            sc_ = pm.scenario
            ok_prof &= isinstance(pm, ap.AusgridProfileManager)
            for i in range(p + c):
                cid = pm.customer_id(i)
                load = (np.round(data["gc"][cid - 1, sc_.day].astype(np.float64), 3)
                        + np.round(data["cl"][cid - 1, sc_.day].astype(np.float64), 3))
                ok_prof &= np.array_equal(pm.load_profiles[i], load)
                if i < p:
                    ok_prof &= np.array_equal(pm.pv_profiles[i],
                                              np.round(data["gg"][cid - 1, sc_.day].astype(np.float64), 3))
                else:
                    ok_prof &= not pm.pv_profiles[i].any()
                ok_prof &= pm.get_inflexible_load(i, 17) == float(pm.load_profiles[i][17]) - float(pm.pv_profiles[i][17])
    check("üretici PV = GG, tüketici PV = 0, yük = GC + CL, hepsi aynı gün", ok_prof)
    cfg_p = SimConfig(n_producers=10, n_consumers=10, profile_source="ausgrid")
    base = make_profile_manager(cfg_p, 7)
    dm = make_profile_manager(SimConfig(n_producers=10, n_consumers=10, profile_source="ausgrid",
                                        demand_multiplier=2.0), 7)
    sm = make_profile_manager(SimConfig(n_producers=10, n_consumers=10, profile_source="ausgrid",
                                        supply_multiplier=0.5), 7)
    ok_mul = (all(np.array_equal(dm.load_profiles[i], base.load_profiles[i] * (2.0 if i >= 10 else 1.0))
                  and np.array_equal(dm.pv_profiles[i], base.pv_profiles[i]) for i in range(20))
              and all(np.array_equal(sm.pv_profiles[i], base.pv_profiles[i] * (0.5 if i < 10 else 1.0))
                      and np.array_equal(sm.load_profiles[i], base.load_profiles[i]) for i in range(20)))
    check("talep çarpanı yalnızca tüketici yükü, arz çarpanı yalnızca üretici PV'si", ok_mul)

    from src.experiment import ausgrid_eval_community
    e1 = ausgrid_eval_community(SimConfig(n_producers=4, n_consumers=4, seed=1, profile_source="ausgrid"))
    e5 = ausgrid_eval_community(SimConfig(n_producers=4, n_consumers=4, seed=5, profile_source="ausgrid"))
    check("değerlendirme listesi deney seed'inden bağımsız (seed 1 = seed 5)", e1 == e5)

    from src.environment import EnergyTradingEnv
    env = EnergyTradingEnv(SimConfig(n_producers=4, n_consumers=4, seed=3, profile_source="ausgrid"))
    with contextlib.redirect_stdout(io.StringIO()):
        obs = env.reset(seed=ap.EVAL_SEED_BASE + 5)      # yalnızca reset; adım / öğrenme yok
    check("EnergyTradingEnv.reset Ausgrid profillerini kullanır",
          isinstance(env.profiles, ap.AusgridProfileManager)
          and env.profiles.scenario == ap.eval_scenarios(4, 4)[5]
          and all(obs[i].inflexible_load == env.profiles.get_inflexible_load(i, 0) for i in range(8)))

    print("6. Bilgi: miktar ölçeği ve değerlendirme toplulukları")
    for (p, c) in POPS:
        ev = ausgrid_eval_community(SimConfig(n_producers=p, n_consumers=c, profile_source="ausgrid"))
        em = np.array([e["e_max_kwh"] for e in ev])
        print(f"       {p}+{c}: ΣE_max = {em.sum():8.2f} kWh, gün başına E_max medyan {np.median(em):6.2f}, "
              f"min {em.min():6.2f}, E_max = 0 olan gün {int((em == 0).sum())}, "
              f"ΣS = {sum(e['surplus_kwh'] for e in ev):8.2f}, ΣD = {sum(e['deficit_kwh'] for e in ev):8.2f}")
    nets = []
    for s in range(1, 201):
        pm = make_profile_manager(SimConfig(n_producers=10, n_consumers=10, profile_source="ausgrid"), s)
        nets.append(np.array([pm.load_profiles[i] - pm.pv_profiles[i] for i in range(20)]))
    a = np.abs(np.concatenate(nets).ravel())
    print(f"       10+10 eğitim senaryoları (200 gün) slot başına |net|: medyan {np.median(a):.3f}, "
          f"p99 {np.percentile(a, 99):.3f} kWh; |net| <= 0.01 (emir yok) payı {np.mean(a <= 0.01) * 100:.2f}%, "
          f"0 < |net| <= 0.01 payı {np.mean((a > 0) & (a <= 0.01)) * 100:.2f}%")

    print()
    if _FAILS:
        print(f"BAŞARISIZ ({len(_FAILS)}): " + "; ".join(_FAILS))
        return 1
    print("Tüm kontroller geçti.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
