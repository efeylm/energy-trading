"""
Gerçek Ausgrid ev verisiyle günlük profiller (SimConfig.profile_source = "ausgrid").

Veri: data/ausgrid/2011-2012 Solar home electricity data v2.csv
  300 ev (müşteri id 1..300), 366 gün (1 Tem 2011 .. 30 Haz 2012), satır başına 48
  yarım saatlik değer, birim: YARIM SAAT BAŞINA kWh. Kategoriler: GC (genel tüketim),
  CL (kontrollü yük, yalnızca bazı evler), GG (brüt PV üretimi).
  Sütun "0:30" = 00:00-00:30 aralığı (aralık SONU etiketi) → simülasyon slotu t = 0;
  "0:00" (son sütun) = 23:30-24:00 → t = 47.

Birim: simülatör gözlemdeki net yükü (Observation.inflexible_load) DOĞRUDAN emir
miktarı (kWh) olarak kullanır; delta_t hiçbir yerde uygulanmaz. Ausgrid değerleri
zaten slot başına kWh olduğundan dönüştürülmeden kullanılır: bir ajanın bir slotta
işlem gördüğü enerji = o slotun Ausgrid kWh değeri.

Roller (sabit, deterministik):
  * Üretici havuzu : Generator Capacity'si en büyük 100 ev (eşitlikte küçük id).
                     Ajan: yük = GC + CL, PV = GG.
  * Tüketici havuzu: kalan 200 ev. Ajan: yük = GC + CL, PV = 0 (PV'leri yok sayılır).
Ev bölünmesi (rol içinde, SPLIT_SEED ile tek sefer): üretici 80 eğitim / 20 test,
tüketici 170 eğitim / 30 test. Gün bölünmesi: d % 5 == 4 → test (73 gün), kalan
293 gün eğitim (d = 0 → 1 Tem 2011).

Senaryo = bir topluluk (üretici + tüketici evleri) × bir takvim günü; bütün ajanlar
AYNI günü yaşar (hava ortaktır), gürültü eklenmez.
  * Eğitim senaryosu (seed s, 0 <= s < EVAL_SEED_BASE): s'den tohumlanan RNG bir
    eğitim günü, eğitim üreticilerinden n_p ve eğitim tüketicilerinden n_c farklı
    ev seçer. (s, n_p, n_c)'nin deterministik fonksiyonudur.
  * Değerlendirme senaryoları: test günleri sırasıyla 73 senaryo; k'ıncısı için
    (EVAL_ENTROPY, k) ile tohumlanan RNG topluluğu TEST evlerinden çeker. Liste
    yalnızca (n_p, n_c)'ye bağlıdır, deney seed'ine asla. Mevcut reset(seed=...) /
    run_evaluation(seed=...) tesisatı kullanılabilsin diye ayrılmış seed aralığı
    EVAL_SEED_BASE + k ile erişilir.

Ölçeğe bağlı sabitler (Q kova sınırları, NN normalizasyonu) yalnızca EĞİTİM verisinden
(eğitim evleri × eğitim günleri) bir kez hesaplanır ve bölünmeyle birlikte
data/ausgrid/split.json'a yazılır (depoya girer; makale bu değerleri raporlayabilir).
SimConfig(profile_source="ausgrid") bu değerleri JSON'dan okur (bkz. config.py).

CSV'yi ayrıştırmak birkaç saniye sürer; sonuç CACHE_PATH'e (.npz, .gitignore'da)
atomik olarak yazılır ve sonraki çağrılar onu okur. Paralel koşucu (run_experiments)
önbelleği işçileri başlatmadan önce ana süreçte kurar.
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.profiles import ProfileManager

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "ausgrid")
CSV_PATH = os.path.join(DATA_DIR, "2011-2012 Solar home electricity data v2.csv")
CACHE_PATH = os.path.join(DATA_DIR, "ausgrid_2011_2012_cache.npz")
SPLIT_PATH = os.path.join(DATA_DIR, "split.json")

CACHE_FORMAT = "ausgrid-cache/1"
SPLIT_FORMAT = "ausgrid-split/1"

N_CUSTOMERS = 300
N_DAYS = 366
N_SLOTS = 48
START_DATE = _dt.date(2011, 7, 1)
DECIMALS = 3                      # CSV değerleri en fazla 3 ondalık (ayrıştırmada doğrulanır)

N_PRODUCER_POOL = 100
N_PRODUCER_TEST = 20              # → 80 eğitim
N_CONSUMER_TEST = 30              # → 170 eğitim
TEST_DAY_MOD = 5
TEST_DAY_REM = 4                  # d % 5 == 4 → test günü (73 gün)

# Sabit tohumlar — deney seed'inden bağımsız.
SPLIT_SEED = 20111                # ev bölünmesi
TRAIN_ENTROPY = 7_301_2011        # eğitim senaryosu RNG'si: SeedSequence([TRAIN_ENTROPY, s])
EVAL_ENTROPY = 7_302_2011         # değerlendirme senaryosu RNG'si: SeedSequence([EVAL_ENTROPY, k])

# Ayrılmış değerlendirme seed aralığı: EVAL_SEED_BASE + k, k = 0..72.
# Eğitim seed'leri (config.seed + episode) bu değerin altında kalmak ZORUNDADIR.
EVAL_SEED_BASE = 1_000_000_000

# Q kova sınırlarının üst ucu. searchsorted + clip yüzünden son sınır hiçbir kovayı
# değiştirmez (sentetikteki 20.0 ile aynı rol); okunaklılık için aynı değer.
Q_TOP_EDGE = 20.0
# Rol eşiği: |net| <= ROLE_EPS olan ajan emir vermez (agent'lardaki 0.01 ile aynı).
ROLE_EPS = 0.01


# ---------------------------------------------------------------------------
# Ham veri: CSV → önbellek (.npz)
# ---------------------------------------------------------------------------

def _parse_csv(path: str = CSV_PATH) -> Dict[str, np.ndarray]:
    """CSV'yi [300, 366, 48] float32 dizilerine ayrıştırır (GC, CL, GG) + kapasite."""
    gc = np.zeros((N_CUSTOMERS, N_DAYS, N_SLOTS), dtype=np.float32)
    cl = np.zeros((N_CUSTOMERS, N_DAYS, N_SLOTS), dtype=np.float32)
    gg = np.zeros((N_CUSTOMERS, N_DAYS, N_SLOTS), dtype=np.float32)
    seen = {k: np.zeros((N_CUSTOMERS, N_DAYS), dtype=bool) for k in ("GC", "CL", "GG")}
    capacity = np.full(N_CUSTOMERS, np.nan, dtype=np.float64)
    arrays = {"GC": gc, "CL": cl, "GG": gg}
    max_dec = 0

    with open(path, newline="", encoding="utf-8-sig") as f:
        next(f)                                   # 1. satır: veri notu
        reader = csv.reader(f)
        header = next(reader)
        if header[:5] != ["Customer", "Generator Capacity", "Postcode",
                          "Consumption Category", "date"] or len(header) < 5 + N_SLOTS:
            raise ValueError(f"Beklenmeyen Ausgrid başlığı: {header[:8]}")
        if header[5] != "0:30" or header[5 + N_SLOTS - 1] != "0:00":
            raise ValueError("Beklenmeyen saat sütunları (0:30 .. 0:00 bekleniyordu)")
        for row in reader:
            if not row or not row[0].strip():
                continue
            cid = int(row[0])
            if not 1 <= cid <= N_CUSTOMERS:
                raise ValueError(f"Beklenmeyen müşteri id: {cid}")
            cap = float(row[1])
            if np.isnan(capacity[cid - 1]):
                capacity[cid - 1] = cap
            elif capacity[cid - 1] != cap:
                raise ValueError(f"Müşteri {cid}: tutarsız Generator Capacity")
            cat = row[3].strip()
            if cat not in arrays:
                raise ValueError(f"Beklenmeyen kategori: {cat}")
            d_, m_, y_ = (int(x) for x in row[4].split("/"))
            day = (_dt.date(y_, m_, d_) - START_DATE).days
            if not 0 <= day < N_DAYS:
                raise ValueError(f"Aralık dışı tarih: {row[4]}")
            if seen[cat][cid - 1, day]:
                raise ValueError(f"Tekrarlanan satır: müşteri {cid}, {cat}, {row[4]}")
            seen[cat][cid - 1, day] = True
            vals = row[5:5 + N_SLOTS]
            for v in vals:
                if "." in v:
                    max_dec = max(max_dec, len(v.split(".")[1]))
            arrays[cat][cid - 1, day] = np.array([float(v) for v in vals], dtype=np.float32)

    if np.isnan(capacity).any():
        raise ValueError("Bazı müşteriler veride yok")
    if not (seen["GC"].all() and seen["GG"].all()):
        raise ValueError("Eksik GC/GG ev-günü var")
    if max_dec > DECIMALS:
        raise ValueError(f"CSV'de {max_dec} ondalık var; DECIMALS={DECIMALS} varsayımı geçersiz")
    if (gc < 0).any() or (cl < 0).any() or (gg < 0).any():
        raise ValueError("Negatif değer var")
    return {"gc": gc, "cl": cl, "gg": gg, "capacity": capacity,
            "has_cl": seen["CL"].any(axis=1)}


def _csv_signature(path: str = CSV_PATH) -> str:
    return f"{os.path.basename(path)}|{os.path.getsize(path)}"


def ensure_cache(verbose: bool = True) -> str:
    """Önbellek yoksa (veya CSV değiştiyse) CSV'den kurar; yolunu döndürür.

    Atomik: önce süreç id'li geçici dosyaya yazılır, sonra os.replace ile taşınır;
    eşzamanlı iki kurucu birbirini bozmaz.
    """
    sig = _csv_signature()
    if os.path.exists(CACHE_PATH):
        try:
            with np.load(CACHE_PATH) as z:
                if str(z["format"]) == CACHE_FORMAT and str(z["signature"]) == sig:
                    return CACHE_PATH
        except Exception:                          # bozuk / eski önbellek → yeniden kur
            pass
    if verbose:
        print(f"[ausgrid] CSV ayrıştırılıyor → {CACHE_PATH}", flush=True)
    d = _parse_csv()
    tmp = f"{CACHE_PATH}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, format=np.array(CACHE_FORMAT), signature=np.array(sig), **d)
    os.replace(tmp, CACHE_PATH)
    return CACHE_PATH


_DATA: Optional[Dict[str, np.ndarray]] = None


def load_data() -> Dict[str, np.ndarray]:
    """Önbellekten (gerekirse kurarak) veriyi süreç başına bir kez yükler."""
    global _DATA
    if _DATA is None:
        ensure_cache()
        with np.load(CACHE_PATH) as z:
            _DATA = {k: z[k] for k in ("gc", "cl", "gg", "capacity", "has_cl")}
    return _DATA


def _kwh(x: np.ndarray) -> np.ndarray:
    """float32 önbellek değeri → CSV'deki ondalık kWh değeri (float64).

    CSV en fazla 3 ondalıklı olduğundan yuvarlama, float32 dönüşüm hatasını
    tam olarak geri alır (0.219f → 0.219).
    """
    return np.round(np.asarray(x, dtype=np.float64), DECIMALS)


def home_load(data: Dict[str, np.ndarray], cid: int, day: int) -> np.ndarray:
    """Ev yükü = GC + CL (kWh / yarım saat), float64 [48]."""
    return _kwh(data["gc"][cid - 1, day]) + _kwh(data["cl"][cid - 1, day])


def home_pv(data: Dict[str, np.ndarray], cid: int, day: int) -> np.ndarray:
    """Ev PV'si = GG (kWh / yarım saat), float64 [48]."""
    return _kwh(data["gg"][cid - 1, day])


# ---------------------------------------------------------------------------
# Bölünme (evler, günler) ve ölçek sabitleri → split.json
# ---------------------------------------------------------------------------

def train_days() -> np.ndarray:
    return np.array([d for d in range(N_DAYS) if d % TEST_DAY_MOD != TEST_DAY_REM], dtype=np.int64)


def test_days() -> np.ndarray:
    return np.array([d for d in range(N_DAYS) if d % TEST_DAY_MOD == TEST_DAY_REM], dtype=np.int64)


def _role_pools(capacity: np.ndarray) -> Tuple[List[int], List[int]]:
    """(üretici havuzu, tüketici havuzu): kapasiteye göre ilk 100, eşitlikte küçük id."""
    ids = list(range(1, N_CUSTOMERS + 1))
    order = sorted(ids, key=lambda c: (-float(capacity[c - 1]), c))
    prod = sorted(order[:N_PRODUCER_POOL])
    cons = sorted(order[N_PRODUCER_POOL:])
    return prod, cons


def _split_pool(pool: Sequence[int], n_test: int, rng: np.random.Generator) -> Tuple[List[int], List[int]]:
    perm = rng.permutation(np.asarray(pool, dtype=np.int64))
    test = sorted(int(c) for c in perm[:n_test])
    train = sorted(int(c) for c in perm[n_test:])
    return train, test


def _scale_samples(data: Dict[str, np.ndarray], prod_train: Sequence[int],
                   cons_train: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Eğitim evleri × eğitim günleri: (pozitif üretici fazlası, pozitif açık) örnekleri.

    Yalnızca bir ajanın emir verdiği (|net| > ROLE_EPS) slotlar alınır: Q kovası ve
    NN girdisi yalnızca bu slotlarda kodlanır. Fazla: üreticilerin max(PV - yük, 0)
    değerleri (tüketicilerin PV'si 0). Açık: hem üreticilerin hem tüketicilerin
    max(yük - PV, 0) değerleri (ikisi de alıcı olabilir), birlikte havuzlanır.
    """
    days = train_days()
    surplus, deficit = [], []
    for c in prod_train:
        for d in days:
            net = home_load(data, c, int(d)) - home_pv(data, c, int(d))
            surplus.append(-net[net < -ROLE_EPS])
            deficit.append(net[net > ROLE_EPS])
    for c in cons_train:
        for d in days:
            net = home_load(data, c, int(d))
            deficit.append(net[net > ROLE_EPS])
    return np.concatenate(surplus), np.concatenate(deficit)


def _split_id(obj: Dict) -> str:
    body = {k: v for k, v in obj.items() if k != "split_id"}
    s = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def compute_split(data: Optional[Dict[str, np.ndarray]] = None) -> Dict:
    """Bölünmeyi ve ölçek sabitlerini veriden deterministik olarak hesaplar."""
    if data is None:
        data = load_data()
    prod, cons = _role_pools(data["capacity"])
    rng = np.random.default_rng(SPLIT_SEED)
    prod_train, prod_test = _split_pool(prod, N_PRODUCER_TEST, rng)
    cons_train, cons_test = _split_pool(cons, N_CONSUMER_TEST, rng)

    surplus, deficit = _scale_samples(data, prod_train, cons_train)
    q_s = [float(x) for x in np.percentile(surplus, [25, 50, 75])]
    q_d = [float(x) for x in np.percentile(deficit, [25, 50, 75])]
    out = {
        "format": SPLIT_FORMAT,
        "source_csv": os.path.basename(CSV_PATH),
        "split_seed": SPLIT_SEED,
        "train_entropy": TRAIN_ENTROPY,
        "eval_entropy": EVAL_ENTROPY,
        "eval_seed_base": EVAL_SEED_BASE,
        "roles": {
            "producer_pool_rule": f"top {N_PRODUCER_POOL} by Generator Capacity, ties -> lower id",
            "producer_profile": "load = GC + CL, PV = GG",
            "consumer_profile": "load = GC + CL, PV = 0",
            "producer_capacity_kw_min_max": [
                float(min(data["capacity"][c - 1] for c in prod)),
                float(max(data["capacity"][c - 1] for c in prod))],
        },
        "producers": {"train": prod_train, "test": prod_test},
        "consumers": {"train": cons_train, "test": cons_test},
        "days": {"n_days": N_DAYS, "start": START_DATE.isoformat(),
                 "test_rule": f"d % {TEST_DAY_MOD} == {TEST_DAY_REM}",
                 "n_train": int(len(train_days())), "n_test": int(len(test_days()))},
        "scales": {
            "rule": ("train homes x train days, slots with |net| > 0.01 kWh; "
                     "Q edges = [0, p25, p50, p75, 20]; NN norm = p99. "
                     "surplus: producers' max(GG - (GC+CL), 0); "
                     "deficit: producers' and consumers' max(load - PV, 0) pooled"),
            "n_surplus_samples": int(surplus.size),
            "n_deficit_samples": int(deficit.size),
            "q_surplus_edges": [0.0] + q_s + [Q_TOP_EDGE],
            "q_deficit_edges": [0.0] + q_d + [Q_TOP_EDGE],
            "nn_surplus_norm": float(np.percentile(surplus, 99)),
            "nn_deficit_norm": float(np.percentile(deficit, 99)),
        },
    }
    out["split_id"] = _split_id(out)
    return out


_SPLIT: Optional[Dict] = None


def write_split(split: Dict, path: str = SPLIT_PATH) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def load_split() -> Dict:
    """split.json'u okur (yoksa veriden hesaplayıp yazar). Süreç başına bir kez."""
    global _SPLIT
    if _SPLIT is None:
        if os.path.exists(SPLIT_PATH):
            with open(SPLIT_PATH, encoding="utf-8") as f:
                s = json.load(f)
            if s.get("format") != SPLIT_FORMAT:
                raise ValueError(f"{SPLIT_PATH}: beklenmeyen format {s.get('format')}")
            if s.get("split_id") != _split_id(s):
                raise ValueError(f"{SPLIT_PATH}: split_id içerikle uyuşmuyor (dosya elle değişmiş?)")
        else:
            s = compute_split()
            write_split(s)
        _SPLIT = s
    return _SPLIT


def ausgrid_scales() -> Dict[str, object]:
    """SimConfig alanları için ölçek sabitleri (split.json'dan)."""
    sc = load_split()["scales"]
    return {
        "q_surplus_edges": tuple(float(x) for x in sc["q_surplus_edges"]),
        "q_deficit_edges": tuple(float(x) for x in sc["q_deficit_edges"]),
        "nn_surplus_norm": float(sc["nn_surplus_norm"]),
        "nn_deficit_norm": float(sc["nn_deficit_norm"]),
    }


def split_identity() -> Dict[str, object]:
    """Sonuç JSON'larına yazılacak kısa bölünme kimliği."""
    s = load_split()
    return {"split_id": s["split_id"], "split_seed": s["split_seed"],
            "source_csv": s["source_csv"], "test_day_rule": s["days"]["test_rule"],
            "eval_seed_base": s["eval_seed_base"], "n_eval_scenarios": s["days"]["n_test"]}


# ---------------------------------------------------------------------------
# Senaryolar
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    kind: str                       # "train" | "eval"
    seed: int                       # reset(seed=...) değeri
    day: int                        # gün indeksi (0 = 1 Tem 2011)
    producers: Tuple[int, ...]      # müşteri id'leri, ajan sırasıyla (ajan 0..n_p-1)
    consumers: Tuple[int, ...]      # müşteri id'leri (ajan n_p..n_p+n_c-1)

    @property
    def date(self) -> str:
        return (START_DATE + _dt.timedelta(days=self.day)).isoformat()


def _draw_homes(rng: np.random.Generator, pool: Sequence[int], n: int, what: str) -> Tuple[int, ...]:
    if n > len(pool):
        raise ValueError(
            f"Ausgrid: {what} havuzunda {len(pool)} ev var, popülasyon {n} istiyor "
            f"(üretici havuzu eğitim/test = {100 - N_PRODUCER_TEST}/{N_PRODUCER_TEST}, "
            f"tüketici = {200 - N_CONSUMER_TEST}/{N_CONSUMER_TEST})")
    if n == 0:
        return ()
    return tuple(int(c) for c in rng.choice(np.asarray(pool, dtype=np.int64), size=n, replace=False))


def is_eval_seed(seed: int) -> bool:
    return seed >= EVAL_SEED_BASE


def train_scenario(seed: int, n_producers: int, n_consumers: int) -> Scenario:
    """Eğitim senaryosu: (seed, n_p, n_c)'nin deterministik fonksiyonu."""
    seed = int(seed)
    if not 0 <= seed < EVAL_SEED_BASE:
        raise ValueError(f"Ausgrid eğitim seed'i [0, {EVAL_SEED_BASE}) aralığında olmalı: {seed}")
    s = load_split()
    rng = np.random.default_rng(np.random.SeedSequence([TRAIN_ENTROPY, seed]))
    days = train_days()
    day = int(days[rng.integers(len(days))])
    prods = _draw_homes(rng, s["producers"]["train"], n_producers, "eğitim üretici")
    cons = _draw_homes(rng, s["consumers"]["train"], n_consumers, "eğitim tüketici")
    return Scenario("train", seed, day, prods, cons)


_EVAL_CACHE: Dict[Tuple[int, int], List[Scenario]] = {}


def eval_scenarios(n_producers: int, n_consumers: int) -> List[Scenario]:
    """73 değerlendirme senaryosu (test günleri sırasıyla); yalnızca (n_p, n_c)'ye bağlı."""
    key = (int(n_producers), int(n_consumers))
    if key not in _EVAL_CACHE:
        s = load_split()
        out = []
        for k, day in enumerate(test_days()):
            rng = np.random.default_rng(np.random.SeedSequence([EVAL_ENTROPY, k]))
            prods = _draw_homes(rng, s["producers"]["test"], key[0], "test üretici")
            cons = _draw_homes(rng, s["consumers"]["test"], key[1], "test tüketici")
            out.append(Scenario("eval", EVAL_SEED_BASE + k, int(day), prods, cons))
        _EVAL_CACHE[key] = out
    return list(_EVAL_CACHE[key])


def eval_seeds(n_producers: int, n_consumers: int) -> List[int]:
    return [sc.seed for sc in eval_scenarios(n_producers, n_consumers)]


def scenario_for_seed(seed: int, n_producers: int, n_consumers: int) -> Scenario:
    """reset(seed) → senaryo: ayrılmış aralık değerlendirme, altı eğitim."""
    seed = int(seed)
    if is_eval_seed(seed):
        k = seed - EVAL_SEED_BASE
        ev = eval_scenarios(n_producers, n_consumers)
        if not 0 <= k < len(ev):
            raise ValueError(f"Ausgrid: değerlendirme seed'i aralık dışında: {seed} "
                             f"(geçerli: {EVAL_SEED_BASE}..{EVAL_SEED_BASE + len(ev) - 1})")
        return ev[k]
    return train_scenario(seed, n_producers, n_consumers)


def assert_train_seeds_ok(first_seed: int, n_episodes: int) -> None:
    """Eğitim seed'leri first_seed .. first_seed+n_episodes-1 ayrılmış aralığa girmemeli."""
    last = int(first_seed) + int(n_episodes) - 1
    assert int(first_seed) >= 0 and last < EVAL_SEED_BASE, (
        f"Eğitim seed'leri [{first_seed}, {last}] değerlendirme aralığıyla "
        f"(>= {EVAL_SEED_BASE}) çakışıyor")


# ---------------------------------------------------------------------------
# ProfileManager arayüzü
# ---------------------------------------------------------------------------

class AusgridProfileManager(ProfileManager):
    """ProfileManager ile aynı arayüz; profiller seed'in senaryosundan (gerçek veri).

    pv_profiles / load_profiles listeleri, get_* metodları ve talep / arz çarpanı
    anlamı (tüketicilerin yükü, üreticilerin PV'si) ProfileManager'dan miras alınır.
    Üst sınıfın kurduğu self.rng hiç kullanılmaz (veri gürültüsüzdür).
    """

    def __init__(
        self,
        n_producers: int,
        n_consumers: int,
        seed: int = 42,
        demand_multiplier: float = 1.0,
        supply_multiplier: float = 1.0,
    ):
        self.scenario = scenario_for_seed(seed, n_producers, n_consumers)
        super().__init__(n_producers, n_consumers, seed=seed,
                         demand_multiplier=demand_multiplier,
                         supply_multiplier=supply_multiplier)

    def _generate_all_profiles(self):
        data = load_data()
        sc = self.scenario
        for i, cid in enumerate(sc.producers):
            self.load_profiles[i] = home_load(data, cid, sc.day)
            self.pv_profiles[i] = home_pv(data, cid, sc.day)
        for j, cid in enumerate(sc.consumers):
            i = self.n_producers + j
            self.load_profiles[i] = home_load(data, cid, sc.day)
            self.pv_profiles[i] = np.zeros(N_SLOTS)       # tüketicinin PV'si yok sayılır

    def customer_id(self, agent_id: int) -> int:
        sc = self.scenario
        if agent_id < self.n_producers:
            return sc.producers[agent_id]
        return sc.consumers[agent_id - self.n_producers]

    def summary(self) -> str:
        sc = self.scenario
        lines = [f"=== Profile Summary (Ausgrid, {sc.kind}, gün {sc.day} = {sc.date}) ==="]
        for i in range(self.n_agents):
            agent_type = "Producer" if i < self.n_producers else "Consumer"
            total_pv = self.pv_profiles[i].sum()      # değerler zaten slot başına kWh
            total_load = self.load_profiles[i].sum()
            lines.append(
                f"  Agent {i} ({agent_type}, ev {self.customer_id(i)}): "
                f"PV={total_pv:.1f}kWh, Load={total_load:.1f}kWh, "
                f"Net={total_pv - total_load:.1f}kWh"
            )
        return "\n".join(lines)
