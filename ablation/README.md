# Girdi ablasyonu — Q-Learning ile REINFORCE'a verilen bilgiyi eşitleme

## Amaç

Hakem itirazı: *NN, Q-Learning'in görmediği fiyat bilgisini görüyor; performans farkının
öğrenme algoritmasından mı yoksa girdilerden mi geldiği belli değil.*

Mevcut girdiler:

| Öğrenici | Gördüğü bilgi |
|---|---|
| Q-Learning (`src/ql_agent.py`) | (saat kovası 3) × (kendi net miktar kovası 4) = 12 durum; fiyat bilgisi **yok** |
| REINFORCE (`src/reinforce_agent.py`) | [saat, kendi miktarı, `obs.last_avg_price`] + son 12 takas fiyatı (`TimeSeriesEncoder`, d_model 16) → 19 boyutlu girdi |

Bu klasör, bilgiyi **iki yönde** eşitleyen varyantları aynı protokolle eğitir/değerlendirir:
NN'den fiyat bilgisini alır, Q-Learning'e fiyat bilgisi verir. Başlık karşılaştırması
**`nn_noprice` vs `q_base`**: ikisi de tam olarak aynı bilgiyi görür (saat + kendi miktarı).

`src/` ve `results/` **değiştirilmez**; tüm sınıflar src sınıflarını miras alır.

## Varyantlar

| Varyant | Aile | Girdiler | Durum / girdi boyutu | Parametre/ağ | Episode |
|---|---|---|---|---|---|
| `q_base` | Q | (saat, miktar) — src aynen | 12 durum/tablo | – | 3000 |
| `q_price3` | Q | (saat, miktar, fiyat kovası), kova ∈ {takas yok, 3 eşit aralık} | 12×4 = 48 | – | 3000 |
| `q_price5` | Q | aynı, 5 eşit aralık + takas yok | 12×6 = 72 | – | 3000 |
| `q_price3_x2` | Q | `q_price3` | 48 | – | 6000 |
| `q_price5_x2` | Q | `q_price5` | 72 | – | 6000 |
| `nn_full` | NN | [saat, miktar, son fiyat] + 12 fiyatlık geçmiş — src aynen | 19 | 5729 | 7000 |
| `nn_nohist` | NN | [saat, miktar, son fiyat] (kodlayıcı yok) | 3 | 4481 | 7000 |
| `nn_noprice` | NN | [saat, miktar] (hiç fiyat yok) | 2 | 4417 | 7000 |

Her ajanın iki tablosu/ağı vardır (satıcı + alıcı); sayılar tablo/ağ başınadır.

### Tam tanımlar

**`obs.last_avg_price` nedir?** Önceki slotun hacim ağırlıklı ortalama takas fiyatı.
Şu durumlarda **0.0**'dır:
- günün ilk slotu: `reset()` her gün yeni bir `PartialMatchDoubleAuction` kurar
  (`src/environment.py:111`), `clearing_results` boştur → `get_market_info()` 0.0 döner (`src/market.py:276`);
- önceki slotta hiç takas olmadıysa: `compute_stats()` `average_price = 0.0` yazar
  (`src/market.py:101-103`), bu değer sonraki gözleme geçer (`src/market.py:280`, `src/environment.py:139`).

Takas olduğunda tüm öğrenen emirleri [FiT, ToU] = [0.06, 0.28] aralığında olduğundan fiyat da bu aralıktadır.

**NN tarafı** (`ablation/nn_agents.py`)
- `nn_full`: src `REINFORCEAgent`'ın kendisi. src, `last_avg_price`'ı state'te `min(p/0.5, 1)` ile
  normalize eder (0 → 0), geçmiş penceresinde 0 yerine FiT koyar (`src/reinforce_agent.py:271,310`).
- `nn_nohist`: `AblationREINFORCEAgent(state_dim=3)`. Politika ağı `StatePolicyNet`: PolicyNet'in
  politika başlığıyla aynı yapı (Linear→ReLU ×2 → Linear(64,1) → Sigmoid), **kodlayıcı yok**,
  `forward(state, price_history)` geçmişi yok sayar. Durum src'deki 3 boyutlu durumun aynısı
  (son fiyat 0 → 0 olarak görülür).
- `nn_noprice`: `AblationREINFORCEAgent(state_dim=2)`; durum src durumunun ilk iki bileşeni
  [saat/47, min(miktar/10, 1)]. Fiyat hiçbir yoldan ağa ulaşmaz.
- Ağlar ve Adam optimizer'ları `super().__init__` sonrasında aynı gizli genişlik/derinlik ve lr ile
  yeniden kurulur. Gürültü, σ takvimi (0.25 → 0.05, ×0.993/episode), REINFORCE kaybı, getiri
  normalizasyonu, gradyan kırpma, ödül, episode akışı **miras** (değişmez).
- Not: src `__init__`'i önce 19 girdili ağları kurduğu için torch RNG'si tüketilir; küçük ağların
  başlangıç ağırlıkları bu yüzden `nn_full`'unkilerle aynı değildir (boyutlar farklı olduğundan zaten
  olamaz). Sonuçlar yine tamamen deterministiktir.

**Q tarafı** (`ablation/q_agents.py`)
- `PriceQLearningAgent`: durum `(saat_kovası, miktar_kovası, fiyat_kovası)`. Fiyat kovası
  **yalnızca `obs.last_avg_price`'tan**:
  - 0 = `last_avg_price <= 0` (önceki slotta takas yok / günün ilk slotu),
  - 1..K = `linspace(FiT, ToU, K+1)` sınırlarıyla eşit genişlikli kovalar; sınırdaki değer üst kovaya,
    ToU son kovaya, aralık dışı değerler kenar kovalara.
  - K=3 sınırlar: 0.06 / 0.1333 / 0.2067 / 0.28; K=5: 0.06 / 0.104 / 0.148 / 0.192 / 0.236 / 0.28.
- Q-tabloları `(3, 4, K+1, 15)`; `decide_action`, `update_q` (bootstrap dahil, bir sonraki
  gözlemin kendi fiyat kovasıyla), `end_episode`, ε takvimi miras. (shock_lab'daki eşik/geri dönüş
  kuralı burada **kullanılmaz**.)
- ε: 1.0'dan ×0.985/episode, taban 0.05 → **~199. episode'da tabana iner** (σ ~230. episode'da).
  Dolayısıyla `_x2` varyantlarının ek 3000 episode'u çoğunlukla sömürü + %5 kalıntı keşiftir;
  "daha fazla keşif" değil "daha fazla güncelleme" kontrolüdür. src davranışı bilerek değiştirilmedi.
- Ortak rastgele sayılar: ε-greedy her kararda aynı sayıda global RNG çağrısı yaptığı için Q
  varyantları aynı seed'de aynı keşif gürültüsünü paylaşır (eşleştirilmiş karşılaştırma için iyi).

**Protokol** (her varyant, `src/experiment.py` ile aynı ve onun kodu çağrılarak):
`seed_everything(seed)` → ortam kur → `src.experiment._run_learner` (`env.train(N)` +
`env.run_evaluation(seed=seed)` + `_metrics_dict` + süreler). Eğitim günleri `seed + episode`,
değerlendirme günü `seed`. E_max src'deki gibi değerlendirme gününün profillerinden. Tek fark:
ilerleme satırı ~%2'de bir (src %10); sarmalayıcı RNG kullanmaz.
Ajanlar ortam kurulumunda önceden oluşturulur; `selfcheck` bu yolun `src.experiment.run_ql` /
`run_nn` ile **bit düzeyinde aynı** (metrikler + eğitim eğrisi) olduğunu doğrular.

## Başka makinede çalıştırma

Tüm komutlar **repo kökünden**; venv'de numpy, matplotlib, torch (CPU) olmalı.

```bash
git pull
source <venv>/bin/activate            # mevcut venv
python -m ablation.selfcheck          # < 1 dk, 5/5 PASS beklenir
python -m ablation.run_all --dry-run --workers 16      # plan + süre tahmini
nohup python -m ablation.run_all --workers 16 > ablation/run_all.out 2>&1 &
```

Seçenekler: `--populations 10+10,4+4`, `--seeds 1,2,3,4,5`, `--variants q_base,nn_noprice`,
`--episodes-q 3000 --episodes-nn 7000`, `--smoke` (4+4, seed 1–2, 20 episode → `ablation/results_smoke/`),
`--no-summary`. JSON'u zaten olan iş atlanır (yarıda kalırsa aynı komutu yeniden çalıştırın).
Tek iş: `python -m ablation.run_job --producers 10 --consumers 10 --seed 1 --variant nn_noprice`.
Özet yeniden: `python -m ablation.summarize ablation/results`.

İzleme:

```bash
tail -f ablation/run_all.out                      # BAŞLA / BİTTİ satırları
tail -f ablation/results/logs/nn_full_p10c10_seed1.log   # ~%2'de bir: ms/ep ve kalan süre
ls ablation/results/*.json | wc -l                # biten iş sayısı (40 hedef)
```

## Beklenen süreler

Yayımlanmış `results/A_p10c10_*` sunucusunda: Q ≈ 20 ms/episode, `nn_full` ≈ 136 ms/episode (10+10).
M1 dizüstünde ölçülen (10+10, Python 3.9, torch 2.8): q_base 32, q_price3 34, q_price5 34,
nn_full 209, nn_nohist 138, nn_noprice 143 ms/episode.

Varsayılan plan (10+10 × 5 seed × 8 varyant = 40 iş), sunucu hızıyla:
`nn_full` işi ≈ 16 dk, `nn_nohist`/`nn_noprice` ≈ 11 dk, Q ≈ 1 dk (`_x2` ≈ 2 dk).
Toplam iş süresi ≈ 3.7 saat (tek çekirdek); **16–21 işçiyle duvar saati ≈ 16–20 dk** (en uzun iş
belirler; fiziksel çekirdek sayısı işçi sayısından azsa orantılı uzar). `--populations 10+10,4+4` ≈ 20 dk.
`--dry-run` güncel tahmini yazar.

## Çıktılar

`ablation/results/`:
- `<varyant>_p<P>c<C>_seed<S>.json` — varyant, girdi açıklaması, mimari (durum sayısı / girdi
  boyutu / parametre), episode, metrikler (src anahtarlarıyla), eğitim/değerlendirme/CPU süresi,
  eğitim eğrisi, Q tanılamaları (fiyat kovası ziyaretleri, ziyaret edilen durumlar, son ε),
  enerji dengesi kontrolü, sim_config, topluluk dengesi (E_max, S, D), sürümler, hostname, zaman, argv.
- `logs/<iş>.log` — ilerleme logları.
- `ablation_summary.txt` / `ablation_summary.json` — İngilizce tablolar (makale için): varyant tablosu,
  aile içi eşleştirilmiş farklar, aileler arası farklar (başlık: `nn_noprice − q_base`; ayrıca
  `nn_full − q_base` özgün fark ve `nn_nohist − q_price5`), seed başına η, Q tanılamaları,
  yayımlanmış `results/A_*` ile yeniden üretim kontrolü.
- `fig_ablation_eta.png`, `fig_ablation_price.png`, `fig_ablation_learning.png`.

## Sonra ne commit edilecek

```bash
git add ablation/results
git add -f ablation/results/logs/*.log     # kök .gitignore '*.log' kuralı logları yok sayar
git add ablation/run_all.out               # isteğe bağlı (.out yok sayılmaz)
git commit -m "Girdi ablasyonu sonuçları"
git push
```

`ablation/results_smoke/` `ablation/.gitignore` ile yok sayılır.
