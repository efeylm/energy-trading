# shock_lab — şoklara yeniden eğitim olmadan uyum

Kişisel yan deney (makaleden bağımsız). Soru: ajanlar arz/talep şoklarına
**yeniden eğitilmeden** doğru tepki vermeyi öğrenebilir mi?

Bugünkü durum: tabular Q-Learning'in durumu (saat kovası × kendi miktar kovası)
piyasa hakkında hiçbir bilgi içermez; bu yüzden bir şoka ancak ~2000 episode
yeniden eğitimle uyum sağlar. Burada ajanlar **rastgele şoklarla** (domain
randomization) eğitilir, sonra politikalar **dondurulur** ve eğitimde hiç
görülmemiş şok büyüklükleri dahil çeşitli şoklarda test edilir.

`src/` hiç değiştirilmez; simülatör (piyasa, profiller, ajanlar, ödüller, metrikler)
içe aktarılarak kullanılır. Tüm yeni kod bu klasördedir.

## Rejimler

Bir rejim `(d, s)` çiftidir:

- `d`: her **tüketici** ajanın (indeks ≥ üretici sayısı) yük profili × d
- `s`: her **üretici** ajanın PV üretimi × s

Çarpanlar günün profilleri src'deki gibi üretildikten **sonra** (RNG çekiliş sırası
değişmeden) ve ajanların gördüğü ilk gözlemden **önce** uygulanır.

- Eğitim ızgarası: d, s ∈ {0.5, 1, 2} → 9 rejim. Her episode'da 1/3 olasılıkla
  normal (1,1), aksi hâlde diğer 8'den biri eşit olasılıkla
  (`--train-regimes`, `--normal-prob`, `--regime-weights` ile değiştirilebilir).
- Test: 9 görülen rejim + görülmemiş büyüklükler: (0.75,1), (1.5,1), (1,0.75),
  (1,1.5), (1.5,0.75), (0.75,1.5), (1.5,1.5), (0.75,0.75) (`--test-regimes`).

## Varyantlar

| Varyant | Yöntem | Durum / girdi | Eğitim rejimleri |
|---|---|---|---|
| `q_normal` | Q-Learning | src'deki gibi (saat × miktar) | yalnızca normal (bugünkü durum, referans) |
| `q_blind` | Q-Learning | src'deki gibi | rastgele |
| `q_price` | Q-Learning | + önceki slotun ort. takas fiyatı kovası (3 kova, eşikler 0.135 / 0.185 $/kWh) | rastgele |
| `q_oracle` | Q-Learning | + rejimin kendisi (d seviyesi × s seviyesi, 3×3) | rastgele |
| `nn_normal` | REINFORCE | src'deki gibi (saat, miktar, son fiyat + son 12 fiyat) | yalnızca normal |
| `nn_price` | REINFORCE | src'deki gibi (fiyatı zaten görüyor) | rastgele |
| `nn_oracle` | REINFORCE | + log2(d), log2(s) | rastgele |

Tüm Q varyantları: 15 fiyat seviyesi (FiT..ToU), α = 0.1, γ = 0.95, ödül src ile aynı.
NN: src ağı (12 fiyatlık zaman serisi kodlayıcı, 2×64 gizli katman, Adam 5e-4).

Ayrıntılar:

- **`q_price` fiyat kovası:** önceki slotta takas olduysa onun ortalama fiyatı;
  olmadıysa (ortalama 0) **bugün görülen son pozitif takas fiyatı**; bugün henüz
  hiç takas olmadıysa (gece, PV yok) **orta kova**. Önceki günden bilgi taşınmaz.
- **`q_oracle` seviye eşlemesi:** değer, eğitim seviyelerinden (0.5, 1, 2) **log
  ölçeğinde en yakınına** eşlenir. Sonuç: 0.75 → normal, 1.5 → yüksek (log2 1.5 =
  0.585 > 0.5). Matematiksel olarak "her rejim için ayrı bir Q-tablosu" ile aynıdır.
- **Keşif takvimi:** src'nin ε·0.985 azalması 40000 episode için çok hızlı
  (~200 episode'da tabana iner). Varsayılan: taban eğitimin %30'unda erişilir,
  `decay = (taban/başlangıç)^(1/(0.3·N))` (Q: ε 1.0 → 0.05; NN: σ 0.25 → 0.05).
  `--explore-frac`, `--eps-decay`, `--sigma-decay` ile değiştirilebilir.
- **Uzman (specialist) referansı:** her test rejimi × seed için sadece o rejimde
  sıfırdan eğitilmiş `q_blind` tipi Q seti (varsayılan 5000 episode). "Tam yeniden
  eğitilmiş bir ajan bu rejimde hangi fiyata ulaşır?" sorusunun cevabı.

## Kurulum (başka makinede)

```bash
git clone https://github.com/efeylm/energy-trading.git
cd energy-trading
git checkout revize
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU sürümü yeterli
```

**GPU işe yaramaz:** ağlar çok küçük (adım başına tek örnek), her iş tek CPU
iş parçacığıyla çalışır. Önemli olan **CPU çekirdek sayısıdır**; `--workers`
fiziksel çekirdek sayısı civarında seçin (32 iş parçacıklı makinede 16).

Tüm komutlar **repo kök dizininden** çalıştırılır.

## Çalıştırma

```bash
# 0) hızlı doğruluk kontrolü (~15 s): rejim uygulaması + src ile birebir eşdeğerlik + süre
python -m shock_lab.selfcheck

# 1) smoke testi (~2 dk, 4 worker): küçük bütçe → shock_lab/runs_smoke, shock_lab/eval_smoke
python -m shock_lab.run_all --smoke --workers 4

# 2) planı görmek (hiçbir şey çalıştırmaz)
python -m shock_lab.run_all --dry-run

# 3) tam çalıştırma (oturum kapansa da devam eder)
nohup python -m shock_lab.run_all --workers 16 > shock_lab/run_all.out 2>&1 &
tail -f shock_lab/run_all.out
tail -f shock_lab/runs/logs/nn_price_p10c10_seed1.log      # tek işin ilerlemesi (~%1'de bir satır)
```

Varsayılan plan: 7 varyant × seed 1, 2, 3 = 21 iş, 10 üretici + 10 tüketici, her
iş 40000 episode; sonra değerlendirme (`--specialists` dahil). Yarıda kesilirse aynı
komut tekrar çalıştırılır: `run.json`'u olan işler atlanır, değerlendirme de
`shock_lab/eval/cache/` önbelleğinden devam eder.

Seçenekler: `--seeds 1,2,3`, `--variants q_price,nn_price`, `--episodes-q`,
`--episodes-nn`, `--producers/--consumers`, `--no-eval`, `--no-specialists`,
`--specialist-episodes`, `--test-days`, `--timeline-days 20,20,20`,
`--train-args "--normal-prob 0.5"` (train.py'ye aynen geçer).

Tek parça çalıştırmak için:

```bash
python -m shock_lab.train --variant q_price --seed 1 --producers 10 --consumers 10 \
    --episodes 40000 --out shock_lab/runs
python -m shock_lab.evaluate --runs shock_lab/runs --out shock_lab/eval --specialists --workers 16
python -m shock_lab.evaluate --out shock_lab/eval --plot-only     # yalnızca özet + grafikleri yeniden üret
```

## Beklenen süreler (10+10)

M1 (tek çekirdek) ölçümü: Q ≈ 26–34 ms/episode, NN ≈ 230–280 ms/episode.

| İş | 40000 episode |
|---|---|
| Q varyantı | ~20–25 dk |
| NN varyantı | ~2.5–3 saat |
| Değerlendirme (koşu başına) | Q ~20 s, NN ~1.5 dk |
| Uzmanlar (17 rejim × 3 seed × 5000 ep) | ~2.5 CPU-saat → 16 worker ile ~10 dk |

32 iş parçacıklı makinede `--workers 16` ile tüm iş paralel başlar; duvar saati
≈ en uzun NN işi + değerlendirme ≈ **3–5 saat** (çekirdek başına hız M1'den
yavaşsa üst sınıra yakın).

## Çıktılar

```
shock_lab/runs/<varyant>_p10c10_seed<S>/
    run.json            yapılandırma, argv, makine, süreler, eğitim özeti, enerji dengesi
    policy.npz|.pt      Q-tabloları / NN state_dict'leri
    train_metrics.npz   episode başına: d, s, fiyat, P2P, E_max, η, şebeke alım/satım,
                        satıcı geliri, alıcı maliyeti, sistem ödülü, ε/σ
shock_lab/runs/logs/<iş>.log     (repo .gitignore'u *.log'u yok sayar → push edilmez)
shock_lab/eval/
    eval_results.json   tüm gün bazlı sonuçlar + toplamlar + skorlar
    eval_summary.txt    tablolar (aşağıda)
    fig_price_by_regime.png, fig_eta_by_regime.png
    fig_timeline_price_<şok>.png, fig_timeline_p2p_<şok>.png
```

`runs/` ve `eval/` git'e eklenebilir (`runs_smoke/`, `eval_smoke/`, `eval/cache/`
yok sayılır).

Test günleri tüm eğitim günlerinden ayrıktır: rejim testi seed `10_000_000 + k`
(k = 0..29), zaman çizelgesi seed `20_000_000 + gün`; tüm varyantlar aynı günlerde
test edilir. Değerlendirmede öğrenme kapalıdır (Q: ε = 0, NN: ağ ortalaması).

## Özeti okumak (`eval_summary.txt`)

1. **Metrik tabloları** (fiyat, η = P2P/E_max, alıcı maliyeti, satıcı geliri):
   satır = rejim, sütun = varyant (+ uzman). Hücre: seed'ler üzerinden ort ± std
   (her seed = 30 test gününün ortalaması). Görülen / görülmemiş rejimler ayrı blokta.
   `*_normal` varyantları için (1,1) dışındaki her rejim görülmemiştir.
2. **SHOCK RESPONSE:** her rejimde fiyatın normal rejime göre değişimi (Δ).
   Uzman sütunu "doğru" tepkidir. İyi bir varyantın Δ'sı uzmanınkine yakın olmalı;
   `q_normal`'ın Δ'sı yalnızca piyasa mekaniğinden gelir (politika değişmez).
3. **SCORE:** uzmana uzaklık — `price_mae` = |fiyat − uzman fiyatı| ortalaması,
   `delta_mae` = |Δ − uzman Δ| ortalaması; görülen ve görülmemiş rejimler ayrı.
   Düşük = tam yeniden eğitilmiş ajana yakın.
4. **TIMELINE:** 60 günlük senaryoda (1–20 normal, 21–40 şok, 41–60 normal) faz
   ortalamaları. Dondurulmuş politika şok başladığı gün tepki vermeli ve bittiği
   gün eski düzeye dönmelidir (öğrenme yok, gecikme yok).
5. **ENERGY BALANCE CHECK:** her eğitim ve test gününde
   |D − P2P − şebeke alımı| ve |S − P2P − şebeke satımı| ≤ 1e-6·max(1, ·);
   ihlal sayısı 0 olmalı.

Not: Gece PV olmadığı için takas yoktur; NN'nin 12 slotluk fiyat geçmişi her
sabah 06:00'dan önce FiT ile dolar. Bu yüzden önceki günün (başka rejimin)
fiyatları ertesi günün kararlarını etkilemez; yine de her test bloğu başında
geçmiş sıfırlanır.
