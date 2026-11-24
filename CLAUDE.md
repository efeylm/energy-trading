# Proje rehberi (önce bunu oku)

Bu dosya depoya yeni giren bir kişi ya da agent için tek giriş noktasıdır: ne nerede, hangi sayı hangi dosyadan geliyor, ne bitti, ne kaldı. Son güncelleme: 24 Kasım 2026.

## 1. Proje nedir?

P2P (eşler arası) yerel elektrik piyasası simülasyonu ve bunun üzerine yazılmış makale:

> **"Reinforcement Learning Based Pricing Strategies in Local Energy Markets"**
> Dergi: Int. J. Adv. Eng. Pure Sci. (JEPS). Durum: hakem revizyonu. **Düzeltme son tarihi: 1 Aralık 2026.**
> Yazarlar: Hikmet Topak ve ekip; danışmanlar Kamil Erdayandı (sorumlu yazar), Mehmet Kadir Baran.

Simülasyon: günde 48 yarım saatlik dilim, çift taraflı açık artırma (double auction), sabit tarifeler FiT = 0.06 ve ToU = 0.28 $/kWh, talep esnek değil, depolama yok. Üç fiyatlama stratejisi karşılaştırılıyor:

| Strateji | Kod | Girdi | Çıktı |
|---|---|---|---|
| Random baseline (zero-intelligence) | `src/random_agent.py` | yok | U(FiT, ToU) rastgele fiyat |
| Tabular Q-Learning | `src/ql_agent.py` | saat (3 kova) × net miktar (4 kova) = 12 durum | 15 sabit fiyat seviyesi |
| REINFORCE (NN) | `src/reinforce_agent.py` | saat, net miktar, son takas fiyatı + son 12 takas fiyatı | sürekli fiyat |

**Eski kural tabanlı baseline (MB / UILI) makaleden ve koddan tamamen çıkarıldı.** Tek baseline random'dır. Ausgrid verisi simülasyona hiçbir yerden girmiyor (bkz. Bölüm 8).

## 2. Dallar

| Dal | Durum |
|---|---|
| **`jeps-final`** | **Güncel ve tek çalışma dalı.** Makale, bütün deneyler, ablation ve shock_lab burada; eski baseline temizlenmiş. |
| `revize` | `jeps-final`'in atası: makale deneyleri + ablation (temizlik öncesi). Arşiv. |
| `shock-lab` | shock_lab yan deneyi (temizlik öncesi). Arşiv; içeriği `jeps-final`'e birleştirildi. |
| `nn_vs_q` | İlk gönderilen makalenin kodu (eski baseline'lı). Arşiv. |
| `main` | Eski; `jeps-final` henüz buraya alınmadı. |

Uzak depo: `github.com/efeylm/energy-trading`.

## 3. Klasör haritası

```
CLAUDE.md                  bu rehber
EXPERIMENTS.md             makaledeki ana deneylerin (A/B/C) başka makinede nasıl koşulacağı
requirements.txt           numpy, matplotlib, torch (CPU yeterli)

paper/                     MAKALE BELGELERİ
  JEPS_Economy_RL_revised_v4.docx     güncel revize makale (hakem etiketli Word yorumlarıyla)
  JEPS_Response_to_Reviewers.docx     hakemlere yanıt belgesi (Comment -> Response -> Action Taken)
  poster.pdf                          proje posteri
  eski/                               ilk gönderilen makale (docx + pdf) ve revised_v2

src/                       SİMÜLATÖR VE MAKALE DENEYLERİ (Bölüm 5)
results/                   Deney A/B/C sonuçları (makale Bölüm 4.5 ve 4.8)
  referans_4+4_seed1/      4+4, seed 1 referans gününün ayrıntılı dökümü (makale Bölüm 4.1-4.3)
shock_results/             talep şoku, 10+10 (makale Bölüm 4.4, Şekil 1)
supply_shock_results/      arz şoku, 10+10 (makale Bölüm 4.4, Şekil 2)

ablation/                  GİRDİ ABLATION DENEYİ (Hakem 4, Yorum 5) - makaleye henüz eklenmedi
  README.md, results/, report/ablation_rapor.pdf

shock_lab/                 YAN DENEY: şoklara yeniden eğitimsiz uyum (makalede YOK, kişisel)
  README.md, runs/, eval/, report/shock_lab_rapor.pdf

data/ausgrid/              Ausgrid ham verisi (61 MB). Hiçbir kod kullanmıyor (bkz. Bölüm 8).
```

## 4. Makaledeki her tablo ve şekil nereden geliyor?

Bölüm, tablo ve şekil numaraları `paper/JEPS_Economy_RL_revised_v4.docx` içindeki numaralardır.

| Makalede | Kaynak dosya | Üreten komut |
|---|---|---|
| Tablo 4-7 (4+4, seed 1: genel, satıcı geliri, alıcı maliyeti, mutabakat) | `results/referans_4+4_seed1/random_baseline_summary.txt` (random sütunu) ve `overall_summary.txt` (Q-Learning ve NN bölümleri, ajan bazlı) | `python -m src.random_environment --draws 200 --seed 1`, `python -m src.comparison --episodes 3000 --nn_episodes 7000 --seed 1` |
| Tablo 8, Şekil 1 (talep şoku) | `shock_results/shock_summary.txt`, `fig_shock_panels.png` | `python -m src.experiment_shock --shock demand` |
| Tablo 8, Şekil 2 (arz şoku) | `supply_shock_results/supply_shock_summary.txt`, `supply_fig_shock_panels.png` | `python -m src.experiment_shock --shock supply --out supply_shock_results` |
| Şekil 3 (eğitim eğrileri) | `results/fig_training_curves.png` | `python -m src.summarize_experiments results` |
| Tablo 9 (Ausgrid kullanan çalışmalar) | literatür; koddan gelmiyor | - |
| Tablo 10-13, Şekil 4 (Deney A: 4+4, 10+10, 20+20, 10+20; 5 seed) | `results/summary.txt` bölüm 1-2, `results/A_*.json`, `fig_eta_by_population.png` | `python -m src.run_experiments`, sonra `summarize_experiments` |
| Tablo 14, Şekil 5 (Deney B: 3 ve 4 katmanlı NN) | `results/summary.txt` bölüm 3, `results/B_*.json`, `fig_nn_eta_by_depth.png` | aynı |
| Tablo 15 (Deney C: 20+20, 2 kat bütçe) | `results/summary.txt` bölüm 4, `results/C_*.json` | aynı |
| Tablo 16 (çalışma süreleri) | `results/summary.txt` bölüm 5 | aynı |
| Tablo 1-3 (hiperparametreler, ağ mimarisi) | `src/ql_agent.py`, `src/reinforce_agent.py`, `src/config.py` | - |
| Denklem 15-16 (sentetik profiller) | `src/profiles.py` | - |

`results/*.json` dosya adları: `A_p<üretici>c<tüketici>_seed<S>.json`, `B_..._L<katman>_seed<S>.json`, `C_p20c20_x2_seed<S>.json`. Her JSON'da `results.{random,ql,nn}.metrics` altında tablo metrikleri (η = `efficiency_pct`, `p2p_kwh`, `avg_price`, `buyer_cost`, `seller_revenue`, ...) ve `train_curve` vardır. Hepsinin tek tablosu: `results/summary.csv`.

**Dikkat:** `results/referans_4+4_seed1/overall_summary.txt` eski bir koşunun dökümüdür; içindeki "BASELINE" bölümü kaldırılan kural tabanlı baseline'a aittir ve makalede kullanılmaz. Yalnızca "Q-LEARNING" ve "REINFORCE-NN" bölümleri geçerlidir.

## 5. Kod haritası (`src/`)

Çekirdek:
- `config.py` — `SimConfig`: popülasyon, 48 dilim, seed, FiT/ToU, ödül katsayıları, `demand_multiplier` / `supply_multiplier` (şoklar).
- `profiles.py` — `ProfileManager`: seed'li sentetik günlük PV ve yük profilleri (makale Denklem 15-16).
- `market.py` — `PartialMatchDoubleAuction`: fiyat öncelikli, kısmi eşleşmeli çift taraflı açık artırma, orta fiyattan takas.
- `agent.py` — `Observation`, `Action`, `EnergyAgent` temel sınıfı (takas mutabakatı, şebeke hesabı, ödül).
- `environment.py` — `EnergyTradingEnv`: gün döngüsü (gözlem -> emir -> takas -> şebekeyle mutabakat -> ödül).
- `metrics.py` — `MetricsCollector`: P2P/şebeke hacimleri, fiyat, satıcı geliri, alıcı maliyeti.
- `seeding.py` — `seed_everything`.

Stratejiler:
- `random_agent.py`, `random_environment.py` — random baseline; ayrıca `community_balance` (S, D, E_max = Σ min(S_t, D_t)).
- `ql_agent.py`, `ql_environment.py` — tabular Q-Learning (satıcı ve alıcı için ayrı tablo).
- `reinforce_agent.py`, `reinforce_environment.py` — REINFORCE; 12 fiyatlık geçmiş kodlayıcı, 2×64 gizli katman, satıcı ve alıcı için ayrı ağ.

Koşucular:
- `experiment.py` — tek iş (popülasyon × seed × yöntemler) -> bir JSON.
- `run_experiments.py` — Deney A/B/C planının paralel koşucusu (47 iş) -> `results/`.
- `summarize_experiments.py` — `results/summary.txt`, `summary.csv` ve şekiller.
- `experiment_shock.py` — üç fazlı talep/arz şoku (Q-Learning eğitimi sürerken) ve şekilleri.
- `comparison.py` — tek seed'lik üçlü karşılaştırma (Random | QL | NN), ajan bazlı döküm.
- `random_environment.py` (CLI) — random baseline'ın 200 çekilişlik özeti.

Temel büyüklükler: η = P2P / E_max (eşleşme verimliliği); "seed" = eğitim günleri `seed + episode`, değerlendirme günü `seed`.

## 6. Ek deney paketleri

Her ikisi de `src/` sınıflarını miras alır, `src/`'yi değiştirmez; ayrıntı kendi README'lerinde.

### `ablation/` — girdi ablation (Hakem 4, Yorum 5)
Soru: NN ile Q arasındaki fark algoritmadan mı, NN'nin fiyat girdisinden mi?
- Varyantlar: `q_base`, `q_price3`, `q_price5`, `q_price3_x2`, `q_price5_x2`, `nn_full`, `nn_nohist`, `nn_noprice` (tanımlar `ablation/variants.py`).
- Koşu: 4+4 ve 10+10, 5 seed, makaledeki bütçeler. Sonuç: `ablation/results/ablation_summary.txt`, şekiller `fig_ablation_*.png`, Türkçe rapor `ablation/report/ablation_rapor.pdf`.
- Bulgu: fiyatsız NN (`nn_noprice`, Q ile aynı bilgi) verimliliği koruyor (10+10: %99.90; tam girdili %99.53); Q'ya göre fark duruyor (10+10: +3.35 ± 1.67 puan; makaledeki +2.98). Q'ya fiyat eklemek verimliliği 5-15 puan düşürüyor, 2 kat eğitim düzeltmiyor.
- `q_base` ve `nn_full`, `results/A_*.json` ile birebir aynı sonucu verir (yeniden üretim kontrolü özet dosyasında).

### `shock_lab/` — şoklara yeniden eğitimsiz uyum (makalede yok)
Soru: rastgele rejimlerle (talep çarpanı d, arz çarpanı s) eğitilip dondurulan ajan, şoka doğru tepki verir mi?
- Varyantlar: `q_normal`, `q_blind`, `q_price`, `q_oracle`, `nn_normal`, `nn_price`, `nn_oracle`; 10+10, 3 seed, 40000 bölüm.
- Sonuç: `shock_lab/eval/eval_summary.txt`, şekiller `shock_lab/eval/fig_*.png`, kayıtlı politikalar `shock_lab/runs/`, Türkçe rapor `shock_lab/report/shock_lab_rapor.pdf`.
- Bulgu: rejimi durumuna ekleyen Q (`q_oracle`) yeniden eğitilmiş uzmana çok yakın; NN rejim bilgisine rağmen fiyatını pek değiştirmiyor.

## 7. Nasıl çalıştırılır?

Bütün komutlar depo kök dizininden, `python -m ...` biçiminde çalışır.

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# hızlı doğruluk kontrolleri (saniyeler)
python -m ablation.selfcheck
python -m shock_lab.selfcheck

# planı görmek (hiçbir şey koşmaz)
python -m src.run_experiments --dry-run
python -m ablation.run_all --dry-run
python -m shock_lab.run_all --dry-run

# özet ve şekilleri mevcut sonuçlardan yeniden üretmek (eğitim yok)
python -m src.summarize_experiments results
python -m ablation.summarize ablation/results
python -m shock_lab.evaluate --out shock_lab/eval --plot-only
```

Tam eğitimler uzak makinede koşulur (bkz. Bölüm 9). GPU gerekmez; önemli olan CPU çekirdek sayısıdır (her iş tek iş parçacığı). Yaklaşık süreler: Deney A/B/C toplam birkaç saat; ablation ~25 dk; shock_lab ~1 saat 45 dk (21 işçiyle).

## 8. Ausgrid konusu (sık sorulan)

Ausgrid verisi simülasyonda **kullanılmıyor**. Profiller sentetik (Denklem 15-16, `src/profiles.py`); Ausgrid'den yalnızca 30 dakikalık çözünürlük, günlük eğri şekli ve (Qiu ve ark. üzerinden) 4+4 kurulumu örnek alındı. Makale bunu Bölüm 3.9'da açıkça söylüyor. Gerçek ölçümlerin tek doğrudan kullanımı eski baseline'ın marginal benefit katsayılarıydı; o baseline ile birlikte kaldırıldı. `data/ausgrid/` yalnızca, istenirse sentetik profilleri gerçek ortalama eğriyle karşılaştıran bir doğrulama şekli çizebilmek için saklanıyor.

## 9. Çalışma düzeni ve kurallar

- **Makale düzenleme kuralı:** yeni eklenen her metin kırmızı yazılır ve "Hakem-N, yorum M (…)" biçiminde bir Word yorumu ile hangi hakem yorumuna karşılık eklendiği belirtilir. **Mevcut Word yorumları silinmez** (hocalar okuyor). Hakem etiketleri: Hakem-2 (14 madde), Hakem-3 (Ausgrid karşılaştırması), Hakem-4 (deney tasarımı; 10 madde).
- **docx düzenleme:** dosya açılıp `word/document.xml` üzerinde çalışılır; hocalar belgeyi Google Docs'ta da düzenliyor, bu yüzden depodaki kopya ile onların kopyası farklı olabilir. Düzenlemeden önce güncel kopya istenmeli.
- **Tam eğitimler bu Mac'te koşulmaz;** uzak makinede (`engin`, Tailscale, Ubuntu, çok çekirdekli) koşulur. Mac'te yalnızca küçük testler.
- **Uzak makine GitHub'a push edemez.** Sonuçlar orada commit edilir, Mac'te `git pull engin:energy-trading <dal>` ile çekilir ve GitHub'a Mac'ten pushlanır.
- **Commit ve push yalnızca kullanıcı söyleyince yapılır.**
- Sayısal not: Mac ve sunucu arasında Q-Learning sonuçlarında ~1e-14 düzeyinde kayan nokta farkı olabilir; aynı makinede sonuçlar birebir tekrarlanır.
- Eski sonuç JSON'larının `sim_config` alanında kaldırılmış anahtarlar (`producer_mb`, `lambda_uili`, ...) durur; hiçbir kod bunları okumaz.

## 10. Durum: ne bitti, ne kaldı?

Biten:
- Revize makale v4: random baseline, Deney A/B/C, Bölüm 4.4 (talep ve arz şoku), fizibilite sınırı ve muhasebe özdeşlikleri (Bölüm 3.11), zorunlu üç atıf ([26]-[28]) ve [29].
- Hakemlere yanıt belgesi (ilk tam sürüm).
- Ablation deneyi koşuldu ve raporlandı; hocaların onayı bekleniyor.
- Kod temizliği: eski baseline ve kullanılmayan kod çıkarıldı; sonuçların birebir aynı kaldığı doğrulandı.

Kalan (1 Aralık 2026'ya kadar):
1. **Ablation'ı makaleye eklemek:** Bölüm 4.8'e paragraf + tablo + şekil; Bölüm 3.4, 4.9 ve V'teki "identical-input ablation yapılmadı / gelecek çalışma" cümlelerini sonuçla değiştirmek (aksiyon uzayı farkı sınır olarak kalır).
2. **Yanıt belgesini güncellemek:** Hakem 4, Yorum 5 ("could not be completed" -> yapıldı ve sonucu); ayrıca belge Bölüm 4.4 eklenmeden önceki numaralarla yazıldı, bölüm/tablo/şekil numaraları v4'e göre düzeltilmeli.
3. İsteğe bağlı: Ausgrid doğrulama şekli (Bölüm 8); `overall_summary.txt`'yi yeni `src.comparison` ile yeniden üretmek (eski baseline bölümü kalksın diye; sunucuda koşulmalı).
4. `jeps-final`'i `main`'e almak (karar verilmedi).
