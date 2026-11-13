# Ek deneyler — popülasyon boyutu, derin ağ, yakınsama

Hakem ve danışman istekleri için ek deneylerin başka bir makinede
(Ubuntu, 32 iş parçacığı) nasıl çalıştırılacağı. Her iş ayrı bir süreçtir ve
tek CPU iş parçacığı kullanır; GPU gerekmez.

## Deneyler

Popülasyonlar (üretici + tüketici): **4+4, 10+10, 20+20, 10+20**.

- **A — popülasyon boyutu:** her popülasyon için üç strateji: rastgele
  (zero-intelligence) baseline (200 fiyat çekilişi ortalaması), Q-Learning
  (3000 episode) ve REINFORCE (7000 episode, 2 gizli katman × 64). Seed 1..5 → 20 iş.
- **B — derin ağ:** her popülasyon için yalnızca REINFORCE, 7000 episode,
  3 ve 4 gizli katman (genişlik 64, geri kalan her şey aynı). Seed 1..3 → 24 iş.
  2 katmanlı referans, A'nın aynı seed'li REINFORCE sonucudur (tekrar koşulmaz).
- **C — yakınsama kontrolü:** 20+20, Q-Learning 6000 ve REINFORCE 14000 episode
  (2× bütçe, varsayılan ağ). Seed 1..3 → 3 iş; A'nın 20+20 sonuçlarıyla karşılaştırılır.

"Seed" repo genelindeki anlamıyla kullanılır (`SimConfig.seed`): eğitim günleri
`seed + episode`, değerlendirme günü `seed`; ayrıca her yöntemden önce
`seed_everything(seed)` çağrılır. Bu yüzden bir yöntemin sonucu aynı işte başka
hangi yöntemlerin koştuğuna bağlı değildir. Toplam 47 iş.

## Kurulum

```bash
git clone https://github.com/efeylm/energy-trading.git
cd energy-trading
git checkout revize          # zaten klonluysa: git fetch && git checkout revize && git pull
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tüm komutlar repo kök dizininden çalıştırılır.

## 1. Smoke testi (1 dakikadan kısa)

```bash
python -m src.run_experiments --smoke --workers 4
python -m src.summarize_experiments results_smoke
```

Tüm işler "BİTTİ" olmalı ve sonunda "Tüm işler başarılı." yazmalı.
`results_smoke/` git'e eklenmez.

## 2. Tam çalıştırma

Önce listeye bakın (hiçbir şey çalıştırmaz):

```bash
python -m src.run_experiments --dry-run
```

Sonra arka planda başlatın (oturum kapansa da devam eder):

```bash
nohup python -m src.run_experiments --workers 16 > run_experiments.out 2>&1 &
```

veya `tmux new -s deney` içinde `python -m src.run_experiments --workers 16`
(ayrılmak: `Ctrl-b d`, geri dönmek: `tmux attach -t deney`).

Yalnızca bir kısmı çalıştırmak için: `--only A`, `--only B,C` (tekrarlanabilir).
Kaba tahmin (M1 tek çekirdek ölçümünden ekstrapolasyon): toplam ≈ 24 CPU-saat,
16 işçiyle ≈ 2 saat; en uzun tek iş C (20+20, ≈ 1.5 saat).

## İlerlemeyi izleme

```bash
tail -f run_experiments.out                  # iş başlama/bitiş satırları, n/47, geçen süre
tail -f results/logs/A_p20c20_seed1.log      # tek bir işin %10'luk ilerleme satırları
ls results/*.json | wc -l                    # biten iş sayısı
```

## Kesilirse / devam ettirme

Aynı komutu tekrar çalıştırın. JSON'u zaten olan işler atlanır; yarım kalan
işler baştan koşar (JSON atomik yazılır, yarım dosya kalmaz). Başarısız iş
diğerlerini durdurmaz; sonda listelenir, log'u `results/logs/<iş>.log` içindedir.
Bir işi yeniden koşmak için JSON'unu silip komutu tekrar çalıştırın.

## Özet

```bash
python -m src.summarize_experiments results
```

Üretilenler: `results/summary.txt` (tablolar), `results/summary.csv`,
`results/fig_eta_by_population.png`, `results/fig_nn_eta_by_depth.png`,
`results/fig_training_curves.png`.

## Geri gönderme

```bash
git add results/*.json results/summary.txt results/summary.csv results/*.png
git add -f results/logs/*.log        # isteğe bağlı: .gitignore *.log'u yok sayıyor
git commit -m "Ek deney sonuçları (A, B, C)"
git push origin revize
```
