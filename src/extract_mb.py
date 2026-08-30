"""
Ausgrid tüketim verisinden saatlik Marginal Benefit (MB) parametrelerini çıkartır.

ÖNEMLİ — Tek Kaynak İlkesi (single source of truth):
    MB eğrisinin tavanı ve tabanı ARTIK BURADA HARD-CODED DEĞİL.
    Her ikisi de src/config.py'deki tarife değerlerinden okunur:

        max_price = SimConfig.tou_price   (varsayılan 0.28 $/kWh)
        min_price = SimConfig.fit_price   (varsayılan 0.06 $/kWh)

    Gerekçe: alıcının ödemeye razı olduğu en yüksek fiyat, şebekeden alım
    fiyatı olan ToU'dur (daha fazlasını ödemek yerine şebekeden alır);
    en düşük fiyat ise satıcının şebekeye ihracat taban fiyatı FiT'tir.
    Bu iki değer config'de değişirse MB eğrisi de otomatik takip eder.

    NOT: Ajan tarafında (src/agent.py) alpha/beta değerleri her çalıştırmada
    config'den + Q_max'ten YENİDEN türetilir. Dolayısıyla FiT/ToU değiştiğinde
    bu script'i tekrar çalıştırmak ZORUNLU DEĞİLDİR; JSON'daki alpha/beta
    yalnızca bilgilendirme amaçlıdır. Bu script sadece Ausgrid verisi veya
    seçilen müşteriler değiştiğinde yeniden çalıştırılır.
"""

import pandas as pd
import numpy as np
import json
import os

from src.config import SimConfig

# MB eğrisi türetilirken kullanılan alt sınır: Q_max sıfıra yakınsa
# beta patlamasın diye uygulanan güvenlik tabanı (kWh).
Q_MAX_FLOOR = 0.1


def extract_mb_parameters(data_path, customer_id, max_price=None, min_price=None):
    """
    Belirli bir müşteri ID'si için MB parametrelerini çıkartır.

    max_price / min_price verilmezse SimConfig'deki ToU / FiT kullanılır.
    """
    cfg = SimConfig()
    if max_price is None:
        max_price = cfg.tou_price
    if min_price is None:
        min_price = cfg.fit_price

    if not os.path.exists(data_path):
        print(f"Hata: {data_path} bulunamadı!")
        return None

    # CSV'yi oku
    df = pd.read_csv(data_path, skiprows=1)

    # Sadece ilgili müşteriyi ve GC kategorisini al
    df_agent = df[(df['Customer'] == customer_id) & (df['Consumption Category'] == 'GC')].copy()

    if df_agent.empty:
        print(f"Uyarı: Customer {customer_id} için veri bulunamadı.")
        return None

    time_cols = [c for c in df_agent.columns if ':' in c]
    df_long = df_agent.melt(id_vars=['date'], value_vars=time_cols, var_name='time', value_name='consumption')

    # Bu müşterinin kendi Q_max profilini çıkar
    q_max_profile = df_long.groupby('time')['consumption'].max()

    mb_params = {}
    for time_str, q_max in q_max_profile.items():
        safe_q_max = max(Q_MAX_FLOOR, q_max)
        alpha = max_price
        # alpha * exp(-beta * safe_q_max) = min_price
        # => beta = -ln(min_price / alpha) / safe_q_max
        beta = -np.log(min_price / max_price) / safe_q_max
        mb_params[time_str] = {
            "alpha": round(alpha, 4),
            "beta": round(beta, 4),
            "Q_max": round(q_max, 4)
        }

    return mb_params


if __name__ == "__main__":
    cfg = SimConfig()
    DATA_FILE = "csv/2011-2012 Solar home electricity data v2.csv"
    os.makedirs("src/data", exist_ok=True)

    print(f"MB egri sinirlari config'den okundu: "
          f"ToU(max)={cfg.tou_price:.4f} $/kWh, FiT(min)={cfg.fit_price:.4f} $/kWh")

    # 4 Tüketici ajan için (A4, A5, A6, A7) Ausgrid'den ilk 4 müşteriyi seçelim
    customers = [1, 2, 3, 4]
    agent_ids = [4, 5, 6, 7]

    first_params = None
    for cust_id, agent_id in zip(customers, agent_ids):
        params = extract_mb_parameters(DATA_FILE, customer_id=cust_id)
        if params:
            if first_params is None:
                first_params = params
            out_file = f"src/data/mb_agent_{agent_id}.json"
            with open(out_file, 'w') as f:
                json.dump(params, f, indent=4)
            print(f"Agent {agent_id} (Customer {cust_id}) MB parametreleri kaydedildi.")

    if first_params:
        out_file = "src/data/mb_hourly_params.json"
        with open(out_file, 'w') as f:
            json.dump(first_params, f, indent=4)
        print("Global mb_hourly_params.json başarıyla kaydedildi.")
