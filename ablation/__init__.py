"""
Girdi ablasyonu (hakem yorumu: "NN, Q-Learning'in görmediği fiyat bilgisini görüyor").

İki öğreniciye verilen bilgiyi her iki yönde eşitleyen varyantlar:
  NN tarafı : nn_full (src aynen), nn_nohist (12 fiyatlık geçmiş yok),
              nn_noprice (hiç fiyat bilgisi yok = Q-Learning'in bilgisi)
  Q tarafı  : q_base (src aynen), q_price3 / q_price5 (son takas fiyatı kovası),
              q_price3_x2 / q_price5_x2 (2× eğitim bütçesi)

src/ ve results/ DEĞİŞTİRİLMEZ; tüm sınıflar src sınıflarını miras alır.
Ayrıntılar: ablation/README.md
"""
