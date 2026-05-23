"""
Demo / scratch script for the PartialMatchDoubleAuction system.

Runs three scenarios that test the five market rules:

  Scenario 1 — Temel eşleşme (2 birimlik satıcı, 2 alıcı farklı fiyat)
  Scenario 2 — Kısmi eşleşme (2.5 birimlik satıcı, eşleşmeyen kısım bekler)
  Scenario 3 — Çoklu satıcıyla tamamlama (alıcı 2 satıcıdan alır)

Her senaryo için alıcı/satıcı grafikleri ve market genel görünümü
``charts/`` klasörüne kaydedilir.

Çalıştırmak için:
    cd /Users/emirhanunsal/Desktop/GitHub/bitirme/energy-trading
    python scratch_plot.py
"""

import os
import sys

# Ensure src/ is on the path when run from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.market import Order, PartialMatchDoubleAuction
from src.visualization import AgentVizInfo, plot_agent_curves

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "charts")


# ------------------------------------------------------------------ #
# Helper                                                               #
# ------------------------------------------------------------------ #

def print_result(label: str, result):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Periyot       : {result.period}")
    print(f"  Toplam Trade  : {len(result.trades)}")
    print(f"  Toplam Hacim  : {result.total_traded_kwh:.3f} kWh")
    print(f"  Ort. Fiyat    : ${result.average_price:.4f}" if result.trades else "  Ort. Fiyat    : N/A")
    print()
    for t in result.trades:
        print(
            f"    Trade  — Alıcı#{t.buyer_id} ↔ Satıcı#{t.seller_id}"
            f"  qty={t.quantity:.3f} kWh"
            f"  price=${t.price:.4f}"
            f"  (bid=${t.buyer_bid:.4f}, ask=${t.seller_ask:.4f})"
        )
    if result.pending_buy_orders:
        print()
        for po in result.pending_buy_orders:
            print(f"    [BEKLEYEN BUY]  Alıcı#{po.agent_id}  kalan={po.remaining_quantity:.3f} kWh  @ ${po.price:.4f}")
    if result.pending_sell_orders:
        print()
        for po in result.pending_sell_orders:
            print(f"    [BEKLEYEN SELL] Satıcı#{po.agent_id}  kalan={po.remaining_quantity:.3f} kWh  @ ${po.price:.4f}")


# ------------------------------------------------------------------ #
# Scenario 1 — Basic: 2-unit seller, 2 buyers at different prices     #
# ------------------------------------------------------------------ #

def run_scenario_1():
    """
    Kural 1–3 testi:
      Satıcı 10  →  2 kWh @ $0.15/kWh
      Alıcı  1   →  1 kWh @ $0.20/kWh   (yüksek teklif, önce sıralanır)
      Alıcı  2   →  1 kWh @ $0.18/kWh   (daha düşük teklif)

    Beklenti:
      - Alıcı 1, Satıcı 10 ile $0.175 üzerinden eşleşir (mid-price)
      - Alıcı 2, Satıcı 10 ile $0.165 üzerinden eşleşir (farklı fiyat!)
      - Satıcının 2 birimi tamamen satılmış olur
    """
    market = PartialMatchDoubleAuction()

    market.submit_order(Order(agent_id=10, price=0.15, quantity=2.0, is_buy=False))
    market.submit_order(Order(agent_id=1,  price=0.20, quantity=1.0, is_buy=True))
    market.submit_order(Order(agent_id=2,  price=0.18, quantity=1.0, is_buy=True))

    result = market.clear_period()
    print_result("Senaryo 1 — Temel Eşleşme (farklı fiyat ayrımı)", result)

    # Visualization
    agents_info = [
        AgentVizInfo(agent_id=1,  is_buyer=True,  submitted_price=0.20, submitted_quantity=1.0,
                     mb_alpha=0.25, mb_beta=0.30, label="Alıcı 1"),
        AgentVizInfo(agent_id=2,  is_buyer=True,  submitted_price=0.18, submitted_quantity=1.0,
                     mb_alpha=0.22, mb_beta=0.28, label="Alıcı 2"),
        AgentVizInfo(agent_id=10, is_buyer=False, submitted_price=0.15, submitted_quantity=2.0,
                     mc_gamma=0.015, mc_delta=0.04, label="Satıcı 10"),
    ]
    paths = plot_agent_curves(agents_info, result, output_dir=os.path.join(OUTPUT_DIR, "senaryo_1"))
    return result, paths


# ------------------------------------------------------------------ #
# Scenario 2 — Partial: 2.5-unit seller, only 2 units matched         #
# ------------------------------------------------------------------ #

def run_scenario_2():
    """
    Kural 4 testi:
      Satıcı 20  →  2.5 kWh @ $0.14/kWh
      Alıcı  3   →  1.5 kWh @ $0.19/kWh
      Alıcı  4   →  0.5 kWh @ $0.17/kWh

    Beklenen toplam talep: 2.0 kWh  <  2.5 kWh arz
    → 0.5 kWh bekleyen satış emri olarak kalır.
    """
    market = PartialMatchDoubleAuction()

    market.submit_order(Order(agent_id=20, price=0.14, quantity=2.5, is_buy=False))
    market.submit_order(Order(agent_id=3,  price=0.19, quantity=1.5, is_buy=True))
    market.submit_order(Order(agent_id=4,  price=0.17, quantity=0.5, is_buy=True))

    result = market.clear_period()
    print_result("Senaryo 2 — Kısmi Eşleşme (satıcı fazlası bekler)", result)

    agents_info = [
        AgentVizInfo(agent_id=3,  is_buyer=True,  submitted_price=0.19, submitted_quantity=1.5,
                     mb_alpha=0.24, mb_beta=0.25, label="Alıcı 3"),
        AgentVizInfo(agent_id=4,  is_buyer=True,  submitted_price=0.17, submitted_quantity=0.5,
                     mb_alpha=0.21, mb_beta=0.30, label="Alıcı 4"),
        AgentVizInfo(agent_id=20, is_buyer=False, submitted_price=0.14, submitted_quantity=2.5,
                     mc_gamma=0.012, mc_delta=0.035, label="Satıcı 20"),
    ]
    paths = plot_agent_curves(agents_info, result, output_dir=os.path.join(OUTPUT_DIR, "senaryo_2"))
    return result, paths


# ------------------------------------------------------------------ #
# Scenario 3 — Cross-seller: buyer matched against two sellers         #
# ------------------------------------------------------------------ #

def run_scenario_3():
    """
    Kural 5 testi:
      Satıcı 30  →  1.0 kWh @ $0.13/kWh   (ucuz, önce eşleşir)
      Satıcı 31  →  1.0 kWh @ $0.16/kWh   (biraz daha pahalı)
      Alıcı  5   →  2.0 kWh @ $0.22/kWh   (büyük talep)
      Alıcı  6   →  0.5 kWh @ $0.18/kWh

    Alıcı 5 → önce Satıcı 30'dan 1.0 kWh, ardından Satıcı 31'den 1.0 kWh alır.
    Her eşleşmede farklı mid-price oluşur.
    """
    market = PartialMatchDoubleAuction()

    market.submit_order(Order(agent_id=30, price=0.13, quantity=1.0, is_buy=False))
    market.submit_order(Order(agent_id=31, price=0.16, quantity=1.0, is_buy=False))
    market.submit_order(Order(agent_id=5,  price=0.22, quantity=2.0, is_buy=True))
    market.submit_order(Order(agent_id=6,  price=0.18, quantity=0.5, is_buy=True))

    result = market.clear_period()
    print_result("Senaryo 3 — Çoklu Satıcıyla Çapraz Eşleşme", result)

    agents_info = [
        AgentVizInfo(agent_id=5,  is_buyer=True,  submitted_price=0.22, submitted_quantity=2.0,
                     mb_alpha=0.28, mb_beta=0.22, label="Alıcı 5"),
        AgentVizInfo(agent_id=6,  is_buyer=True,  submitted_price=0.18, submitted_quantity=0.5,
                     mb_alpha=0.23, mb_beta=0.27, label="Alıcı 6"),
        AgentVizInfo(agent_id=30, is_buyer=False, submitted_price=0.13, submitted_quantity=1.0,
                     mc_gamma=0.010, mc_delta=0.030, label="Satıcı 30"),
        AgentVizInfo(agent_id=31, is_buyer=False, submitted_price=0.16, submitted_quantity=1.0,
                     mc_gamma=0.018, mc_delta=0.045, label="Satıcı 31"),
    ]
    paths = plot_agent_curves(agents_info, result, output_dir=os.path.join(OUTPUT_DIR, "senaryo_3"))
    return result, paths


# ------------------------------------------------------------------ #
# Scenario 4 — Persistence: unmatched order carries to next period    #
# ------------------------------------------------------------------ #

def run_scenario_4():
    """
    Kural 4 süreklilik testi:
      Periyot 0:  Satıcı 40 → 3.0 kWh @ $0.12, Alıcı 7 → 1.0 kWh @ $0.20
                  → 1 kWh eşleşir, Satıcı 40'ın 2.0 kWh'i bekler.
      Periyot 1:  Alıcı 8 → 1.5 kWh @ $0.17 girer.
                  → Bekleyen Satıcı 40 ile 1.5 kWh daha eşleşir.
                  → Satıcı 40'ın 0.5 kWh'i hâlâ bekliyor.
    """
    market = PartialMatchDoubleAuction()

    # --- Period 0 ---
    market.submit_order(Order(agent_id=40, price=0.12, quantity=3.0, is_buy=False))
    market.submit_order(Order(agent_id=7,  price=0.20, quantity=1.0, is_buy=True))

    result0 = market.clear_period()
    print_result("Senaryo 4 — Periyot 0 (bekleyen emir oluşur)", result0)

    agents0 = [
        AgentVizInfo(agent_id=7,  is_buyer=True,  submitted_price=0.20, submitted_quantity=1.0,
                     mb_alpha=0.25, mb_beta=0.28, label="Alıcı 7"),
        AgentVizInfo(agent_id=40, is_buyer=False, submitted_price=0.12, submitted_quantity=3.0,
                     mc_gamma=0.008, mc_delta=0.028, label="Satıcı 40"),
    ]
    plot_agent_curves(agents0, result0, output_dir=os.path.join(OUTPUT_DIR, "senaryo_4", "period_0"))

    # --- Period 1 (Satıcı 40 hâlâ piyasada) ---
    market.submit_order(Order(agent_id=8, price=0.17, quantity=1.5, is_buy=True))

    result1 = market.clear_period()
    print_result("Senaryo 4 — Periyot 1 (bekleyen satıcıyla eşleşme)", result1)

    agents1 = [
        AgentVizInfo(agent_id=8,  is_buyer=True,  submitted_price=0.17, submitted_quantity=1.5,
                     mb_alpha=0.22, mb_beta=0.26, label="Alıcı 8"),
        # Satıcı 40 hâlâ piyasada — remaining_quantity değişti ama görsel için orijinal qty göster
        AgentVizInfo(agent_id=40, is_buyer=False, submitted_price=0.12, submitted_quantity=2.0,
                     mc_gamma=0.008, mc_delta=0.028, label="Satıcı 40 (kalan)"),
    ]
    plot_agent_curves(agents1, result1, output_dir=os.path.join(OUTPUT_DIR, "senaryo_4", "period_1"))

    return result0, result1


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PartialMatchDoubleAuction — Demo Senaryoları")
    print("="*60)

    r1, p1 = run_scenario_1()
    r2, p2 = run_scenario_2()
    r3, p3 = run_scenario_3()
    r4a, r4b = run_scenario_4()

    print("\n" + "="*60)
    print(f"  Tüm grafikler '{OUTPUT_DIR}/' altına kaydedildi.")
    print("="*60 + "\n")
