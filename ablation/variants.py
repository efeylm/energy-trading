"""
Varyant tanımları ve ortak yardımcılar (torch İÇE AKTARILMAZ).

Her varyant: aile (q | nn), girdiler, durum/girdi boyutu, eğitim bütçesi çarpanı.
Bütçe: Q varyantları --episodes-q × ep_mult, NN varyantları --episodes-nn.
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from typing import Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def chdir_repo_root() -> None:
    """Her giriş noktası çalışma dizinini repo köküne sabitler (göreli çıktı
    yolları, ör. ablation/results, ve `src` paketinin içe aktarımı için)."""
    os.chdir(REPO_ROOT)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)


# Sıra = tablo / grafik sırası. Referanslar her ailenin ilk satırıdır.
VARIANTS: "OrderedDict[str, Dict]" = OrderedDict([
    ("q_base", dict(
        family="q", ref="q_base", price_buckets=0, ep_mult=1,
        label="Q base",
        inputs="hour bucket (3) × own net-quantity bucket (4); no price information",
        description="Unmodified src tabular Q-Learning (reference).")),
    ("q_price3", dict(
        family="q", ref="q_base", price_buckets=3, ep_mult=1,
        label="Q +price(3)",
        inputs="q_base state × bucket of obs.last_avg_price "
               "(0 = no trade in previous slot, 1..3 = equal-width over [FiT, ToU])",
        description="Q-Learning whose state also contains the previous-slot clearing price "
                    "(the same observation field the NN receives), 3 price buckets + no-trade.")),
    ("q_price5", dict(
        family="q", ref="q_base", price_buckets=5, ep_mult=1,
        label="Q +price(5)",
        inputs="q_base state × bucket of obs.last_avg_price "
               "(0 = no trade in previous slot, 1..5 = equal-width over [FiT, ToU])",
        description="As q_price3 with 5 equal-width price buckets + no-trade.")),
    ("q_price3_x2", dict(
        family="q", ref="q_base", price_buckets=3, ep_mult=2,
        label="Q +price(3) 2×",
        inputs="as q_price3",
        description="q_price3 with a doubled training budget (budget control for the larger table).")),
    ("q_price5_x2", dict(
        family="q", ref="q_base", price_buckets=5, ep_mult=2,
        label="Q +price(5) 2×",
        inputs="as q_price5",
        description="q_price5 with a doubled training budget (budget control for the larger table).")),
    ("nn_full", dict(
        family="nn", ref="nn_full", state_dim=3, history=True,
        label="NN full",
        inputs="[hour, own quantity, last_avg_price] + last 12 clearing prices "
               "(TimeSeriesEncoder, d_model 16) → 19-dim head input",
        description="Unmodified src REINFORCE agent (reference).")),
    ("nn_nohist", dict(
        family="nn", ref="nn_full", state_dim=3, history=False,
        label="NN −history",
        inputs="[hour, own quantity, last_avg_price] → 3-dim head input (no price-history encoder)",
        description="REINFORCE without the 12-price history window.")),
    ("nn_noprice", dict(
        family="nn", ref="nn_full", state_dim=2, history=False,
        label="NN −price",
        inputs="[hour, own quantity] → 2-dim head input (no price information at all)",
        description="REINFORCE with exactly the information Q-Learning has (no price inputs).")),
])
VARIANT_ORDER: List[str] = list(VARIANTS)
Q_VARIANTS = [v for v in VARIANT_ORDER if VARIANTS[v]["family"] == "q"]
NN_VARIANTS = [v for v in VARIANT_ORDER if VARIANTS[v]["family"] == "nn"]

# Aileler arası eşleştirilmiş karşılaştırmalar (a − b). İlki BAŞLIK karşılaştırmasıdır
# (aynı bilgi: saat + kendi miktarı, fiyat yok). İkincisi makaledeki özgün fark,
# üçüncüsü fiyat bilgisi her iki tarafta da varken (son takas fiyatı) karşılaştırma.
CROSS_PAIRS = [("nn_noprice", "q_base"), ("nn_full", "q_base"), ("nn_nohist", "q_price5")]


def episodes_for(variant: str, episodes_q: int, episodes_nn: int) -> int:
    v = VARIANTS[variant]
    if v["family"] == "q":
        return int(episodes_q * v["ep_mult"])
    return int(episodes_nn)


def job_name(variant: str, producers: int, consumers: int, seed: int) -> str:
    return f"{variant}_p{producers}c{consumers}_seed{seed}"


def parse_variants(text: str) -> List[str]:
    out = [v.strip() for v in text.split(",") if v.strip()]
    bad = [v for v in out if v not in VARIANTS]
    if bad or not out:
        raise ValueError(f"geçersiz varyant(lar): {bad} (geçerli: {', '.join(VARIANT_ORDER)})")
    return out


def fmt_hms(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"
