"""
Simulation configuration and hyperparameters.

Combines parameters from:
- Qiu et al. (IJCAI-21): ES operating parameters, time structure
- ADD Bitirme: MB/MC curve parameters, starvation thresholds
"""

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np


@dataclass
class AgentMBParams:
    """Marginal Benefit curve parameters."""
    alpha: float = 0.20
    beta: float = 0.3


@dataclass
class AgentMCParams:
    """Marginal Cost curve parameters."""
    gamma: float = 0.00
    delta: float = 0.04


@dataclass
class SimConfig:
    """Master simulation configuration."""
    
    # --- Agent population ---
    n_producers: int = 4        # Agents with PV + load
    n_consumers: int = 4        # Agents with only load
    
    # --- Time structure ---
    t_hours: int = 48           # Simulation day length (steps)
    delta_t: float = 0.5        # Time step (hours)
    
    # --- MB/MC parameters per agent (yalnızca YEDEK değerler) ---
    # DİKKAT: Bunlar sadece Ausgrid JSON'u (src/data/mb_agent_*.json) bulunamadığında
    # devreye giren yedeklerdir. Normal çalışmada alıcı MB eğrisi
    # EnergyAgent._rebuild_mb_curves() tarafından aşağıdaki tou_price / fit_price
    # ve Ausgrid Q_max'inden türetilir.
    # MC (Marginal Cost) parametreleri şu an fiyatlamada KULLANILMIYOR: satıcı
    # fiyatı Use-It-or-Lose-It eğrisiyle (compute_uili_price) belirleniyor.
    # Producer MB params (when buying)
    producer_mb: AgentMBParams = field(default_factory=lambda: AgentMBParams(
        alpha=0.18, beta=0.25
    ))
    # Producer MC params (when selling)
    producer_mc: AgentMCParams = field(default_factory=lambda: AgentMCParams(
        gamma=0.03, delta=0.08
    ))
    # Consumer MB params (when buying)
    consumer_mb: AgentMBParams = field(default_factory=lambda: AgentMBParams(
        alpha=0.22, beta=0.20
    ))
    # Consumer MC params (rarely used if no battery, but kept for consistency)
    consumer_mc: AgentMCParams = field(default_factory=lambda: AgentMCParams(
        gamma=0.025, delta=0.06
    ))
    
    # --- Market parameters ---
    # Baseline alıcının talebi kaç kWh'lik ünitelere bölünüp ayrı ayrı
    # fiyatlanacağını belirler (MB eğrisi ünite başına değerlendirilir).
    unit_size: float = 0.5              # kWh per discrete tradeable unit

    # NOT: Buradaki initial_shout_margin / alpha / max_auction_rounds ve
    # seller_flat_* / buyer_flat_prob alanları kaldırıldı. İlk üçü artık silinmiş
    # olan IterativeDoubleAuction'a, diğerleri de environment.py'de hiçbir zaman
    # etkisi olmayan (hesaplanıp atılan) flat-block bloğuna aitti.

    # --- Random seed for reproducibility ---
    seed: int = 1

    # --- Data Paths ---
    mb_data_path: str = "src/data/mb_hourly_params.json"

    # =====================================================================
    # TARİFELER — TÜM PROJENİN TEK KAYNAĞI (single source of truth)
    # =====================================================================
    # Bu iki değer projede fiyatla ilgili HER ŞEYİ belirler:
    #   * QL aksiyon uzayı        : FiT ile ToU arasında 15 ayrık seviye  (Eq. 7)
    #   * NN aksiyon dönüşümü     : p = FiT + a·(ToU − FiT)               (Eq. 8)
    #   * UILI satıcı eğrisi      : P(q) = FiT + (ToU − FiT)·e^(−κq)      (Eq. 4)
    #   * Alıcı MB eğrisi         : alpha = ToU, q=Q_max'ta FiT'e iner    (Eq. 3)
    #   * Grid fallback           : alıcı ToU'dan alır, satıcı FiT'ten satar
    #   * src/extract_mb.py       : MB parametre çıkarımının tavan/tabanı
    # Hiçbir yerde hard-coded tarife bırakmayın; buradan okuyun.
    # FiT (Feed-in Tariff): minimum guaranteed price for exported energy ($/kWh)
    # Sellers receive this even if unmatched in the P2P market.
    fit_price: float = 0.06   # ~6 cent/kWh (typical AU/EU FiT)
    # ToU (Time-of-Use): maximum retail grid price a buyer would pay ($/kWh)
    # Sellers can price up to ToU since buyers save vs. grid.
    tou_price: float = 0.28   # ~28 cent/kWh (typical peak ToU rate)

    # --- Use-It-or-Lose-It (UILI) seller pricing ---
    # P(Q) = FiT + (ToU - FiT) * exp(-lambda_uili * Q)
    # At Q=0  → price = ToU  (maximum, no production pressure)
    # At Q→∞  → price → FiT (floor, high production forces cheap bids)
    # Higher lambda_uili = faster price decay with quantity.
    lambda_uili: float = 0.15   # decay rate (1/kWh); tune per scenario

    # --- Reward shaping: fiyat primi katsayısı (β) ---
    # reward += β × matched_qty × (clearing_price - FiT)
    # 0.0 = orijinal formül (değişiklik yok)
    # 0.3 = önerilen başlangıç değeri
    reward_beta: float = 0.3

    # --- Reward shaping: kısıtlanan enerji için ödül oranı ---
    # DİKKAT: Bu parametre YALNIZCA grid_fallback=False iken etkilidir.
    # grid_fallback=True (varsayılan) olduğunda eşleşmeyen enerji şebekeye
    # FiT'ten satılır, kısıtlama (curtailment) oluşmaz ve bu oran hiç kullanılmaz.
    # Dolayısıyla varsayılan kurulumda `--curtail-rate` argümanının etkisi yoktur.
    # Eski davranış : curtailed × FiT  (+$0.06/kWh — "ne olursa şebekeye sat")
    # Yeni varsayılan: curtailed × 0.0  (satamadıysan hiç kazanma → sat!)
    # Negatif değer : curtailed × (-x) → aktif ceza
    reward_curtail_rate: float = -0.03

    # --- Grid fallback ---
    # True  → P2P'de eşleşemeyen alıcı grid'den ToU'dan alır,
    #          P2P'de eşleşemeyen satıcı grid'e FiT'ten satar.
    # False → Kapalı sistem (eski davranış).
    grid_fallback: bool = True

    @property
    def n_agents(self) -> int:
        return self.n_producers + self.n_consumers