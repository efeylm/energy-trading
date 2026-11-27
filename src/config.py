"""
Simulation configuration and hyperparameters.

Combines parameters from:
- Qiu et al. (IJCAI-21): time structure
- Makale: FiT / ToU tarifeleri, ödül (reward) şekillendirme, grid fallback,
  şok deneyleri için talep / arz çarpanları
"""

from dataclasses import dataclass

# Sentetik profillerin miktar ölçeği için eski sabitler (bkz. SimConfig alanları).
_SYN_EDGES = (0.0, 1.0, 3.0, 6.0, 20.0)
_SYN_NORM = 10.0


@dataclass
class SimConfig:
    """Master simulation configuration."""
    
    # --- Agent population ---
    n_producers: int = 4        # Agents with PV + load
    n_consumers: int = 4        # Agents with only load
    
    # --- Time structure ---
    t_hours: int = 48           # Simulation day length (steps)
    delta_t: float = 0.5        # Time step (hours)
    
    # --- Random seed for reproducibility ---
    seed: int = 1

    # =====================================================================
    # TARİFELER — TÜM PROJENİN TEK KAYNAĞI (single source of truth)
    # =====================================================================
    # Bu iki değer projede fiyatla ilgili HER ŞEYİ belirler:
    #   * QL aksiyon uzayı        : FiT ile ToU arasında 15 ayrık seviye  (Eq. 7)
    #   * NN aksiyon dönüşümü     : p = FiT + a·(ToU − FiT)               (Eq. 8)
    #   * Random baseline         : fiyat ~ U[FiT, ToU]
    #   * Grid fallback           : alıcı ToU'dan alır, satıcı FiT'ten satar
    # Hiçbir yerde hard-coded tarife bırakmayın; buradan okuyun.
    # FiT (Feed-in Tariff): minimum guaranteed price for exported energy ($/kWh)
    # Sellers receive this even if unmatched in the P2P market.
    fit_price: float = 0.06   # ~6 cent/kWh (typical AU/EU FiT)
    # ToU (Time-of-Use): maximum retail grid price a buyer would pay ($/kWh)
    # Sellers can price up to ToU since buyers save vs. grid.
    tou_price: float = 0.28   # ~28 cent/kWh (typical peak ToU rate)

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

    # --- Talep şoku (demand shock) ---
    # YALNIZCA talep şoku deneyi (src/experiment_shock.py) tarafından kullanılır.
    # Tüketici ajanların (indeks >= n_producers) yük profili bu katsayıyla
    # çarpılır; üreticiler etkilenmez. 1.0 (varsayılan) = hiçbir değişiklik yok,
    # tüm mevcut sonuçlar birebir aynıdır.
    demand_multiplier: float = 1.0

    # --- Arz şoku (supply shock) ---
    # YALNIZCA şok deneyi (src/experiment_shock.py --shock supply) kullanır.
    # Üretici ajanların (indeks < n_producers) PV profili bu katsayıyla
    # çarpılır; tüketiciler etkilenmez. 1.0 (varsayılan) = hiçbir değişiklik yok.
    supply_multiplier: float = 1.0

    # =====================================================================
    # PROFİL KAYNAĞI ve MİKTAR ÖLÇEĞİNE BAĞLI SABİTLER
    # =====================================================================
    # "synthetic" (varsayılan): src/profiles.py ProfileManager — eski davranış.
    # "ausgrid"              : src/ausgrid_profiles.py AusgridProfileManager —
    #                          gerçek Ausgrid ev verisi (slot başına kWh).
    # Profil yöneticisi src.profiles.make_profile_manager() ile kurulur.
    profile_source: str = "synthetic"

    # Q-Learning durum kovaları (src/ql_agent.py): satıcı fazlası / alıcı açığı
    # sınırları (kWh), 5 değer → 4 kova. Varsayılanlar eski sabit PV_EDGES /
    # DEMAND_EDGES = [0, 1, 3, 6, 20] ile birebir aynıdır.
    q_surplus_edges: tuple = (0.0, 1.0, 3.0, 6.0, 20.0)
    q_deficit_edges: tuple = (0.0, 1.0, 3.0, 6.0, 20.0)
    # REINFORCE girdi normalizasyonu (src/reinforce_agent.py):
    # supply_norm = min(fazla / nn_surplus_norm, 1), deficit_norm = min(açık / nn_deficit_norm, 1).
    # Varsayılan 10.0 = eski sabit "/ 10.0".
    nn_surplus_norm: float = 10.0
    nn_deficit_norm: float = 10.0

    def __post_init__(self):
        if self.profile_source not in ("synthetic", "ausgrid"):
            raise ValueError(f"profile_source 'synthetic' veya 'ausgrid' olmalı: {self.profile_source!r}")
        if len(self.q_surplus_edges) != 5 or len(self.q_deficit_edges) != 5:
            raise ValueError("q_surplus_edges / q_deficit_edges 5 değer (4 kova) olmalı")
        # Ausgrid modunda ölçek sabitleri hâlâ sentetik varsayılanlardaysa
        # data/ausgrid/split.json'daki (yalnızca eğitim verisinden hesaplanmış)
        # değerlerle doldurulur. Böylece SimConfig(profile_source="ausgrid") kuran
        # her koşucu (ablation/, shock_lab/ dahil) doğru ölçeği kendiliğinden alır.
        # Açıkça verilmiş (varsayılandan farklı) değerlere dokunulmaz;
        # dataclasses.replace() da zaten doldurulmuş değerleri korur.
        if self.profile_source == "ausgrid" and (
            tuple(self.q_surplus_edges) == _SYN_EDGES
            and tuple(self.q_deficit_edges) == _SYN_EDGES
            and self.nn_surplus_norm == _SYN_NORM
            and self.nn_deficit_norm == _SYN_NORM
        ):
            from src.ausgrid_profiles import ausgrid_scales
            for k, v in ausgrid_scales().items():
                setattr(self, k, v)

    @property
    def n_agents(self) -> int:
        return self.n_producers + self.n_consumers