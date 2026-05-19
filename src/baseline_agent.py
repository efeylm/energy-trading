"""
Kural Tabanlı Baseline Ajan — Öğrenme Yapmaz.

Karar mantığı tamamen deterministik formüllere dayanır:

  Üretim yüksek  (net_load << 0) → Satıcı, fiyatı düşür  (rekabetçi)
  Üretim orta    (net_load ~ 0)  → Sabit referans fiyatı kullan
  Üretim düşük   (net_load > 0)  → Alıcı, referans MB fiyatına yakın teklif ver

Bu ajan Q-Learning eğitimine dahil edilmez; yalnızca
karşılaştırma (benchmark) amacıyla kullanılır.

Referans Fiyat Mantığı
----------------------
- Baz alıcı fiyatı  : MB(q) × hour_factor
- Baz satıcı fiyatı : MC(q) × (2 − hour_factor)

hour_factor:
  07–09 (sabah tepe)  → 1.10   (talep yüksek → alıcılar daha fazla öder)
  10–14 (güneş doruk) → 0.85   (arz yüksek  → satıcılar daha ucuz satar)
  17–20 (akşam tepe)  → 1.15
  diğer              → 1.00
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from src.config import SimConfig, AgentMBParams, AgentMCParams
from src.agent import EnergyAgent, Observation, Action
from src.market import Order


# ---------------------------------------------------------------------------
# Hour factor: fiyatı saate göre ayarlar
# ---------------------------------------------------------------------------

def _hour_factor(hour_step: int) -> float:
    """
    Günün yarım saatlik adımını (0–47) gerçek saate çevirir ve
    bir fiyat çarpanı döner.

    Çarpan > 1 → talep/kıtlık baskısı yüksek (alıcı daha fazla öder, satıcı daha az indirim yapar)
    Çarpan < 1 → arz bolluğu (üreticiler rekabetçi fiyat verir)
    """
    real_hour = (hour_step * 0.5) % 24  # 0.0 – 23.5

    if 7.0 <= real_hour < 9.5:   # Sabah tepe yükü
        return 1.10
    elif 10.0 <= real_hour < 15.0:  # Güneş doruk saatleri: arz fazlası
        return 0.85
    elif 17.0 <= real_hour < 20.5:  # Akşam tepe yükü
        return 1.15
    else:
        return 1.00


class BaselineAgent(EnergyAgent):
    """Kural tabanlı baseline ajan.

    MB/MC eğrilerini kullanır fakat herhangi bir öğrenme mekanizması içermez.
    Fiyat, günün saatine göre deterministik olarak yukarı/aşağı ölçeklenir.

    Parameters
    ----------
    buyer_margin  : MB fiyatının üzerine eklenen sabit marj (oran).
                    Varsayılan 0.0 → tamamen MB fiyatından teklif verir.
    seller_margin : MC fiyatının altına düşülen sabit marj (oran).
                    Varsayılan 0.0 → tamamen MC fiyatından teklif verir.
    """

    def __init__(
        self,
        agent_id: int,
        agent_type: str,
        mb_params: AgentMBParams,
        mc_params: AgentMCParams,
        config: SimConfig,
        buyer_margin: float = 0.0,
        seller_margin: float = 0.0,
    ):
        super().__init__(agent_id, agent_type, mb_params, mc_params, config)
        self.buyer_margin  = buyer_margin
        self.seller_margin = seller_margin

    # ------------------------------------------------------------------
    # Pricing formulas
    # ------------------------------------------------------------------

    def _buyer_price(self, quantity: float, hour: int) -> float:
        """Alıcı teklif fiyatı: MB (Marginal Benefit) eğrisindeki formüle göre."""
        price = self.compute_mb(quantity, hour=hour)
        return float(np.clip(price, 0.001, 2.0))

    def _seller_price(self, quantity: float) -> float:
        """Satıcı istek fiyatı: Use-It-or-Lose-It (UILI) üstel formülüne göre."""
        price = self.compute_uili_price(quantity)
        return float(np.clip(price, 0.001, 2.0))

    # ------------------------------------------------------------------
    # Core override
    # ------------------------------------------------------------------

    def decide_action(self, obs: Observation) -> Action:
        """Deterministik kural tabanlı karar.

        Öğrenme yapılmaz, Q-table yoktur, epsilon yoktur.
        """
        self._current_hour = obs.hour
        net = obs.inflexible_load
        order = None

        if net > 0.01:
            # Alıcı konumunda: MB eğrisi fiyatı
            bid_price = self._buyer_price(net, obs.hour)
            order = Order(
                agent_id=self.agent_id,
                price=bid_price,
                quantity=net,
                is_buy=True,
                is_emergency=False,
            )
        elif net < -0.01:
            # Satıcı konumunda: Use-It-or-Lose-It (UILI) fiyatı
            ask_price = self._seller_price(abs(net))
            order = Order(
                agent_id=self.agent_id,
                price=ask_price,
                quantity=abs(net),
                is_buy=False,
                is_emergency=False,
            )

        return Action(order=order)

    # ------------------------------------------------------------------
    # Öğrenme yok — bu metodlar intentionally boş
    # ------------------------------------------------------------------

    def update_q(self, *args, **kwargs):
        """Baseline ajan öğrenmez — bu metod kasıtlı olarak boştur."""
        pass

    def end_episode(self, *args, **kwargs):
        """Baseline ajan için epsilon decay veya tablo güncellemesi yoktur."""
        pass

    def __repr__(self) -> str:
        return (
            f"BaselineAgent(id={self.agent_id}, type={self.agent_type}, "
            f"cost={self.total_cost:.3f}$)"
        )