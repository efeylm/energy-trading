"""
Kural Tabanlı Baseline Ajan — Öğrenme Yapmaz (makale Bölüm 3.3).

Rol, her adımda net enerji pozisyonuna göre belirlenir (Eq. 1):
  net_load > 0 → ALICI   : fiyat = MB(q)  — üstel Marginal Benefit eğrisi, Eq. (3)
                            ToU tavanıyla sınırlanır (grid'den pahalıya P2P alınmaz)
  net_load < 0 → SATICI  : fiyat = P_UILI(q) — Use-It-or-Lose-It eğrisi, Eq. (4)
                            P(q) = FiT + (ToU − FiT)·e^(−κq)

MB eğrisinin alpha/beta katsayıları Ausgrid verisinden (src/data/mb_agent_*.json)
gelir; tavan/taban ise config'deki ToU/FiT'tir (bkz. src/agent.py _rebuild_mb_curves).
UILI'nin κ katsayısı config.lambda_uili'dir.

Bu ajan öğrenme yapmaz; Q-Learning ve REINFORCE için kontrol grubudur.

NOT: Bu dosyada eskiden bir _hour_factor() yardımcısı ve buyer_margin /
seller_margin parametreleri vardı. Hiçbiri fiyat hesabında kullanılmıyordu
(docstring aksini söylüyordu) — kaldırıldılar.
"""

from __future__ import annotations

import numpy as np

from src.config import SimConfig, AgentMBParams, AgentMCParams
from src.agent import EnergyAgent, Observation, Action
from src.market import Order


class BaselineAgent(EnergyAgent):
    """Kural tabanlı baseline ajan — MB (alıcı) ve UILI (satıcı) eğrileri."""

    def __init__(
        self,
        agent_id: int,
        agent_type: str,
        mb_params: AgentMBParams,
        mc_params: AgentMCParams,
        config: SimConfig,
    ):
        super().__init__(agent_id, agent_type, mb_params, mc_params, config)

    # ------------------------------------------------------------------
    # Pricing formulas
    # ------------------------------------------------------------------

    def _buyer_price(self, quantity: float, hour: int) -> float:
        """Alıcı teklif fiyatı: MB eğrisi, ancak ToU tavanıyla sınırlı.

        Grid fallback varken rasyonel alıcı asla ToU'dan fazla ödemez:
        P2P'de ToU'dan pahalıya almak yerine grid'den alır.
        """
        tou = self.config.tou_price
        price = self.compute_mb(quantity, hour=hour)
        return float(np.clip(price, 0.001, tou))

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