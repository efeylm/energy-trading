"""
Rastgele "Zero-Intelligence" Baseline Ajan — Öğrenme Yapmaz.

Gode & Sunder (1993) "zero-intelligence" tüccarları: ajan hiçbir strateji
izlemez; her adımda tüm açığı/fazlası için TEK bir toplu emir verir ve fiyatı
[FiT, ToU] aralığından bağımsız ve düzgün (uniform) dağılımla çeker.

Rol, BaselineAgent ile birebir aynı kurala göre belirlenir (Eq. 1):
  net_load >  0.01 → ALICI  : Order(fiyat ~ U[FiT, ToU], miktar = net,
                               use_flat_price=True)  — ünite bloklarına bölünmez
  net_load < -0.01 → SATICI : Order(fiyat ~ U[FiT, ToU], miktar = |net|)
  aksi hâlde       → emir yok

Öğrenen ajanlar rastgele tüccarları geçebiliyorsa gerçekten bir şey
öğrenmişlerdir; bu ajan makaledeki kontrol grubudur.

RASTGELELİK: Her ajanın kendine ait bir numpy Generator'ı vardır; tohumu
np.random.SeedSequence([price_seed, agent_id]) ile deterministik olarak
türetilir. Böylece
  * aynı (price_seed, agent_id) → aynı fiyat dizisi (tekrarlanabilirlik),
  * ajanların çağrılma sırası sonucu etkilemez,
  * GLOBAL np.random / random / torch RNG'leri HİÇ tüketilmez.
Son madde zorunludur: comparison.py baseline'dan önce seed_everything()
çağırır ve Q-Learning / REINFORCE sonuçları global RNG durumuna bağlıdır;
baseline global RNG'den sayı çekseydi öğrenen ajanların sonuçları değişirdi.
"""

from __future__ import annotations

import numpy as np

from src.config import SimConfig, AgentMBParams, AgentMCParams
from src.agent import EnergyAgent, Observation, Action
from src.market import Order


class RandomAgent(EnergyAgent):
    """Rastgele (zero-intelligence) baseline ajan — fiyat ~ U[FiT, ToU]."""

    def __init__(
        self,
        agent_id: int,
        agent_type: str,
        mb_params: AgentMBParams,
        mc_params: AgentMCParams,
        config: SimConfig,
        price_seed: int = 0,
    ):
        super().__init__(agent_id, agent_type, mb_params, mc_params, config)
        self.price_seed = price_seed
        # Ajana özel RNG — global RNG'lere dokunulmaz (bkz. modül docstring'i).
        seq = np.random.SeedSequence([int(price_seed), int(agent_id)])
        self.rng = np.random.default_rng(seq)

    # ------------------------------------------------------------------
    # Fiyatlama
    # ------------------------------------------------------------------

    def _random_price(self) -> float:
        """[FiT, ToU] aralığından düzgün dağılımlı fiyat (tarifeler config'den)."""
        return float(self.rng.uniform(self.config.fit_price, self.config.tou_price))

    # ------------------------------------------------------------------
    # Core override
    # ------------------------------------------------------------------

    def decide_action(self, obs: Observation) -> Action:
        """Rastgele fiyatlı tek toplu emir. Öğrenme, Q-table, epsilon yoktur."""
        self._current_hour = obs.hour
        net = obs.inflexible_load
        order = None

        if net > 0.01:
            # Alıcı: tüm açık için TEK toplu emir (MB ünite blokları uygulanmaz)
            order = Order(
                agent_id=self.agent_id,
                price=self._random_price(),
                quantity=net,
                is_buy=True,
                is_emergency=False,
                use_flat_price=True,
            )
        elif net < -0.01:
            # Satıcı: tüm fazla için tek toplu emir
            order = Order(
                agent_id=self.agent_id,
                price=self._random_price(),
                quantity=abs(net),
                is_buy=False,
                is_emergency=False,
            )

        return Action(order=order)

    # ------------------------------------------------------------------
    # Öğrenme yok — bu metodlar intentionally boş
    # ------------------------------------------------------------------

    def update_q(self, *args, **kwargs):
        """Rastgele ajan öğrenmez — bu metod kasıtlı olarak boştur."""
        pass

    def end_episode(self, *args, **kwargs):
        """Rastgele ajan için epsilon decay veya tablo güncellemesi yoktur."""
        pass

    def __repr__(self) -> str:
        return (
            f"RandomAgent(id={self.agent_id}, type={self.agent_type}, "
            f"price_seed={self.price_seed}, cost={self.total_cost:.3f}$)"
        )
