"""
REINFORCE varyantları (torch gerektirir).

nn_full    : src REINFORCEAgent AYNEN (girdi: [saat, miktar, son ort. fiyat]
             + son 12 takas fiyatının TimeSeriesEncoder özeti → 19 boyut).
nn_nohist  : fiyat geçmişi kodlayıcısı YOK; politika başlığı girdisi 3 boyutlu
             anlık durum [saat, miktar, son ort. fiyat].
nn_noprice : HİÇ fiyat bilgisi yok; girdi 2 boyutlu [saat, miktar] — Q-Learning'in
             gördüğü bilginin aynısı (sürekli biçimde).

Girdiler MİMARİ olarak kaldırılır: StatePolicyNet, PolicyNet'in politika başlığıyla
birebir aynı katman yapısına sahiptir (Linear→ReLU ×n_layers → Linear(·,1) → Sigmoid)
ama kodlayıcısı yoktur ve forward(state, price_history) price_history'yi YOK SAYAR.
Gürültü, REINFORCE kaybı, getiri normalizasyonu, gradyan kırpma, Adam (lr), σ takvimi,
episode akışı, ödül — hepsi src REINFORCEAgent'tan miras alınır, değişmez.
Normalizasyon da aynıdır: nn_noprice durumu src _encode_state çıktısının ilk iki
bileşenidir.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import SimConfig
from src.reinforce_agent import REINFORCEAgent
from src.reinforce_environment import REINFORCEEnv

from ablation.q_agents import _agent_type


class StatePolicyNet(nn.Module):
    """PolicyNet'in kodlayıcısız eşi: girdi yalnızca anlık durum."""

    def __init__(self, state_dim: int, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers en az 1 olmalıdır")
        self.state_dim = state_dim
        layers = [nn.Linear(state_dim, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1), nn.Sigmoid()]   # a ∈ [0, 1]
        self.policy_head = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, price_history: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.policy_head(state)   # price_history bilinçli olarak kullanılmaz


class AblationREINFORCEAgent(REINFORCEAgent):
    """REINFORCEAgent + kodlayıcısız ağ (state_dim = 3 → nohist, 2 → noprice).

    src __init__'i 19 girdili ağları kurar (torch RNG'si tüketilir — deterministik);
    burada aynı gizli genişlik / derinlikle kodlayıcısız ağlar ve optimizer'lar
    yeniden kurulur (aynı sırada: satıcı, sonra alıcı; aynı lr).
    """

    def __init__(self, *args, state_dim: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        if state_dim not in (2, 3):
            raise ValueError("state_dim 2 (noprice) veya 3 (nohist) olmalı")
        self.state_dim = state_dim
        hidden = self.policy_net.policy_head[0].out_features
        self.policy_net = StatePolicyNet(state_dim, hidden=hidden, n_layers=self.n_layers)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.buy_policy_net = StatePolicyNet(state_dim, hidden=hidden, n_layers=self.n_layers)
        self.buy_optimizer = optim.Adam(self.buy_policy_net.parameters(), lr=self.lr)

    def _encode_state(self, obs) -> torch.Tensor:
        s = super()._encode_state(obs)          # [hour_n, supply_n, avg_price_n]
        return s if self.state_dim == 3 else s[:2]

    def _encode_buyer_state(self, obs) -> torch.Tensor:
        s = super()._encode_buyer_state(obs)    # [hour_n, deficit_n, avg_price_n]
        return s if self.state_dim == 3 else s[:2]


class AblationNNEnv(REINFORCEEnv):
    """REINFORCEEnv + varyant ajanları.

    Ajanlar kurulumda id sırasıyla önceden oluşturulur. src'de ajanlar ilk reset()'te
    aynı sırayla oluşturulur ve arada torch RNG'sini kullanan başka bir şey yoktur
    (ProfileManager numpy Generator kullanır); bu yüzden ağ başlangıç ağırlıkları
    src ile aynıdır. reset() ajanları ilk gün dahil her gün "mevcut ajan" dalından
    geçirir. state_dim=None → src REINFORCEAgent
    (nn_full). selfcheck bu yolun src run_nn ile bit düzeyinde aynı olduğunu doğrular.
    """

    def __init__(self, config: SimConfig, state_dim: Optional[int] = None, **nn_kwargs):
        super().__init__(config, **nn_kwargs)
        self.variant_state_dim = state_dim
        for i in range(config.n_agents):
            common = dict(
                agent_id=i, agent_type=_agent_type(i, config),
                config=config, lr=self.nn_lr, gamma=self.nn_gamma, sigma=self.nn_sigma,
                sigma_min=self.nn_sigma_min, sigma_decay=self.nn_sigma_decay,
                hidden=self.nn_hidden, seq_len=self.nn_seq_len, d_model=self.nn_d_model,
                n_layers=self.nn_n_layers,
            )
            if state_dim is None:
                self._rf_agents[i] = REINFORCEAgent(**common)
            else:
                self._rf_agents[i] = AblationREINFORCEAgent(state_dim=state_dim, **common)


def head_input_dim(net: nn.Module) -> int:
    return int(net.policy_head[0].in_features)


def n_params(net: nn.Module) -> int:
    return int(sum(p.numel() for p in net.parameters()))
