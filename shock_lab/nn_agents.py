"""
REINFORCE varyantları (torch gerektirir).

nn_normal / nn_price : src REINFORCEAgent AYNEN (girdi: [saat, miktar, son ort. fiyat]
                       + son 12 takas fiyatı kodlayıcısı, 2×64 gizli katman, Adam lr 5e-4).
nn_oracle            : aynı tasarım, anlık duruma iki sürekli girdi eklenir:
                       log2(d), log2(s)  → state_dim 3 → 5.
"""

from __future__ import annotations

from collections import deque
from typing import Dict

import torch
import torch.optim as optim

from src.config import SimConfig
from src.reinforce_agent import REINFORCEAgent, PolicyNet
from src.reinforce_environment import REINFORCEEnv

from shock_lab.common import RegimeMixin, Regime, _agent_type


class OracleREINFORCEAgent(REINFORCEAgent):
    """REINFORCEAgent + rejim girdisi [log2 d, log2 s].

    src'nin __init__'i 3 boyutlu ağları kurar; burada aynı hiperparametrelerle
    5 boyutlu girdili ağlar ve optimizer'lar yeniden kurulur (aynı sırada: satıcı,
    sonra alıcı). Geri kalan her şey (gürültü, REINFORCE kaybı, σ azalması) miras.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = self.policy_net.policy_head[0].out_features
        self.policy_net = PolicyNet(state_dim=5, seq_len=self.seq_len, d_model=self.d_model,
                                    hidden=hidden, n_layers=self.n_layers)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.buy_policy_net = PolicyNet(state_dim=5, seq_len=self.seq_len, d_model=self.d_model,
                                        hidden=hidden, n_layers=self.n_layers)
        self.buy_optimizer = optim.Adam(self.buy_policy_net.parameters(), lr=self.lr)
        self._regime_feat = torch.zeros(2, dtype=torch.float32)

    def start_day(self, regime: Regime) -> None:
        import math
        self._regime_feat = torch.tensor([math.log2(regime[0]), math.log2(regime[1])],
                                         dtype=torch.float32)

    def _encode_state(self, obs):
        return torch.cat([super()._encode_state(obs), self._regime_feat])

    def _encode_buyer_state(self, obs):
        return torch.cat([super()._encode_buyer_state(obs), self._regime_feat])


class ShockNNEnv(RegimeMixin, REINFORCEEnv):
    """REINFORCEEnv + rejim. Ajanlar kurulumda id sırasıyla oluşturulur; ağ
    başlangıç ağırlıkları için torch RNG'si src'nin ilk reset()'teki sırasıyla
    aynı sırada tüketilir (arada torch RNG kullanan başka bir şey yoktur)."""

    def __init__(self, config: SimConfig, oracle: bool = False, **nn_kwargs):
        super().__init__(config, **nn_kwargs)
        cls = OracleREINFORCEAgent if oracle else REINFORCEAgent
        self.oracle = oracle
        for i in range(config.n_agents):
            self._rf_agents[i] = cls(
                agent_id=i, agent_type=_agent_type(i, config),
                config=config, lr=self.nn_lr, gamma=self.nn_gamma, sigma=self.nn_sigma,
                sigma_min=self.nn_sigma_min, sigma_decay=self.nn_sigma_decay,
                hidden=self.nn_hidden, seq_len=self.nn_seq_len, d_model=self.nn_d_model,
                n_layers=self.nn_n_layers,
            )

    def exploration(self) -> float:
        return float(self._rf_agents[0].sigma)

    def set_greedy(self) -> None:
        """Deterministik: ağ ortalaması (gürültü yok), öğrenme kapalı."""
        self.training_enabled = False
        for a in self._rf_agents.values():
            a.training_mode = False

    def reset_price_history(self) -> None:
        for a in self._rf_agents.values():
            a._price_history = deque([a.fit_price] * a.seq_len, maxlen=a.seq_len)

    def policy_state(self) -> Dict:
        return {
            "state_dim": 5 if self.oracle else 3,
            "agents": {i: {"sell": a.policy_net.state_dict(), "buy": a.buy_policy_net.state_dict()}
                       for i, a in self._rf_agents.items()},
        }

    def load_policy_state(self, state: Dict) -> None:
        for i, a in self._rf_agents.items():
            a.policy_net.load_state_dict(state["agents"][i]["sell"])
            a.buy_policy_net.load_state_dict(state["agents"][i]["buy"])
