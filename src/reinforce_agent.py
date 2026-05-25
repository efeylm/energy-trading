"""
REINFORCE (Direct Policy Gradient) Agent — Faz 3.

Hocanın spesifikasyonu:
  State  : (günün saati, arz miktarı) — 2 boyutlu sürekli uzay
           hour_norm   = obs.hour / 47.0           ∈ [0, 1]
           supply_norm = clip(|net_load| / 10, 1)  ∈ [0, 1]

  Network: PolicyNet — Linear(2→64) → ReLU → Linear(64→64) → ReLU → Linear(64→1) → Sigmoid
           W = θ  (hocanın "Theta Weight" ifadesi = ağın tüm parametreleri)

  Action : a ∈ [0, 1]  (Sigmoid çıkışı + eğitimde Gaussian keşif gürültüsü)
           price = FiT + a × (ToU − FiT)   → tamamen sürekli fiyat

  Reward : Faz 2 reward fonksiyonu değişmez (agent.py::get_reward kullanılır):
           Satıcı: sell_income + curtailed × rate + β × price_premium
           Alıcı : −net_cost  (alıcı öğrenmez, MB eğrisi kullanılır)

  Update : Doğrudan Gradient (Direct Policy Optimization — Seçenek 1):
           G_t  = Σ_{k=t}^{T} γ^k × r_k    normalize edilmiş indirimli getiri
           loss = −mean(G_t × a_t)           episode SONUNDA tek gradient adımı
           θ   ← θ − lr × ∇_θ loss

Sadece satıcılar öğrenir (net_load < 0).
Alıcılar MB eğrisini kullanır — Faz 1/2 mantığı korunur.

Q-Learning ile karşılaştırma notu:
  Q-table    : 3×4×15 = 180 değer  →  12 ayrık durum, 15 ayrık aksiyon
  PolicyNet  : ~8 500 parametre   →  sürekli durum, sürekli aksiyon
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional

from src.config import SimConfig, AgentMBParams, AgentMCParams
from src.agent import EnergyAgent, Observation, Action
from src.market import Order


# ---------------------------------------------------------------------------
# Politika Ağı — θ = W
# ---------------------------------------------------------------------------

class PolicyNet(nn.Module):
    """2 giriş → gizli katmanlar → 1 çıkış (Sigmoid) politika ağı.

    Mimari:
        Linear(2 → hidden) → ReLU
        Linear(hidden → hidden) → ReLU
        Linear(hidden → 1) → Sigmoid   →  a ∈ [0, 1]

    Ağırlıklar W = θ: hocanın Theta Weight ifadesi.
    """

    def __init__(self, state_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),   # a ∈ [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # shape: [1] veya [batch, 1]


# ---------------------------------------------------------------------------
# REINFORCE Agent
# ---------------------------------------------------------------------------

class REINFORCEAgent(EnergyAgent):
    """Direct Policy Gradient enerji ticareti ajansı — Faz 3.

    EnergyAgent'tan miras alınan metodlar (DEĞİŞMEZ):
        compute_mb(), compute_mc(), compute_uili_price()
        process_trade_results(), get_reward()   ← Faz 2 reward aynen kullanılır

    Override edilen:
        decide_action() → PolicyNet forward + Gaussian noise (eğitim sırasında)

    Yeni metodlar:
        _encode_state()   → 2-boyutlu normalize tensor
        store_reward()    → episode buffer'a reward ekle (satıcı adımlarında)
        update_policy()   → G_t hesapla, gradient at, ağı güncelle
        end_episode()     → update_policy + sigma decay + buffer temizle
    """

    def __init__(
        self,
        agent_id: int,
        agent_type: str,
        mb_params: AgentMBParams,
        mc_params: AgentMCParams,
        config: SimConfig,
        lr: float = 5e-4,
        gamma: float = 0.95,
        sigma: float = 0.25,
        sigma_min: float = 0.05,
        sigma_decay: float = 0.993,
        hidden: int = 64,
    ):
        super().__init__(agent_id, agent_type, mb_params, mc_params, config)

        self.lr          = lr
        self.gamma       = gamma
        self.sigma       = sigma
        self.sigma_min   = sigma_min
        self.sigma_decay = sigma_decay

        self.fit_price = config.fit_price
        self.tou_price = config.tou_price

        # Politika ağı ve optimizer
        self.policy_net = PolicyNet(state_dim=2, hidden=hidden)
        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Episode buffer: her satıcı adımı için (a_tensor, reward)
        # _episode_a: gradient graph'ı canlı tutan tensor listesi
        # _episode_rewards: float listesi (senkronize)
        self._episode_a: List[torch.Tensor] = []
        self._episode_rewards: List[float]  = []
        self._last_was_seller: bool         = False

        # Eğitim modu:
        #   True  → Gaussian gürültü eklenir (keşif / exploration)
        #   False → deterministik a_base kullanılır (değerlendirme)
        self.training_mode: bool = True

        # İstatistikler — QLearningAgent ile aynı yapı
        self.episode_rewards: List[float] = []
        self._episode_reward_acc: float   = 0.0

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------

    def _encode_state(self, obs: Observation) -> torch.Tensor:
        """Gözlemi 2-boyutlu normalize tensora çevirir.

        hour_norm   = obs.hour / 47.0               ∈ [0, 1]
        supply_norm = clip(|inflexible_load| / 10)  ∈ [0, 1]

        supply_norm: negatif net_load → satıcı fazlası kWh, 10 kWh'da normalize
        """
        hour_norm   = obs.hour / 47.0
        surplus     = max(0.0, -obs.inflexible_load)  # satıcı: net_load < 0
        supply_norm = min(surplus / 10.0, 1.0)
        return torch.tensor([hour_norm, supply_norm], dtype=torch.float32)

    # ------------------------------------------------------------------
    # Policy override
    # ------------------------------------------------------------------

    def decide_action(self, obs: Observation) -> Action:
        """PolicyNet'ten aksiyon üret.

        Eğitim sırasında:  a = clamp(PolicyNet(s) + N(0, σ), 0, 1)
        Değerlendirmede:   a = clamp(PolicyNet(s), 0, 1)   — deterministik

        Gradient graph, _episode_a içinde saklanır ve episode sonunda
        update_policy() tarafından tüketilir.
        """
        self._current_hour = obs.hour
        net = obs.inflexible_load

        if net < -0.01:  # ── SATICI: PolicyNet kullan ──────────────────
            self._last_was_seller = True
            state  = self._encode_state(obs)
            a_base = self.policy_net(state)          # shape [1], requires_grad=True

            if self.training_mode:
                noise = torch.randn_like(a_base) * self.sigma
                a     = (a_base + noise).clamp(0.0, 1.0)   # diferansiyel, grad var
            else:
                a = a_base.clamp(0.0, 1.0)                 # deterministik

            self._episode_a.append(a)  # gradient graph korunuyor

            # Piyasaya gönderilecek fiyat (gradient graph dışında)
            price = self.fit_price + float(a.detach()) * (self.tou_price - self.fit_price)
            price = float(np.clip(price, 0.001, 2.0))

            order = Order(
                agent_id=self.agent_id,
                price=price,
                quantity=abs(net),
                is_buy=False,
                is_emergency=False,
            )

        elif net > 0.01:  # ── ALICI: MB eğrisi (öğrenme yok) ───────────
            self._last_was_seller = False
            bid_price = self.compute_mb(net, hour=obs.hour)
            order = Order(
                agent_id=self.agent_id,
                price=float(np.clip(bid_price, 0.001, 2.0)),
                quantity=net,
                is_buy=True,
                is_emergency=False,
            )

        else:  # ── Nötr ────────────────────────────────────────────────
            self._last_was_seller = False
            order = None

        return Action(order=order)

    # ------------------------------------------------------------------
    # Reward storage
    # ------------------------------------------------------------------

    def store_reward(self, reward: float):
        """Satıcı adımından gelen reward'ı buffer'a ekle.

        Alıcı ve nötr adımlarda _last_was_seller=False olduğundan bu metod
        hiçbir şey yapmaz. Bu sayede _episode_a ve _episode_rewards listeleri
        daima hizalı kalır (her ikisi de yalnızca satıcı adımlarında büyür).
        """
        if self._last_was_seller:
            self._episode_rewards.append(reward)
            self._episode_reward_acc += reward

    # ------------------------------------------------------------------
    # Policy update (episode sonu)
    # ------------------------------------------------------------------

    def update_policy(self):
        """Episode sonu gradient güncellemesi — Direct Policy Gradient.

        Adımlar:
          1. İndirimli getiriler:  G_t = Σ_{k=t}^{T} γ^k × r_k
          2. Normalize et:         G_t ← (G_t - mean) / (std + ε)
          3. Loss:                 loss = −mean(G_t × a_t)
             Yorum: yüksek getirili adımlarda a_t'yi artır,
                    düşük getirili adımlarda a_t'yi azalt.
          4. Gradient:             loss.backward() → optimizer.step()
          5. Gradient clipping:    max_norm = 1.0 (patlamayı önler)
        """
        n = min(len(self._episode_a), len(self._episode_rewards))
        if n == 0:
            return

        # ── Adım 1: İndirimli getiriler ──────────────────────────────
        rewards = self._episode_rewards[:n]
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.tensor(returns, dtype=torch.float32)

        # ── Adım 2: Normalize (varyans azaltma) ───────────────────────
        if n > 1:
            std = returns_t.std()
            if std > 1e-8:
                returns_t = (returns_t - returns_t.mean()) / (std + 1e-8)

        # ── Adım 3 & 4: Loss ve gradient ─────────────────────────────
        # torch.cat: n ayrı [1] tensorı → shape [n], gradient graph'lar birleşir
        a_stack = torch.cat(self._episode_a[:n])
        loss    = -(returns_t * a_stack).mean()

        self.optimizer.zero_grad()
        loss.backward()  # n ayrı forward pass'in gradientları birikmeli hesaplanır

        # ── Adım 5: Gradient clipping ─────────────────────────────────
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        self.optimizer.step()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def end_episode(self):
        """Episode sonu: politikayı güncelle, sigma'yı düşür, buffer'ı temizle."""
        self.update_policy()

        # İstatistik kaydı
        self.episode_rewards.append(self._episode_reward_acc)
        self._episode_reward_acc = 0.0

        # Sigma decay (ε-decay ile aynı mantık: keşiften sömürüye geçiş)
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)

        # Buffer temizle → computation graph serbest bırakılır (bellek)
        self._episode_a       = []
        self._episode_rewards = []
        self._last_was_seller = False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def policy_stats(self) -> dict:
        """Politika ağı için özet istatistikler."""
        params     = list(self.policy_net.parameters())
        param_norm = sum(p.data.norm().item() ** 2 for p in params) ** 0.5
        return {
            "sigma":      self.sigma,
            "param_norm": round(param_norm, 4),
            "n_params":   sum(p.numel() for p in params),
        }

    def __repr__(self) -> str:
        return (
            f"REINFORCEAgent(id={self.agent_id}, type={self.agent_type}, "
            f"σ={self.sigma:.3f}, cost={self.total_cost:.3f}$)"
        )
