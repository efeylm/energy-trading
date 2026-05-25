"""
REINFORCE (Direct Policy Gradient) Agent — Faz 3.

Hocanın spesifikasyonu:
  State  : 5 boyutlu sürekli uzay
           hour_norm   = obs.hour / 47.0              ∈ [0, 1]
           supply_norm = clip(|net_load| / 10, 1)     ∈ [0, 1]
           best_bid_n  = clip(obs.best_bid / 0.5, 1)  ∈ [0, 1]  ← YENİ
           best_ask_n  = clip(obs.best_ask / 0.5, 1)  ∈ [0, 1]  ← YENİ
           avg_price_n = clip(obs.avg_price / 0.5, 1) ∈ [0, 1]  ← YENİ

  Network: PolicyNet — Linear(2→64) → ReLU → Linear(64→64) → ReLU → Linear(64→1) → Sigmoid
           W = θ  (hocanın "Theta Weight" ifadesi = ağın tüm parametreleri)

  Action : a ∈ [0, 1]  (Sigmoid çıkışı + eğitimde Gaussian keşif gürültüsü)
           price = FiT + a × (ToU − FiT)   → tamamen sürekli fiyat

  Reward : Faz 2 reward fonksiyonu değişmez (agent.py::get_reward kullanılır):
           Satıcı: sell_income + curtailed × rate + β × price_premium
           Alıcı : −net_cost  (alıcı öğrenmez, MB eğrisi kullanılır)

  Update : Proper REINFORCE (log_prob policy gradient):
           G_t      = Σ_{k=t}^{T} γ^k × r_k           normalize edilmiş getiri
           log_prob = log Normal(a_raw; μ=PolicyNet(s), σ)
           loss     = −mean(G_t × log_prob)             episode SONUNDA güncelleme
           θ        ← θ − lr × ∇_θ loss
           Gradient: ∂log_prob/∂μ = (a_raw − μ) / σ²  → μ üzerinden akar

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

        # Politika ağı ve optimizer (state_dim=5: saat, miktar, bid, ask, avg_price)
        self.policy_net = PolicyNet(state_dim=5, hidden=hidden)
        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Episode buffer — her satıcı adımı için:
        #   _episode_mu  : μ = PolicyNet(s), gradient graph canlı
        #   _episode_raw : ham örnek a_raw = μ + noise, detached (log_prob için)
        #   _episode_rewards: float listesi (ikisiyle senkronize)
        self._episode_mu: List[torch.Tensor]  = []
        self._episode_raw: List[torch.Tensor] = []
        self._episode_rewards: List[float]    = []
        self._last_was_seller: bool           = False

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
        """Gözlemi 5-boyutlu normalize tensora çevirir.

        [0] hour_norm    = obs.hour / 47.0                ∈ [0, 1]
        [1] supply_norm  = clip(|inflexible_load| / 10)   ∈ [0, 1]
        [2] best_bid_n   = clip(obs.best_bid / 0.5)       ∈ [0, 1]  ← piyasa sinyali
        [3] best_ask_n   = clip(obs.best_ask / 0.5)       ∈ [0, 1]  ← piyasa sinyali
        [4] avg_price_n  = clip(obs.last_avg_price / 0.5) ∈ [0, 1]  ← piyasa sinyali

        Normalizer 0.5: FiT=0.06 → 0.12, ToU=0.28 → 0.56 (0.5 iyi bir orta nokta)
        Sıfır değer (saat 0, henüz işlem yok) geçerli girdi — ağ bunu öğrenir.
        """
        hour_norm   = obs.hour / 47.0
        surplus     = max(0.0, -obs.inflexible_load)
        supply_norm = min(surplus / 10.0, 1.0)
        best_bid_n  = min(obs.best_bid / 0.5, 1.0)
        best_ask_n  = min(obs.best_ask / 0.5, 1.0)
        avg_price_n = min(obs.last_avg_price / 0.5, 1.0)
        return torch.tensor(
            [hour_norm, supply_norm, best_bid_n, best_ask_n, avg_price_n],
            dtype=torch.float32,
        )

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
            state  = self._encode_state(obs)          # 5-boyutlu
            a_base = self.policy_net(state)            # μ ∈ [0,1], requires_grad=True

            if self.training_mode:
                noise   = torch.randn_like(a_base) * self.sigma
                a_raw   = a_base + noise               # ham örnek (log_prob için)
                a_acted = a_raw.clamp(0.0, 1.0)       # piyasaya gönderilen aksiyon
            else:
                a_raw   = a_base                       # deterministik: noise yok
                a_acted = a_base.clamp(0.0, 1.0)

            # μ ve ham örnek ayrı saklanır — proper REINFORCE log_prob için
            self._episode_mu.append(a_base)            # grad var (ağa bağlı)
            self._episode_raw.append(a_raw.detach())   # grad yok (sadece değer)

            price = self.fit_price + float(a_acted.detach()) * (self.tou_price - self.fit_price)
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
        """Episode sonu gradient güncellemesi — Proper REINFORCE (log_prob).

        Adımlar:
          1. İndirimli getiriler:  G_t = Σ_{k=t}^{T} γ^k × r_k
          2. Normalize et:         G_t ← (G_t - mean) / (std + ε)
          3. Log-prob hesapla:     π(a|s) = Normal(μ=PolicyNet(s), σ)
                                   log_prob = log π(a_raw; μ, σ)
                                   ∂log_prob/∂μ = (a_raw - μ) / σ²
          4. Loss:                 loss = −mean(G_t × log_prob)
          5. Gradient:             loss.backward() → optimizer.step()
          6. Gradient clipping:    max_norm = 1.0
        """
        n = min(len(self._episode_mu), len(self._episode_rewards))
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

        # ── Adım 3 & 4 & 5: Proper REINFORCE loss ────────────────────
        # μ: PolicyNet çıktısı, grad var
        # a_raw: ham örnek (clamp öncesi), detached — log_prob değeri için
        # Gradient sadece μ üzerinden akar: ∂log_prob/∂μ = (a_raw - μ) / σ²
        mu_stack  = torch.cat(self._episode_mu[:n])    # shape [n], grad var
        raw_stack = torch.cat(self._episode_raw[:n])   # shape [n], grad yok

        dist      = torch.distributions.Normal(mu_stack, self.sigma)
        log_probs = dist.log_prob(raw_stack)            # shape [n]
        loss      = -(returns_t * log_probs).mean()

        self.optimizer.zero_grad()
        loss.backward()

        # ── Adım 6: Gradient clipping ─────────────────────────────────
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
        self._episode_mu      = []
        self._episode_raw     = []
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
