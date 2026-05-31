"""
REINFORCE (Policy Gradient) Agent — Faz 3.

Mimari:
  State  : 5 boyutlu anlık gözlem
           [hour_norm, supply_norm, best_bid_n, best_ask_n, avg_price_n]

  History: Son K takas fiyatı — sıralı zaman serisi
           [p_{t-K}, p_{t-K+1}, ..., p_{t-1}]   (en eskiden en yeniye)

  Network: PolicyNet
           ├── TimeSeriesEncoder(history) → d_model boyutlu özet
           └── concat(state, ts_özet)    → 64 → 64 → 1 (Sigmoid) → a ∈ [0,1]

  Action : price = FiT + a × (ToU − FiT)   — tamamen sürekli

  Neden Time Series + Positional Embedding?
    Fiyat geçmişi sıralı bir seri. [2, 7, 4] ile [4, 7, 2] farklı anlam taşır.
    Sırayı korumak için her fiyat değerine pozisyon bilgisi eklenir:
        h_i = price_embedding(p_i) + pos_embedding(i)
    Bu Transformer'daki token + positional embedding'in ta kendisi.
    Ağ böylece "dün gece düşük, sabah yüksek, şimdi düşüyor" gibi kalıpları öğrenir.

  Update : Proper REINFORCE (log_prob):
           G_t      = Σ_{k=t}^{T} γ^k × r_k
           log_prob = log Normal(a_raw; μ=PolicyNet(s, hist), σ)
           loss     = −mean(G_t × log_prob)
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import SimConfig, AgentMBParams, AgentMCParams
from src.agent import EnergyAgent, Observation, Action
from src.market import Order


# ---------------------------------------------------------------------------
# Time Series Encoder — Positional Embedding
# ---------------------------------------------------------------------------

class TimeSeriesEncoder(nn.Module):
    """K uzunluğundaki fiyat serisini sabit boyutlu vektöre dönüştürür.

    Her adım:
        h_i = price_proj(p_i) + pos_embedding(i)
              ─────────────────────────────────
              değer bilgisi + sıra bilgisi  ← ikisini TOPLA (Transformer gibi)

    Çıktı: mean(h_0, ..., h_{K-1})  →  d_model boyutlu özet vektör

    Neden concat değil toplama?
        Transformer makalesi (Vaswani et al. 2017) de böyle yapıyor.
        Toplama, değer ve pozisyon bilgisinin aynı uzayda birleşmesini sağlar.
        Concat olsaydı boyut iki katına çıkardı ve gereksiz parametre eklendi.
    """

    def __init__(self, seq_len: int = 12, d_model: int = 16):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # Her fiyat skaleri → d_model boyutlu vektör
        self.price_proj = nn.Linear(1, d_model)

        # Öğrenilmiş pozisyon embeddings: pos_embedding[i] = "i. pozisyonun imzası"
        # Parametre sayısı: seq_len × d_model = 12 × 16 = 192
        self.pos_embedding = nn.Embedding(seq_len, d_model)

    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        """
        prices: shape [K] — normalize edilmiş fiyat serisi (en eskiden en yeniye)
        return: shape [d_model]
        """
        price_emb = self.price_proj(prices.unsqueeze(-1))   # [K, d_model]

        positions = torch.arange(self.seq_len, dtype=torch.long)
        pos_emb   = self.pos_embedding(positions)            # [K, d_model]

        h = price_emb + pos_emb   # [K, d_model] — sıra bilgisi korunuyor
        return h.mean(dim=0)      # [d_model]    — mean pooling


# ---------------------------------------------------------------------------
# Policy Net — State + Time Series → a ∈ [0,1]
# ---------------------------------------------------------------------------

class PolicyNet(nn.Module):
    """İki girdi alan politika ağı: anlık state + geçmiş fiyat serisi.

    Mimari:
        TimeSeriesEncoder(history [K])       → ts_emb [d_model]
        concat(state [state_dim], ts_emb)    → [state_dim + d_model]
        Linear → ReLU → Linear → ReLU → Linear → Sigmoid  →  a ∈ [0,1]

    W = θ: tüm parametreler (hocanın Theta Weight ifadesi).
    """

    def __init__(
        self,
        state_dim: int = 5,
        seq_len:   int = 12,
        d_model:   int = 16,
        hidden:    int = 64,
    ):
        super().__init__()
        self.ts_encoder  = TimeSeriesEncoder(seq_len=seq_len, d_model=d_model)
        combined_dim     = state_dim + d_model   # 5 + 16 = 21

        self.policy_head = nn.Sequential(
            nn.Linear(combined_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),   # a ∈ [0, 1]
        )

    def forward(
        self,
        state:         torch.Tensor,   # [state_dim]
        price_history: torch.Tensor,   # [seq_len]
    ) -> torch.Tensor:
        ts_emb   = self.ts_encoder(price_history)   # [d_model]
        combined = torch.cat([state, ts_emb])        # [state_dim + d_model]
        return self.policy_head(combined)            # [1]


# ---------------------------------------------------------------------------
# REINFORCE Agent
# ---------------------------------------------------------------------------

class REINFORCEAgent(EnergyAgent):
    """Zaman serisi + positional embedding tabanlı REINFORCE ajanı — Faz 3.

    Miras alınan metodlar (DEĞİŞMEZ):
        compute_mb(), compute_mc(), compute_uili_price()
        process_trade_results(), get_reward()

    Override edilen:
        decide_action() → history güncelle → PolicyNet(state, history) → aksiyon

    Yeni metodlar:
        _encode_state()   → 5-boyutlu anlık state tensoru
        _price_hist_tensor() → normalize edilmiş geçmiş fiyat tensoru
        store_reward()    → episode buffer'a reward ekle
        update_policy()   → proper REINFORCE log_prob loss → backward
        end_episode()     → güncelle + sigma decay + buffer temizle
    """

    def __init__(
        self,
        agent_id:    int,
        agent_type:  str,
        mb_params:   AgentMBParams,
        mc_params:   AgentMCParams,
        config:      SimConfig,
        lr:          float = 5e-4,
        gamma:       float = 0.95,
        sigma:       float = 0.25,
        sigma_min:   float = 0.05,
        sigma_decay: float = 0.993,
        hidden:      int   = 64,
        seq_len:     int   = 12,    # kaç adım geçmişe bakılır (12 = 6 saat)
        d_model:     int   = 16,    # positional embedding boyutu
    ):
        super().__init__(agent_id, agent_type, mb_params, mc_params, config)

        self.lr          = lr
        self.gamma       = gamma
        self.sigma       = sigma
        self.sigma_min   = sigma_min
        self.sigma_decay = sigma_decay
        self.seq_len     = seq_len
        self.d_model     = d_model

        self.fit_price = config.fit_price
        self.tou_price = config.tou_price

        # Fiyat geçmişi — sıralı rolling window (deque otomatik eski kaydı siler)
        # Başlangıçta FiT fiyatıyla dolu: nötr ama sıfırdan iyi bir başlangıç
        self._price_history: deque = deque(
            [config.fit_price] * seq_len, maxlen=seq_len
        )

        # Satıcı politika ağı ve optimizer
        self.policy_net = PolicyNet(
            state_dim=5, seq_len=seq_len, d_model=d_model, hidden=hidden
        )
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Alıcı politika ağı ve optimizer (satıcıyla simetrik, ayrı ağırlıklar)
        self.buy_policy_net = PolicyNet(
            state_dim=5, seq_len=seq_len, d_model=d_model, hidden=hidden
        )
        self.buy_optimizer = optim.Adam(self.buy_policy_net.parameters(), lr=lr)

        # Satıcı episode buffer
        self._episode_mu:  List[torch.Tensor] = []
        self._episode_raw: List[torch.Tensor] = []
        self._episode_rewards: List[float]    = []

        # Alıcı episode buffer
        self._buy_episode_mu:  List[torch.Tensor] = []
        self._buy_episode_raw: List[torch.Tensor] = []
        self._buy_episode_rewards: List[float]    = []

        self._last_was_seller: bool = False
        self._last_was_buyer:  bool = False

        # Eğitim modu: True=noise var, False=deterministik (değerlendirme)
        self.training_mode: bool = True

        # İstatistikler
        self.episode_rewards: List[float] = []
        self._episode_reward_acc: float   = 0.0

    # ------------------------------------------------------------------
    # State & history encoding
    # ------------------------------------------------------------------

    def _encode_state(self, obs: Observation) -> torch.Tensor:
        """Satıcı: 5-boyutlu normalize state — surplus ekseninde."""
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

    def _encode_buyer_state(self, obs: Observation) -> torch.Tensor:
        """Alıcı: 5-boyutlu normalize state — deficit ekseninde."""
        hour_norm    = obs.hour / 47.0
        deficit      = max(0.0, obs.inflexible_load)
        deficit_norm = min(deficit / 10.0, 1.0)
        best_bid_n   = min(obs.best_bid / 0.5, 1.0)
        best_ask_n   = min(obs.best_ask / 0.5, 1.0)
        avg_price_n  = min(obs.last_avg_price / 0.5, 1.0)
        return torch.tensor(
            [hour_norm, deficit_norm, best_bid_n, best_ask_n, avg_price_n],
            dtype=torch.float32,
        )

    def _price_hist_tensor(self) -> torch.Tensor:
        """Geçmiş fiyat serisini normalize edilmiş tensora çevirir.

        [FiT, ToU] aralığına normalize: 0 = FiT, 1 = ToU
        Sıra korunur: index 0 = en eski, index K-1 = en yeni
        """
        prices = list(self._price_history)   # en eskiden en yeniye
        price_range = self.tou_price - self.fit_price + 1e-8
        norm = [(p - self.fit_price) / price_range for p in prices]
        return torch.tensor(norm, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Policy override
    # ------------------------------------------------------------------

    def decide_action(self, obs: Observation) -> Action:
        """History'yi güncelle → PolicyNet(state, history) → aksiyon üret."""
        self._current_hour = obs.hour
        net = obs.inflexible_load

        # Her adımda geçmiş fiyat serisini güncelle
        # Takas olmadığında 0.0 gelir → FiT ile değiştir (negatif normalize önlenir)
        last_price = obs.last_avg_price if obs.last_avg_price > 0.0 else self.fit_price
        self._price_history.append(last_price)

        if net < -0.01:   # ── SATICI ───────────────────────────────────
            self._last_was_seller = True
            state      = self._encode_state(obs)       # [5]
            price_hist = self._price_hist_tensor()     # [seq_len]

            a_base = self.policy_net(state, price_hist)  # μ ∈ [0,1], grad var

            if self.training_mode:
                noise   = torch.randn_like(a_base) * self.sigma
                a_raw   = a_base + noise              # ham örnek (log_prob için)
                a_acted = a_raw.clamp(0.0, 1.0)      # piyasaya giden aksiyon
            else:
                a_raw   = a_base
                a_acted = a_base.clamp(0.0, 1.0)     # deterministik

            self._episode_mu.append(a_base)            # grad var
            self._episode_raw.append(a_raw.detach())   # grad yok

            price = self.fit_price + float(a_acted.detach()) * (self.tou_price - self.fit_price)
            price = float(np.clip(price, 0.001, 2.0))

            order = Order(
                agent_id=self.agent_id,
                price=price,
                quantity=abs(net),
                is_buy=False,
                is_emergency=False,
            )

        elif net > 0.01:  # ── ALICI: buy_policy_net ile öğrenen fiyat ──
            self._last_was_seller = False
            self._last_was_buyer  = True
            state      = self._encode_buyer_state(obs)
            price_hist = self._price_hist_tensor()

            a_base = self.buy_policy_net(state, price_hist)

            if self.training_mode:
                noise   = torch.randn_like(a_base) * self.sigma
                a_raw   = a_base + noise
                a_acted = a_raw.clamp(0.0, 1.0)
            else:
                a_raw   = a_base
                a_acted = a_base.clamp(0.0, 1.0)

            self._buy_episode_mu.append(a_base)
            self._buy_episode_raw.append(a_raw.detach())

            bid_price = self.fit_price + float(a_acted.detach()) * (self.tou_price - self.fit_price)
            bid_price = float(np.clip(bid_price, 0.001, 2.0))

            order = Order(
                agent_id=self.agent_id,
                price=bid_price,
                quantity=net,
                is_buy=True,
                is_emergency=False,
            )

        else:
            self._last_was_seller = False
            self._last_was_buyer  = False
            order = None

        return Action(order=order)

    # ------------------------------------------------------------------
    # Reward storage
    # ------------------------------------------------------------------

    def store_reward(self, reward: float):
        """Rol bazlı reward dağıtımı.

        Satıcı adımı → satıcı buffer'ına (_episode_rewards)
        Alıcı adımı  → alıcı buffer'ına  (_buy_episode_rewards)
        Her iki buffer da update_policy()'de ayrı ayrı işlenir.
        """
        if self._last_was_seller:
            self._episode_rewards.append(reward)
            self._episode_reward_acc += reward
        elif self._last_was_buyer:
            self._buy_episode_rewards.append(reward)
            self._episode_reward_acc += reward

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------

    def _reinforce_update(
        self,
        mu_list:  List[torch.Tensor],
        raw_list: List[torch.Tensor],
        rewards:  List[float],
        net:      nn.Module,
        optimizer: optim.Optimizer,
    ):
        """Tek bir REINFORCE güncelleme adımı — satıcı ve alıcı paylaşır."""
        n = min(len(mu_list), len(rewards))
        if n == 0:
            return

        G_list, G = [], 0.0
        for r in reversed(rewards[:n]):
            G = r + self.gamma * G
            G_list.insert(0, G)
        returns_t = torch.tensor(G_list, dtype=torch.float32)

        if n > 1:
            std = returns_t.std()
            if std > 1e-8:
                returns_t = (returns_t - returns_t.mean()) / (std + 1e-8)

        mu_stack  = torch.cat(mu_list[:n])
        raw_stack = torch.cat(raw_list[:n])

        dist      = torch.distributions.Normal(mu_stack, self.sigma)
        log_probs = dist.log_prob(raw_stack)
        loss      = -(returns_t * log_probs).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

    def update_policy(self):
        """Episode sonu gradient güncellemesi — satıcı ve alıcı için ayrı ayrı."""
        self._reinforce_update(
            self._episode_mu, self._episode_raw, self._episode_rewards,
            self.policy_net, self.optimizer,
        )
        self._reinforce_update(
            self._buy_episode_mu, self._buy_episode_raw, self._buy_episode_rewards,
            self.buy_policy_net, self.buy_optimizer,
        )

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def end_episode(self):
        """Episode sonu: politikayı güncelle, sigma'yı düşür, buffer'ı temizle.

        NOT: _price_history temizlenmez — rolling window episode'lar arası devam
        eder. Ajan önceki günün fiyat örüntülerini "hatırlar".
        """
        self.update_policy()

        self.episode_rewards.append(self._episode_reward_acc)
        self._episode_reward_acc = 0.0

        # Sigma decay: keşiften sömürüye (ε-decay ile aynı mantık)
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)

        # Satıcı buffer temizle
        self._episode_mu      = []
        self._episode_raw     = []
        self._episode_rewards = []

        # Alıcı buffer temizle
        self._buy_episode_mu      = []
        self._buy_episode_raw     = []
        self._buy_episode_rewards = []

        self._last_was_seller = False
        self._last_was_buyer  = False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def policy_stats(self) -> dict:
        params     = list(self.policy_net.parameters())
        param_norm = sum(p.data.norm().item() ** 2 for p in params) ** 0.5
        n_params   = sum(p.numel() for p in params)
        return {
            "sigma":       self.sigma,
            "param_norm":  round(param_norm, 4),
            "n_params":    n_params,
            "seq_len":     self.seq_len,
            "d_model":     self.d_model,
            "history_len": len(self._price_history),
        }

    def __repr__(self) -> str:
        return (
            f"REINFORCEAgent(id={self.agent_id}, type={self.agent_type}, "
            f"σ={self.sigma:.3f}, seq={self.seq_len}, d={self.d_model}, "
            f"cost={self.total_cost:.3f}$)"
        )
