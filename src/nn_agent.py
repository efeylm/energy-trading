"""
Deep Q-Network (DQN) tabanlı enerji ticareti ajansı — Faz 3.

Doküman Faz 3 spesifikasyonu:
  State  : sürekli → [sin(2π·t/48), cos(2π·t/48), supply_norm, best_bid_norm, last_price_norm]
  Action : a ∈ [0,1] → price = FiT + a·(ToU − FiT)
  Q-fcn  : Q_θ(s,a) — küçük MLP, Q-tablosunun yerini alır
  Update : TD-target = r + γ·max_{a'} Q_target(s', a')
           loss = MSE(Q_θ(s,a), target)

Stabilizasyon:
  - Replay Buffer  : son 10k geçiş, 64'lük rastgele batch
  - Target Network : her TARGET_SYNC_FREQ episode'da senkron
  - Warm-up        : MIN_BUFFER_SIZE dolu olmadan eğitim yok
  - Grad clipping  : max_norm=1.0

Sadece satıcı konumundayken (inflexible_load < 0) öğrenme gerçekleşir.
Alıcı konumundayken MB eğrisi fiyatı kullanılır (Q-Learning ile aynı).

Cihaz önceliği: MPS (M1/M2 Mac) → CUDA → CPU
"""

from __future__ import annotations

import random
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import SimConfig, AgentMBParams, AgentMCParams
from src.agent import EnergyAgent, Observation, Action
from src.market import Order


# ---------------------------------------------------------------------------
# Hiperparametreler
# ---------------------------------------------------------------------------

N_GRID           = 51        # Aksiyon grid çözünürlüğü (0, 0.02, ..., 1.0)
BUFFER_CAPACITY  = 10_000    # Replay buffer maksimum kapasitesi
BATCH_SIZE       = 64        # Mini-batch boyutu
MIN_BUFFER_SIZE  = 256       # Warm-up: eğitim başlamadan önce gereken minimum geçiş
TARGET_SYNC_FREQ = 10        # Kaç episode'da bir target network senkron edilir
SUPPLY_MAX       = 10.0      # Normalizasyon için maks beklenen PV fazlası (kWh)

# State boyutu:
#   [sin_hour, cos_hour, supply_norm, best_bid_norm, last_price_norm]
#    zaman      zaman     üretim baskısı  alıcı teklifi  son takas fiyatı
STATE_DIM = 5

# Aksiyon grid — [0.0, 0.02, 0.04, ..., 1.0]
ACTION_GRID = np.linspace(0.0, 1.0, N_GRID, dtype=np.float32)


def _get_device() -> torch.device:
    """M1 Mac için MPS, GPU varsa CUDA, yoksa CPU seçer."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    Q_θ(s, a) → skaler Q-değeri.

    Girdi  : [state(3-dim), action(1-dim)] → 4 nöron
    Gizli  : 64 → 64 (ReLU aktivasyonu)
    Çıktı  : 1  (Q-değeri)
    """

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        state  : (batch, 3)
        action : (batch, 1)
        returns: (batch,)
        """
        x = torch.cat([state, action], dim=-1)   # (batch, 4)
        return self.net(x).squeeze(-1)            # (batch,)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    (s, a, r, s', done) geçişlerini tutan dairesel tampon.

    push()   : yeni geçişi ekle (en eski otomatik silinir)
    sample() : rastgele mini-batch döner (numpy array'leri)
    """

    def __init__(self, capacity: int = BUFFER_CAPACITY):
        self.buf: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ):
        self.buf.append((state.copy(), float(action), float(reward), next_state.copy(), float(done)))

    def sample(self, batch_size: int = BATCH_SIZE):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s_, d = zip(*batch)
        return (
            np.array(s,  dtype=np.float32),
            np.array(a,  dtype=np.float32),
            np.array(r,  dtype=np.float32),
            np.array(s_, dtype=np.float32),
            np.array(d,  dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buf)


# ---------------------------------------------------------------------------
# Deep Seller Q-Agent
# ---------------------------------------------------------------------------

class DeepSellerQAgent(EnergyAgent):
    """
    Faz 3 DQN ajansı.

    Tabular QLearningAgent'ın NN karşılığı.
    Aynı EnergyAgent arayüzüne sahip → Environment değişmez.

    Satıcıyken  : NN'den gelen aksiyon a → fiyat = FiT + a*(ToU-FiT)
    Alıcıyken   : MB eğrisi fiyatı (Q-Learning ile aynı, öğrenme yok)
    """

    def __init__(
        self,
        agent_id: int,
        agent_type: str,
        mb_params: AgentMBParams,
        mc_params: AgentMCParams,
        config: SimConfig,
        alpha_lr: float = 3e-4,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.985,
    ):
        super().__init__(agent_id, agent_type, mb_params, mc_params, config)

        self.alpha_lr      = alpha_lr
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.fit_price: float = config.fit_price
        self.tou_price: float = config.tou_price

        # Cihaz seçimi
        self.device = _get_device()

        # Ana Q-network ve target network (STATE_DIM = 5)
        self.q_net      = QNetwork(state_dim=STATE_DIM).to(self.device)
        self.target_net = QNetwork(state_dim=STATE_DIM).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=alpha_lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)

        # Geçiş belleği (bir adım)
        self._prev_state: Optional[np.ndarray] = None
        self._prev_action: Optional[float] = None
        self._prev_obs: Optional[Observation] = None

        # İstatistikler
        self.episode_rewards: list[float] = []
        self._episode_reward_acc: float = 0.0
        self._episode_count: int = 0
        self._train_steps: int = 0

        # Aksiyon grid tensor (cihaz üzerinde)
        self._action_grid_t = torch.tensor(
            ACTION_GRID, dtype=torch.float32, device=self.device
        )  # (N_GRID,)

    # ------------------------------------------------------------------
    # State encoding — Faz 3 sürekli state
    # ------------------------------------------------------------------

    def _encode_state(self, obs: Observation) -> np.ndarray:
        """
        Gözlemi 5-boyutlu sürekli state vektörüne dönüştürür.

          [sin_h, cos_h, supply_norm, best_bid_norm, last_price_norm]

        sin/cos       : 23:30→00:00 geçişinde süreklilik (tabular bucket yok)
        supply_norm   : PV fazlası → [0,1] — satıcı baskısı ne kadar?
        best_bid_norm : alıcıların en yüksek teklifi / ToU — biri çok istiyorsa yüksek fiyat sor
        last_price_norm: son takas fiyatı / ToU — piyasa nerede temizlendi?

        İlk adımda best_bid ve last_avg_price 0 olur (henüz ticaret yok),
        NN bunu "piyasa bilgisi yok" olarak yorumlar.
        """
        t = obs.hour  # 0..47
        angle = 2.0 * np.pi * t / 48.0
        sin_h = np.sin(angle)
        cos_h = np.cos(angle)

        surplus = max(0.0, -obs.inflexible_load)
        supply_norm = float(np.clip(surplus / SUPPLY_MAX, 0.0, 1.0))

        # ToU ile normalize et — 0 = hiç teklif yok / ticaret yok, 1 = tam ToU seviyesi
        best_bid_norm   = float(np.clip(obs.best_bid      / self.tou_price, 0.0, 1.5))
        last_price_norm = float(np.clip(obs.last_avg_price / self.tou_price, 0.0, 1.5))

        return np.array(
            [sin_h, cos_h, supply_norm, best_bid_norm, last_price_norm],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Policy — ε-greedy with action grid
    # ------------------------------------------------------------------

    def _q_values_batch(self, state_np: np.ndarray, net: nn.Module) -> torch.Tensor:
        """
        Verilen state için N_GRID aksiyonun Q-değerlerini tek batch'te hesaplar.
        state_np: (STATE_DIM,) numpy array
        returns : (N_GRID,) tensor
        """
        state_t = torch.tensor(state_np, dtype=torch.float32, device=self.device)
        # state_t: (D,) → (N_GRID, D)  — boyut otomatik
        state_batch = state_t.unsqueeze(0).expand(N_GRID, -1)
        # action: (N_GRID, 1)
        action_batch = self._action_grid_t.unsqueeze(1)
        with torch.no_grad():
            q_vals = net(state_batch, action_batch)  # (N_GRID,)
        return q_vals

    def _select_action(self, state_np: np.ndarray) -> float:
        """ε-greedy aksiyon seçimi → a ∈ [0,1]."""
        if np.random.random() < self.epsilon:
            return float(np.random.choice(ACTION_GRID))
        q_vals = self._q_values_batch(state_np, self.q_net)
        best_idx = int(torch.argmax(q_vals).item())
        return float(ACTION_GRID[best_idx])

    # ------------------------------------------------------------------
    # Core overrides
    # ------------------------------------------------------------------

    def decide_action(self, obs: Observation) -> Action:
        """DQN politikasıyla aksiyon seç."""
        self._current_hour = obs.hour
        state = self._encode_state(obs)

        # Her adımda state/obs kaydet (update_q bunu kullanır)
        self._prev_state = state
        self._prev_obs   = obs
        self._prev_action = None  # sadece satıcı ise doldurulur

        net = obs.inflexible_load
        order = None

        if net > 0.01:
            # ALICI: MB eğrisi (Q-Learning ile aynı, öğrenme yok)
            bid_price = self.compute_mb(net, hour=obs.hour)
            order = Order(
                agent_id=self.agent_id,
                price=float(np.clip(bid_price, 0.001, 2.0)),
                quantity=net,
                is_buy=True,
                is_emergency=False,
            )

        elif net < -0.01:
            # SATICI: NN'den gelen aksiyon → fiyat
            action = self._select_action(state)
            self._prev_action = action

            ask_price = self.fit_price + action * (self.tou_price - self.fit_price)
            order = Order(
                agent_id=self.agent_id,
                price=float(np.clip(ask_price, 0.001, 2.0)),
                quantity=abs(net),
                is_buy=False,
                is_emergency=False,
            )

        return Action(order=order)

    def update_q(self, reward: float, next_obs: Optional[Observation]):
        """
        Replay buffer'a geçiş ekle; buffer yeterliyse bir gradient adımı at.

        Sadece satıcı konumundayken (prev_action is not None) güncelleme yapılır.
        """
        # Satıcı değilse veya geçiş yok → atla
        if self._prev_obs is None or self._prev_action is None:
            return
        if self._prev_obs.inflexible_load > 0:
            return

        # Sonraki state
        done: float
        if next_obs is not None:
            next_state = self._encode_state(next_obs)
            done = 0.0
        else:
            next_state = np.zeros(STATE_DIM, dtype=np.float32)
            done = 1.0

        # Buffer'a ekle
        self.replay_buffer.push(
            self._prev_state,
            self._prev_action,
            reward,
            next_state,
            done,
        )
        self._episode_reward_acc += reward

        # Buffer yeterliyse eğit
        if len(self.replay_buffer) >= MIN_BUFFER_SIZE:
            self._train_step()

    def _train_step(self):
        """Bir mini-batch gradient güncelleme adımı."""
        s, a, r, s_, d = self.replay_buffer.sample(BATCH_SIZE)

        s_t  = torch.tensor(s,  dtype=torch.float32, device=self.device)   # (B,3)
        a_t  = torch.tensor(a,  dtype=torch.float32, device=self.device).unsqueeze(1)  # (B,1)
        r_t  = torch.tensor(r,  dtype=torch.float32, device=self.device)   # (B,)
        s__t = torch.tensor(s_, dtype=torch.float32, device=self.device)   # (B,3)
        d_t  = torch.tensor(d,  dtype=torch.float32, device=self.device)   # (B,)

        # Mevcut Q değerleri
        q_current = self.q_net(s_t, a_t)  # (B,)

        # TD hedefi: r + γ·max_{a'} Q_target(s', a')
        with torch.no_grad():
            B = s__t.shape[0]
            D = s__t.shape[1]  # state boyutu — hardcode değil, otomatik alınır
            # s__t: (B, D) → (B*N_GRID, D) — her s' için N_GRID aksiyon dene
            s_exp = s__t.unsqueeze(1).expand(-1, N_GRID, -1).reshape(-1, D)
            # aksiyon grid: (N_GRID,) → (B*N_GRID, 1)
            a_exp = (
                self._action_grid_t
                .unsqueeze(0)
                .expand(B, -1)
                .reshape(-1, 1)
            )
            q_next_all = self.target_net(s_exp, a_exp).reshape(B, N_GRID)  # (B, N_GRID)
            q_next_max = q_next_all.max(dim=1).values                       # (B,)
            target = r_t + self.gamma * q_next_max * (1.0 - d_t)           # (B,)

        loss = F.mse_loss(q_current, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._train_steps += 1

    def end_episode(self):
        """Episode sonu: ε decay, target network sync, reward kaydı."""
        self.episode_rewards.append(self._episode_reward_acc)
        self._episode_reward_acc = 0.0
        self._episode_count += 1

        # ε azalt
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Target network senkron
        if self._episode_count % TARGET_SYNC_FREQ == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Geçiş belleğini temizle
        self._prev_state  = None
        self._prev_action = None
        self._prev_obs    = None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def agent_stats(self) -> dict:
        """Özet istatistikler."""
        return {
            "device":       str(self.device),
            "epsilon":      round(self.epsilon, 4),
            "buffer_size":  len(self.replay_buffer),
            "train_steps":  self._train_steps,
            "episodes":     self._episode_count,
        }

    def greedy_policy_prices(self) -> np.ndarray:
        """Her saat (0-47) için greedy fiyatı döner — Q-Learning ile karşılaştırılabilir.

        Temsili varsayımlar:
          supply_norm=0.5  (ortalama üretim baskısı)
          best_bid_norm=0.5 (ortalama alıcı teklifi ≈ ToU/2)
          last_price_norm=0.5 (ortalama takas fiyatı ≈ ToU/2)
        """
        prices = []
        for hour in range(48):
            angle = 2.0 * np.pi * hour / 48.0
            state = np.array(
                [np.sin(angle), np.cos(angle), 0.5, 0.5, 0.5],
                dtype=np.float32,
            )
            q_vals = self._q_values_batch(state, self.q_net)
            best_idx = int(torch.argmax(q_vals).item())
            a = float(ACTION_GRID[best_idx])
            prices.append(self.fit_price + a * (self.tou_price - self.fit_price))
        return np.array(prices)

    def __repr__(self) -> str:
        return (
            f"DeepSellerQAgent(id={self.agent_id}, type={self.agent_type}, "
            f"ε={self.epsilon:.3f}, device={self.device}, "
            f"buffer={len(self.replay_buffer)}, steps={self._train_steps})"
        )
