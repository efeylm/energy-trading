"""
Tabular Q-Learning Agent for P2P Energy Trading.

State space: Discretized observation of (net_load_bin, hour_bin, price_bin)
Action space: Price multiplier adjustments on top of MB/MC base price

Bu modül sadece Q-Learning mantığını içerir.
Baseline karşılaştırması için baseline_agent.py dosyasına bakınız.

Design:
- State: (net_load_bin, hour_bin, last_price_bin)  → 3D discrete table
- Action: price_adjustment_index → multiplier applied to MB/MC base price
- Reward: -net_cost_this_hour  (maximize revenue / minimize cost)
- Update: Q(s,a) ← Q(s,a) + α [r + γ·max Q(s',·) − Q(s,a)]
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional, Tuple

from src.config import SimConfig, AgentMBParams, AgentMCParams
from src.agent import EnergyAgent, Observation, Action
from src.market import Order


# ---------------------------------------------------------------------------
# Discretisation helpers
# ---------------------------------------------------------------------------

def _bin(value: float, edges: np.ndarray) -> int:
    """Return the bin index for *value* given monotone *edges* (len N → N-1 bins)."""
    return int(np.clip(np.searchsorted(edges, value, side="right") - 1, 0, len(edges) - 2))


# State-space edges — these define the resolution of the Q-table.
# Net load (kWh): negative = surplus producer, positive = deficit consumer
NET_LOAD_EDGES = np.array([-10.0, -4.0, -1.0, 0.0, 1.0, 4.0, 10.0])   # 6 bins
# Hour of day edges (0-47 steps, mapped to 0-24 h)
HOUR_EDGES      = np.array([0, 8, 16, 24, 32, 40, 48])                  # 6 bins
# Last average price ($/kWh)
PRICE_EDGES     = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 1.0])  # 6 bins

N_NET_LOAD = len(NET_LOAD_EDGES) - 1  # 6
N_HOUR     = len(HOUR_EDGES)     - 1  # 6
N_PRICE    = len(PRICE_EDGES)    - 1  # 6

# Action space: price multipliers applied to the MB/MC base price
# Values < 1.0 → underbid / underask  (aggressive)
# Values > 1.0 → overbid / overask    (conservative)
PRICE_MULTIPLIERS = np.array([0.70, 0.85, 1.00, 1.15, 1.30])
N_ACTIONS = len(PRICE_MULTIPLIERS)  # 5


class QLearningAgent(EnergyAgent):
    """Tabular Q-Learning energy trading agent.

    Extends EnergyAgent but overrides decide_action() to use a
    learned Q-table instead of the raw MB/MC heuristic.

    Parameters
    ----------
    alpha_lr   : Q-learning step size (learning rate)
    gamma      : Discount factor
    epsilon    : Initial ε for ε-greedy exploration
    epsilon_min: Minimum ε (floor after decay)
    epsilon_decay: Multiplicative decay applied after each episode (day)
    """

    def __init__(
        self,
        agent_id: int,
        agent_type: str,
        mb_params: AgentMBParams,
        mc_params: AgentMCParams,
        config: SimConfig,
        alpha_lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.97,
    ):
        super().__init__(agent_id, agent_type, mb_params, mc_params, config)

        self.alpha_lr      = alpha_lr
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: shape (N_NET_LOAD, N_HOUR, N_PRICE, N_ACTIONS)
        # Initialised to small optimistic values to encourage exploration
        self.q_table: np.ndarray = np.zeros(
            (N_NET_LOAD, N_HOUR, N_PRICE, N_ACTIONS), dtype=np.float64
        )

        # Transition memory for single-step update
        self._prev_state: Optional[Tuple[int, int, int]] = None
        self._prev_action_idx: Optional[int] = None
        self._prev_obs: Optional[Observation] = None

        # Episode statistics
        self.episode_rewards: list[float] = []
        self._episode_reward_acc: float = 0.0

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------

    def _encode_state(self, obs: Observation) -> Tuple[int, int, int]:
        """Map a continuous Observation into a discrete (s0, s1, s2) tuple."""
        s0 = _bin(obs.inflexible_load, NET_LOAD_EDGES)
        s1 = _bin(float(obs.hour),     HOUR_EDGES)
        s2 = _bin(obs.last_avg_price,  PRICE_EDGES)
        return s0, s1, s2

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def _select_action(self, state: Tuple[int, int, int]) -> int:
        """ε-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        return int(np.argmax(self.q_table[state]))

    def _base_price(self, obs: Observation) -> float:
        """MB/MC base price (same logic as parent heuristic)."""
        net = obs.inflexible_load
        if net > 0.0:
            return self.compute_mb(net, hour=obs.hour)
        else:
            return self.compute_mc(abs(net))

    # ------------------------------------------------------------------
    # Core overrides
    # ------------------------------------------------------------------

    def decide_action(self, obs: Observation) -> Action:
        """Select action using learned Q-table with ε-greedy exploration."""
        self._current_hour = obs.hour
        state = self._encode_state(obs)
        action_idx = self._select_action(state)

        # Store for Q-update after reward is received
        self._prev_state = state
        self._prev_action_idx = action_idx
        self._prev_obs = obs

        net = obs.inflexible_load
        multiplier = PRICE_MULTIPLIERS[action_idx]

        order = None
        if net > 0.01:
            # BUYER: Only use the raw MB curve price. 
            # No strategic multiplier applied (as per "Only Sellers Learn" logic).
            bid_price = self.compute_mb(net, hour=obs.hour)
            order = Order(
                agent_id=self.agent_id,
                price=float(np.clip(bid_price, 0.001, 2.0)),
                quantity=net,
                is_buy=True,
                is_emergency=False,
            )
        elif net < -0.01:
            # SELLER: Apply Q-Learning multiplier to find the strategic ask price.
            base = self.compute_mc(abs(net))
            ask_price = float(np.clip(base * multiplier, 0.001, 2.0))
            order = Order(
                agent_id=self.agent_id,
                price=ask_price,
                quantity=abs(net),
                is_buy=False,
                is_emergency=False,
            )

        return Action(order=order)

    def update_q(self, reward: float, next_obs: Optional[Observation]):
        """Apply Q-learning update after observing reward and next state.

        Called externally by the training loop (QLearningEnv) once per step.

        Parameters
        ----------
        reward   : Scalar reward for the completed step.
        next_obs : Observation for the next step (None if terminal).
        """
        if self._prev_state is None or self._prev_action_idx is None or self._prev_obs is None:
            return

        # SADECE SATICIYKEN ÖĞREN: Eğer önceki adımda alıcıysak tabloyu güncelleme.
        # Bu, ajanın sadece satış stratejilerini optimize etmesini sağlar.
        if self._prev_obs.inflexible_load > 0:
            return

        s  = self._prev_state
        a  = self._prev_action_idx

        # Bootstrap from next state
        if next_obs is not None:
            s_next = self._encode_state(next_obs)
            max_q_next = float(np.max(self.q_table[s_next]))
        else:
            max_q_next = 0.0

        # Q-update rule
        current_q = self.q_table[s][a]
        td_target  = reward + self.gamma * max_q_next
        td_error   = td_target - current_q
        self.q_table[s][a] += self.alpha_lr * td_error

        self._episode_reward_acc += reward

    def end_episode(self):
        """Call at the end of each simulation day to decay ε and log reward."""
        self.episode_rewards.append(self._episode_reward_acc)
        self._episode_reward_acc = 0.0
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Reset transition memory
        self._prev_state      = None
        self._prev_action_idx = None
        self._prev_obs        = None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def q_table_stats(self) -> dict:
        """Return summary statistics for the Q-table (useful for debugging)."""
        return {
            "mean":    float(np.mean(self.q_table)),
            "std":     float(np.std(self.q_table)),
            "max":     float(np.max(self.q_table)),
            "min":     float(np.min(self.q_table)),
            "nonzero": int(np.count_nonzero(self.q_table)),
        }

    def greedy_policy_summary(self) -> np.ndarray:
        """Return the greedy action index for each state (3-D array)."""
        return np.argmax(self.q_table, axis=-1)

    def __repr__(self) -> str:
        return (
            f"QLearningAgent(id={self.agent_id}, type={self.agent_type}, "
            f"ε={self.epsilon:.3f}, cost={self.total_cost:.3f}$)"
        )