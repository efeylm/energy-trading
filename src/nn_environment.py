"""
Deep Q-Learning için genişletilmiş ortam: NNQLearningEnv

QLearningEnv ile aynı pattern — sadece ajan tipi farklı.
QLearningAgent (Q-tablo) → DeepSellerQAgent (NN)

Her adımdan sonra:
  agent.update_q(reward, next_obs)

Her episode sonunda:
  agent.end_episode()   → ε decay + target network sync
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.config import SimConfig
from src.environment import EnergyTradingEnv
from src.metrics import MetricsCollector
from src.nn_agent import DeepSellerQAgent
from src.agent import Observation


class NNQLearningEnv(EnergyTradingEnv):
    """EnergyTradingEnv + DeepSellerQAgent factory + update loop.

    QLearningEnv'in birebir NN karşılığı. Aynı training ve evaluation
    arayüzünü sunar → comparison.py her ikisini de aynı şekilde kullanabilir.
    """

    def __init__(
        self,
        config: SimConfig,
        alpha_lr: float = 3e-4,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.985,
    ):
        super().__init__(config)
        self.nn_alpha_lr      = alpha_lr
        self.nn_gamma         = gamma
        self.nn_epsilon       = epsilon
        self.nn_epsilon_min   = epsilon_min
        self.nn_epsilon_decay = epsilon_decay

        self.training_enabled = True

        # NN ağırlıkları ve replay buffer episode'lar boyunca korunur
        self._nn_agents: Dict[int, DeepSellerQAgent] = {}

    # ------------------------------------------------------------------
    # Agent factory
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Dict:
        """Environment'ı sıfırla; NN ağırlıklarını ve buffer'ı koru."""
        obs = super().reset(seed=seed)

        new_agents = []
        for agent in self.agents:
            aid = agent.agent_id

            if aid in self._nn_agents:
                # Mevcut NN-ajan: ağırlıkları ve buffer'ı KORU, sayaçları sıfırla
                existing = self._nn_agents[aid]

                existing.agent_type = agent.agent_type
                existing.mb_params  = agent.mb_params
                existing.mc_params  = agent.mc_params
                existing.config     = self.config
                existing.fit_price  = self.config.fit_price
                existing.tou_price  = self.config.tou_price

                existing.total_cost       = 0.0
                existing.total_reward     = 0.0
                existing.total_traded_kwh = 0.0
                existing.unmet_demand     = 0.0
                existing.curtailed_energy = 0.0
                existing.hourly_log       = []

                existing._prev_state  = None
                existing._prev_action = None
                existing._prev_obs    = None

                new_agents.append(existing)
            else:
                # İlk oluşturma
                nn_agent = DeepSellerQAgent(
                    agent_id=aid,
                    agent_type=agent.agent_type,
                    mb_params=agent.mb_params,
                    mc_params=agent.mc_params,
                    config=self.config,
                    alpha_lr=self.nn_alpha_lr,
                    gamma=self.nn_gamma,
                    epsilon=self.nn_epsilon,
                    epsilon_min=self.nn_epsilon_min,
                    epsilon_decay=self.nn_epsilon_decay,
                )
                self._nn_agents[aid] = nn_agent
                new_agents.append(nn_agent)

        self.agents = new_agents
        return obs

    # ------------------------------------------------------------------
    # Step override — Q-update entegrasyonu
    # ------------------------------------------------------------------

    def step(self) -> Tuple[Dict, Dict[int, float], bool, Dict]:
        """Bir adım çalıştır; ardından tüm NN-ajanları güncelle."""
        next_obs_dict, rewards, done, info = super().step()

        if self.training_enabled:
            for agent in self.agents:
                if not isinstance(agent, DeepSellerQAgent):
                    continue
                aid      = agent.agent_id
                reward   = rewards.get(aid, 0.0)
                next_obs = next_obs_dict.get(aid, None)
                agent.update_q(reward=reward, next_obs=next_obs)

        return next_obs_dict, rewards, done, info

    def end_episode(self):
        """Episode sonu: ε decay ve target sync."""
        for agent in self.agents:
            if isinstance(agent, DeepSellerQAgent):
                agent.end_episode()

    # ------------------------------------------------------------------
    # Eğitim döngüsü
    # ------------------------------------------------------------------

    def train(self, n_episodes: int, verbose: bool = True) -> List[float]:
        """n_episodes boyunca Deep-Q eğitimi yap.

        Returns
        -------
        episode_rewards : Her episode'daki toplam sistem ödülü listesi.
        """
        episode_rewards: List[float] = []

        for ep in range(n_episodes):
            seed = self.config.seed + ep
            self.reset(seed=seed)

            done = False
            while not done:
                _, _, done, _ = self.step()

            self.end_episode()

            total_reward = sum(
                agent.episode_rewards[-1]
                for agent in self.agents
                if isinstance(agent, DeepSellerQAgent)
            )
            episode_rewards.append(total_reward)

            if verbose and (ep % max(1, n_episodes // 10) == 0 or ep == n_episodes - 1):
                eps_val = next(
                    (a.epsilon for a in self.agents if isinstance(a, DeepSellerQAgent)),
                    0.0,
                )
                buf_size = next(
                    (len(a.replay_buffer) for a in self.agents if isinstance(a, DeepSellerQAgent)),
                    0,
                )
                print(
                    f"  [Deep-Q] Episode {ep+1:4d}/{n_episodes} | "
                    f"System Reward: {total_reward:9.2f} | "
                    f"ε={eps_val:.3f} | "
                    f"Buffer={buf_size}"
                )

        return episode_rewards

    def run_evaluation(self, seed: Optional[int] = None, verbose: bool = True) -> MetricsCollector:
        """Epsilon=0 (tam greedy) ile bir değerlendirme günü çalıştır."""
        eval_seed = seed if seed is not None else self.config.seed + 9999
        self.reset(seed=eval_seed)

        saved_epsilons = {}
        old_training   = self.training_enabled
        self.training_enabled = False

        for agent in self.agents:
            if isinstance(agent, DeepSellerQAgent):
                saved_epsilons[agent.agent_id] = agent.epsilon
                agent.epsilon = 0.0

        if verbose:
            print("\n" + "=" * 60)
            print("  Deep-Q — Evaluation Run (ε=0, Greedy Policy)")
            print("=" * 60)

        try:
            done = False
            while not done:
                _, _, done, _ = self.step()
        finally:
            if verbose:
                print()
                print(self.metrics.summary())

            self.training_enabled = old_training
            for agent in self.agents:
                if isinstance(agent, DeepSellerQAgent):
                    agent.epsilon = saved_epsilons.get(agent.agent_id, 0.0)

        return self.metrics
