"""
Q-Learning için genişletilmiş ortam: QLearningEnv

EnergyTradingEnv'i miras alır, yalnızca agent oluşturma ve
adım sonrası Q-güncelleme mantığını ekler.

Her adımdan sonra:
  agent.update_q(reward, next_obs)

Her episode sonunda:
  agent.end_episode()   → ε decay

Baseline karşılaştırması için BaselineEnv da bu dosyadadır;
aynı profilleri, aynı market parametrelerini kullanır,
sadece ajanları BaselineAgent olarak oluşturur.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from src.config import SimConfig, AgentMBParams, AgentMCParams
from src.environment import EnergyTradingEnv
from src.metrics import MetricsCollector
from src.ql_agent import QLearningAgent
from src.baseline_agent import BaselineAgent
from src.agent import Observation


class QLearningEnv(EnergyTradingEnv):
    """EnergyTradingEnv + Q-Learning agent factory + update loop."""

    def __init__(
        self,
        config: SimConfig,
        alpha_lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.985,
    ):
        super().__init__(config)
        self.ql_alpha_lr      = alpha_lr
        self.ql_gamma         = gamma
        self.ql_epsilon       = epsilon
        self.ql_epsilon_min   = epsilon_min
        self.ql_epsilon_decay = epsilon_decay

        self.training_enabled = True

        # Q-tabloları episode'lar boyunca korunur; reset() yeniden oluşturmaz.
        # Bu dict agent_id → QLearningAgent eşlemesini sağlar.
        self._ql_agents: Dict[int, QLearningAgent] = {}

    # ------------------------------------------------------------------
    # Agent factory — parent'ın reset() metodu bunu çağırır
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Dict:
        """Environment'ı sıfırla; Q-ajanları koru (Q-tabloları silinmez)."""
        # Parent'ın reset'ini çağır (profiller, market, metrics yenilenir)
        obs = super().reset(seed=seed)

        # Parent'ın oluşturduğu EnergyAgent'ları QLearningAgent ile değiştir.
        # Daha önce oluşturulmuş Q-tabloları mevcut ise onları koru.
        new_agents = []

        for agent in self.agents:
            aid = agent.agent_id

            if aid in self._ql_agents:
                # Mevcut Q-ajan: Öğrenilen Q-tabloyu KORU, ama ortam parametrelerini yenile.
                # Bu, her episode'da değişen ±%15'lik varyasyonun Q-ajana da yansımasını sağlar.
                existing = self._ql_agents[aid]
                
                # Karakteristikleri yenile
                existing.agent_type = agent.agent_type
                existing.mb_params  = agent.mb_params
                existing.mc_params  = agent.mc_params
                existing.config     = self.config
                
                # Sayaçları sıfırla
                existing.total_cost       = 0.0
                existing.total_reward     = 0.0
                existing.total_traded_kwh = 0.0
                existing.unmet_demand     = 0.0
                existing.curtailed_energy = 0.0
                existing.hourly_log       = []
                
                # Belleği temizle (Yeni bir güne taze başlangıç)
                existing._prev_state      = None
                existing._prev_action_idx = None
                existing._prev_obs        = None
                
                new_agents.append(existing)
            else:
                # İlk oluşturma
                ql_agent = QLearningAgent(
                    agent_id=aid,
                    agent_type=agent.agent_type,
                    mb_params=agent.mb_params,
                    mc_params=agent.mc_params,
                    config=self.config,
                    alpha_lr=self.ql_alpha_lr,
                    gamma=self.ql_gamma,
                    epsilon=self.ql_epsilon,
                    epsilon_min=self.ql_epsilon_min,
                    epsilon_decay=self.ql_epsilon_decay,
                )
                self._ql_agents[aid] = ql_agent
                new_agents.append(ql_agent)

        self.agents = new_agents
        return obs

    # ------------------------------------------------------------------
    # Step override — Q-update entegrasyonu
    # ------------------------------------------------------------------

    def step(self) -> Tuple[Dict, Dict[int, float], bool, Dict]:
        """Bir adım çalıştır; ardından tüm Q-ajanları güncelle."""
        # Standart adım
        next_obs_dict, rewards, done, info = super().step()

        # Her ajan için Q-update (Sadece eğitim açıksa)
        if self.training_enabled:
            for agent in self.agents:
                aid = agent.agent_id
                if not isinstance(agent, QLearningAgent):
                    continue
                reward   = rewards.get(aid, 0.0)
                next_obs = next_obs_dict.get(aid, None)
                agent.update_q(reward=reward, next_obs=next_obs)

        return next_obs_dict, rewards, done, info

    def end_episode(self):
        """Episode sonu: tüm Q-ajanlarında ε decay uygula."""
        for agent in self.agents:
            if isinstance(agent, QLearningAgent):
                agent.end_episode()

    # ------------------------------------------------------------------
    # Eğitim döngüsü
    # ------------------------------------------------------------------

    def train(self, n_episodes: int, verbose: bool = True) -> List[float]:
        """n_episodes boyunca Q-Learning eğitimi yap.

        Her episode bir tam simülasyon günüdür.

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

            # Yeni Metrik: Toplam Sistem Ödülü (Öğrenmeyi gösteren gerçek değer)
            total_reward = sum(
                agent.episode_rewards[-1] for agent in self.agents
                if isinstance(agent, QLearningAgent)
            )
            episode_rewards.append(total_reward)

            if verbose and (ep % max(1, n_episodes // 10) == 0 or ep == n_episodes - 1):
                eps = self.agents[0].epsilon if self.agents else 0.0
                print(
                    f"  [Q-Learning] Episode {ep+1:4d}/{n_episodes} | "
                    f"System Reward: {total_reward:9.2f} | "
                    f"ε={eps:.3f}"
                )

        return episode_rewards

    def run_evaluation(self, seed: Optional[int] = None, verbose: bool = True) -> MetricsCollector:
        """Epsilon=0 (tam greedy) ile bir değerlendirme günü çalıştır."""
        eval_seed = seed if seed is not None else self.config.seed + 9999
        self.reset(seed=eval_seed)

        # Epsilon'u geçici olarak sıfırla ve eğitimi kapat
        saved_epsilons = {}
        old_training_enabled = self.training_enabled
        self.training_enabled = False

        for agent in self.agents:
            if isinstance(agent, QLearningAgent):
                saved_epsilons[agent.agent_id] = agent.epsilon
                agent.epsilon = 0.0

        if verbose:
            print("\n" + "=" * 60)
            print("  Q-Learning — Evaluation Run (ε=0, Greedy Policy)")
            print("=" * 60)

        try:
            done = False
            while not done:
                _, _, done, _ = self.step()
        finally:
            if verbose:
                print()
                print(self.metrics.summary())

            # Epsilon'u ve eğitim durumunu geri yükle
            self.training_enabled = old_training_enabled
            for agent in self.agents:
                if isinstance(agent, QLearningAgent):
                    agent.epsilon = saved_epsilons.get(agent.agent_id, 0.0)

        return self.metrics


class BaselineEnv(EnergyTradingEnv):
    """EnergyTradingEnv + BaselineAgent factory."""

    def reset(self, seed: Optional[int] = None) -> Dict:
        obs = super().reset(seed=seed)

        # Parent EnergyAgent'ları BaselineAgent ile değiştir
        new_agents = []
        for agent in self.agents:
            bl_agent = BaselineAgent(
                agent_id=agent.agent_id,
                agent_type=agent.agent_type,
                mb_params=agent.mb_params,
                mc_params=agent.mc_params,
                config=self.config,
            )
            new_agents.append(bl_agent)
        self.agents = new_agents
        return obs

    def run_baseline(self, seed: Optional[int] = None, verbose: bool = True) -> MetricsCollector:
        """Baseline ajanlarla tek bir simülasyon günü çalıştır."""
        run_seed = seed if seed is not None else self.config.seed
        self.reset(seed=run_seed)

        if verbose:
            print("\n" + "=" * 60)
            print("  Baseline (Rule-Based) — Single Run")
            print("=" * 60)

        done = False
        while not done:
            _, _, done, _ = self.step()

        if verbose:
            print()
            print(self.metrics.summary())
        return self.metrics