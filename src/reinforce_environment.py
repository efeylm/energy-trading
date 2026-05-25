"""
REINFORCE için genişletilmiş ortam: REINFORCEEnv.

QLearningEnv ile birebir aynı mimari desen:
  - EnergyTradingEnv'i miras alır (environment.py'ye dokunulmaz)
  - reset()          → NN ağırlıklarını korur, episode sayaçlarını sıfırlar
  - step()           → parent adımı çalıştırır, reward'ları agent'a iletir
  - end_episode()    → gradient güncelleme + sigma decay (tüm ajanlar)
  - train()          → n_episodes eğitim döngüsü
  - run_evaluation() → sigma=0 deterministik değerlendirme

Miras zinciri:
  EnergyTradingEnv
       └── REINFORCEEnv   (bu dosya)

Dokunulmayan dosyalar:
  environment.py, agent.py, market.py, metrics.py,
  baseline_agent.py, ql_agent.py, ql_environment.py
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.config import SimConfig, AgentMBParams, AgentMCParams
from src.environment import EnergyTradingEnv
from src.metrics import MetricsCollector
from src.reinforce_agent import REINFORCEAgent


class REINFORCEEnv(EnergyTradingEnv):
    """EnergyTradingEnv + REINFORCEAgent factory + gradient update loop.

    QLearningEnv ile karşılaştırma:
      QLearningEnv.step()        → agent.update_q(reward)         [her adımda TD]
      REINFORCEEnv.step()        → agent.store_reward(reward)     [her adımda topla]
      REINFORCEEnv.end_episode() → agent.end_episode()            [episode sonunda gradient]
    """

    def __init__(
        self,
        config: SimConfig,
        lr: float = 5e-4,
        gamma: float = 0.95,
        sigma: float = 0.25,
        sigma_min: float = 0.05,
        sigma_decay: float = 0.993,
        hidden: int = 64,
    ):
        super().__init__(config)
        self.nn_lr          = lr
        self.nn_gamma       = gamma
        self.nn_sigma       = sigma
        self.nn_sigma_min   = sigma_min
        self.nn_sigma_decay = sigma_decay
        self.nn_hidden      = hidden

        self.training_enabled = True

        # NN ağırlıkları episode'lar boyunca bu dict'te korunur.
        # QLearningEnv'deki _ql_agents ile aynı mantık.
        self._rf_agents: Dict[int, REINFORCEAgent] = {}

    # ------------------------------------------------------------------
    # Agent factory — parent'ın reset() metodunu genişletir
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Dict:
        """Environment'ı sıfırla; NN ağırlıklarını koru.

        Parent reset() çağrısı: profiller, market, metrics yenilenir.
        Ardından parent'ın EnergyAgent'ları REINFORCEAgent ile değiştirilir.
        Mevcut NN ağırlıkları KORUNUR — yalnızca episode sayaçları sıfırlanır.
        """
        obs = super().reset(seed=seed)

        new_agents = []
        for agent in self.agents:
            aid = agent.agent_id

            if aid in self._rf_agents:
                # ── Mevcut ajan: NN ağırlıklarını KORU ───────────────
                existing = self._rf_agents[aid]

                # Ortam parametrelerini yenile (±15% varyasyon güncellemesi)
                existing.agent_type = agent.agent_type
                existing.mb_params  = agent.mb_params
                existing.mc_params  = agent.mc_params
                existing.config     = self.config

                # EnergyAgent sayaçlarını sıfırla
                existing.total_cost       = 0.0
                existing.total_reward     = 0.0
                existing.total_traded_kwh = 0.0
                existing.unmet_demand     = 0.0
                existing.curtailed_energy = 0.0
                existing.hourly_log       = []

                # Episode buffer'ı temizle (önceki episode artığı kalmasın)
                existing._episode_a       = []
                existing._episode_rewards = []
                existing._last_was_seller = False

                new_agents.append(existing)

            else:
                # ── İlk oluşturma ─────────────────────────────────────
                rf_agent = REINFORCEAgent(
                    agent_id=aid,
                    agent_type=agent.agent_type,
                    mb_params=agent.mb_params,
                    mc_params=agent.mc_params,
                    config=self.config,
                    lr=self.nn_lr,
                    gamma=self.nn_gamma,
                    sigma=self.nn_sigma,
                    sigma_min=self.nn_sigma_min,
                    sigma_decay=self.nn_sigma_decay,
                    hidden=self.nn_hidden,
                )
                self._rf_agents[aid] = rf_agent
                new_agents.append(rf_agent)

        self.agents = new_agents
        return obs

    # ------------------------------------------------------------------
    # Step override — reward iletimi
    # ------------------------------------------------------------------

    def step(self) -> Tuple[Dict, Dict[int, float], bool, Dict]:
        """Bir adım çalıştır; reward'ları REINFORCE ajanlarına ilet.

        Q-Learning farkı:
          QLearningEnv.step() → agent.update_q(reward)    [anında TD güncelleme]
          Bu metod         → agent.store_reward(reward)   [buffer'a ekle, bekle]
        Gradient güncellemesi episode bittikten sonra end_episode()'da yapılır.
        """
        next_obs, rewards, done, info = super().step()

        if self.training_enabled:
            for agent in self.agents:
                if isinstance(agent, REINFORCEAgent):
                    reward = rewards.get(agent.agent_id, 0.0)
                    agent.store_reward(reward)

        return next_obs, rewards, done, info

    def end_episode(self):
        """Episode sonu: tüm REINFORCE ajanlarında gradient güncelle + sigma decay."""
        for agent in self.agents:
            if isinstance(agent, REINFORCEAgent):
                agent.end_episode()

    # ------------------------------------------------------------------
    # Eğitim döngüsü
    # ------------------------------------------------------------------

    def train(self, n_episodes: int, verbose: bool = True) -> List[float]:
        """n_episodes boyunca REINFORCE eğitimi yap.

        Her episode = 1 tam simülasyon günü (48 adım).
        Episode sonunda gradient hesaplanır ve NN ağırlıkları güncellenir.

        Q-Learning ile fark:
          QL   → her adımda Q[s,a] += α × TD_error   (online)
          REINF → episode sonunda loss.backward()      (batch)

        Returns
        -------
        episode_rewards : Her episode'daki toplam sistem ödülü.
        """
        episode_rewards: List[float] = []

        for ep in range(n_episodes):
            seed = self.config.seed + ep
            self.reset(seed=seed)

            done = False
            while not done:
                _, _, done, _ = self.step()

            self.end_episode()

            # Toplam sistem ödülü (sadece REINFORCE ajanlarından)
            total_reward = sum(
                agent.episode_rewards[-1]
                for agent in self.agents
                if isinstance(agent, REINFORCEAgent)
            )
            episode_rewards.append(total_reward)

            if verbose and (ep % max(1, n_episodes // 10) == 0 or ep == n_episodes - 1):
                sigma = self._rf_agents[0].sigma if 0 in self._rf_agents else 0.0
                print(
                    f"  [REINFORCE] Episode {ep+1:4d}/{n_episodes} | "
                    f"System Reward: {total_reward:9.2f} | "
                    f"σ={sigma:.3f}"
                )

        return episode_rewards

    # ------------------------------------------------------------------
    # Deterministik değerlendirme
    # ------------------------------------------------------------------

    def run_evaluation(
        self,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> MetricsCollector:
        """Sigma=0 (gürültüsüz) deterministik değerlendirme günü çalıştır.

        Q-Learning'deki ε=0 greedy evaluation ile eşdeğerdir.
        """
        eval_seed = seed if seed is not None else self.config.seed + 9999
        self.reset(seed=eval_seed)

        # Eğitimi kapat, deterministik moda geç
        old_training = self.training_enabled
        self.training_enabled = False
        for agent in self.agents:
            if isinstance(agent, REINFORCEAgent):
                agent.training_mode = False

        if verbose:
            print("\n" + "=" * 60)
            print("  REINFORCE-NN — Evaluation Run (σ=0, Deterministic)")
            print("=" * 60)

        try:
            done = False
            while not done:
                _, _, done, _ = self.step()
        finally:
            if verbose:
                print()
                print(self.metrics.summary())

            # Ayarları geri yükle
            self.training_enabled = old_training
            for agent in self.agents:
                if isinstance(agent, REINFORCEAgent):
                    agent.training_mode = True

        return self.metrics
