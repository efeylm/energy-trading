"""
POMG Environment for P2P Energy Trading.

Implements the Partially Observable Markov Game (POMG) from
Qiu et al. (IJCAI-21) Section 3.3:

    POMG = (N, S, {O_n}, {A_n}, T, {R_n})

- N agents (prosumers + consumers) with pre-determined daily profiles
- Each hour is one auction period with batch clearing
- No grid connection: unmatched energy is curtailed (starvation possible)
- Emergency bidding for agents near starvation threshold

The environment orchestrates:
1. Profile-based generation/consumption per hour
2. Agent observations and decisions
3. Market order submission
4. Batch clearing (Algorithm 1)
5. Trade settlement and battery updates
6. Reward computation
"""

from typing import List, Dict, Tuple, Optional
import numpy as np

from src.config import SimConfig
from src.battery import Battery
from src.agent import EnergyAgent, Observation, Action
from src.market import DoubleAuctionMarket, Order, ClearingResult
from src.profiles import ProfileManager
from src.metrics import MetricsCollector, HourlyMetrics


class EnergyTradingEnv:
    """POMG environment for P2P energy trading simulation.
    
    Runs a 24-hour simulation day where agents trade energy
    through a batch double auction market.
    """
    
    def __init__(self, config: SimConfig):
        self.config = config
        self.market = DoubleAuctionMarket()
        self.metrics = MetricsCollector(config.n_agents)
        self.profiles: Optional[ProfileManager] = None
        self.agents: List[EnergyAgent] = []
        self.current_hour = 0
        
    def reset(self, seed: Optional[int] = None) -> Dict:
        """Initialize/reset the environment for a new simulation day.
        
        Creates agents, generates profiles, resets market and metrics.
        
        Returns:
            Initial observations for all agents.
        """
        actual_seed = seed if seed is not None else self.config.seed
        
        # Generate daily profiles
        self.profiles = ProfileManager(
            n_prosumers=self.config.n_prosumers,
            n_consumers=self.config.n_consumers,
            battery_config=self.config.battery,
            seed=actual_seed,
        )
        
        # Create agents
        self.agents = []
        for i in range(self.config.n_agents):
            if i < self.config.n_prosumers:
                agent_type = "prosumer"
                mb_params = self.config.prosumer_mb
                mc_params = self.config.prosumer_mc
            else:
                agent_type = "consumer"
                mb_params = self.config.consumer_mb
                mc_params = self.config.consumer_mc
            
            # Create battery with profile-determined initial energy
            battery = Battery(
                capacity_min=self.config.battery.capacity_min,
                capacity_max=self.config.battery.capacity_max,
                power_max=self.config.battery.power_max,
                eta_charge=self.config.battery.eta_charge,
                eta_discharge=self.config.battery.eta_discharge,
                initial_energy=self.profiles.get_initial_battery(i),
            )
            
            agent = EnergyAgent(
                agent_id=i,
                agent_type=agent_type,
                mb_params=mb_params,
                mc_params=mc_params,
                battery=battery,
                config=self.config,
            )
            self.agents.append(agent)
        
        # Reset market and metrics
        self.market.reset()
        self.metrics = MetricsCollector(self.config.n_agents)
        self.current_hour = 0
        
        # Print profile summary
        print(self.profiles.summary())
        print()
        
        return self._get_all_observations()
    
    def _get_all_observations(self) -> Dict[int, Observation]:
        """Generate observations for all agents at current hour."""
        market_info = self.market.get_market_info()
        observations = {}
        
        for agent in self.agents:
            obs = Observation(
                inflexible_load=self.profiles.get_inflexible_load(
                    agent.agent_id, self.current_hour
                ),
                battery_state=agent.battery.get_state(),
                best_bid=market_info["best_bid"],
                best_ask=market_info["best_ask"],
                last_avg_price=market_info["last_avg_price"],
                hour=self.current_hour,
                pv_generation=self.profiles.get_pv_generation(
                    agent.agent_id, self.current_hour
                ),
                load_demand=self.profiles.get_load_demand(
                    agent.agent_id, self.current_hour
                ),
            )
            observations[agent.agent_id] = obs
        
        return observations
    
    def step(self) -> Tuple[Dict, Dict[int, float], bool, Dict]:
        """Execute one hour (auction period) of the simulation.
        
        Steps:
        1. Generate observations for all agents
        2. Each agent decides action (battery + market order)
        3. Apply battery actions
        4. Submit orders to market
        5. Run batch clearing (Algorithm 1)
        6. Settle trades and handle unmatched orders
        7. Compute rewards and metrics
        
        Returns:
            (observations, rewards, done, info)
        """
        hour = self.current_hour
        dt = self.config.delta_t
        
        # --- Step 1: Observations ---
        observations = self._get_all_observations()
        
        # --- Step 2: Agent decisions ---
        actions: Dict[int, Action] = {}
        for agent in self.agents:
            obs = observations[agent.agent_id]
            action = agent.decide_action(obs)
            actions[agent.agent_id] = action
        
        # --- Step 3: Apply battery actions ---
        for agent in self.agents:
            action = actions[agent.agent_id]
            agent.apply_battery_action(action.battery_charge, dt)
        
        # --- Step 4: Submit orders to market ---
        n_buyers = 0
        n_sellers = 0
        for agent in self.agents:
            action = actions[agent.agent_id]
            if action.order is not None:
                self.market.submit_order(action.order)
                if action.order.is_buy:
                    n_buyers += 1
                else:
                    n_sellers += 1
        
        # --- Step 5: Batch clearing ---
        clearing_result = self.market.clear()
        
        # --- Step 6: Settle trades and handle unmatched ---
        rewards = {}
        total_unmet = 0.0
        total_curtailed = 0.0
        n_starvation = 0
        agent_battery_soc = {}
        
        for agent in self.agents:
            agent_id = agent.agent_id
            
            # Find trades involving this agent
            trades_as_buyer = [
                t for t in clearing_result.trades if t.buyer_id == agent_id
            ]
            trades_as_seller = [
                t for t in clearing_result.trades if t.seller_id == agent_id
            ]
            
            # Calculate matched quantities
            bought_kwh = sum(t.quantity for t in trades_as_buyer)
            sold_kwh = sum(t.quantity for t in trades_as_seller)
            
            # Determine unmet demand / curtailed surplus
            action = actions[agent_id]
            unmet_demand = 0.0
            curtailed_surplus = 0.0
            
            if action.order is not None:
                if action.order.is_buy:
                    # Check how much of the buy order was fulfilled
                    requested = action.order.quantity
                    unmet_demand = max(0.0, requested - bought_kwh)
                else:
                    # Check how much of the sell order was fulfilled
                    offered = action.order.quantity
                    curtailed_surplus = max(0.0, offered - sold_kwh)
            
            # Count starvation events
            if unmet_demand > 0.01:
                n_starvation += 1
            
            total_unmet += unmet_demand
            total_curtailed += curtailed_surplus
            
            # Process results in agent
            agent.process_trade_results(
                trades_as_buyer=trades_as_buyer,
                trades_as_seller=trades_as_seller,
                unmet_demand=unmet_demand,
                curtailed_surplus=curtailed_surplus,
                hour=hour,
            )
            
            # Get reward
            rewards[agent_id] = agent.get_reward(hour)
            
            # Track battery SoC
            agent_battery_soc[agent_id] = agent.battery.soc_fraction
            
            # Record agent metrics
            buy_cost = sum(t.price * t.quantity for t in trades_as_buyer)
            sell_income = sum(t.price * t.quantity for t in trades_as_seller)
            self.metrics.record_agent_hour(
                agent_id=agent_id,
                cost=buy_cost - sell_income,
                bought=bought_kwh,
                sold=sold_kwh,
                unmet=unmet_demand,
                curtailed=curtailed_surplus,
            )
        
        # --- Step 7: Record hourly metrics ---
        hourly = HourlyMetrics(
            hour=hour,
            average_price=clearing_result.average_price,
            total_traded_kwh=clearing_result.total_traded_kwh,
            n_trades=len(clearing_result.trades),
            total_unmet_demand=total_unmet,
            total_curtailed=total_curtailed,
            n_buyers=n_buyers,
            n_sellers=n_sellers,
            n_starvation_events=n_starvation,
            agent_battery_soc=agent_battery_soc,
        )
        self.metrics.record_hour(hourly)
        
        # Print hour summary
        self._print_hour_summary(hour, clearing_result, total_unmet, total_curtailed)
        
        # Advance time
        self.current_hour += 1
        done = self.current_hour >= self.config.t_hours
        
        # Next observations
        next_obs = self._get_all_observations() if not done else {}
        
        info = {
            "clearing_result": clearing_result,
            "total_unmet_demand": total_unmet,
            "total_curtailed": total_curtailed,
            "n_starvation_events": n_starvation,
        }
        
        return next_obs, rewards, done, info
    
    def run_day(self) -> MetricsCollector:
        """Run a complete 24-hour simulation day.
        
        Returns:
            MetricsCollector with all recorded data.
        """
        self.reset()
        
        print("=" * 60)
        print("  Starting 24-hour P2P Energy Trading Simulation")
        print("=" * 60)
        print()
        
        done = False
        while not done:
            _, _, done, _ = self.step()
        
        print()
        print(self.metrics.summary())
        
        return self.metrics
    
    def _print_hour_summary(
        self,
        hour: int,
        result: ClearingResult,
        unmet: float,
        curtailed: float,
    ):
        """Print a concise summary of one hour's activity."""
        price_str = f"${result.average_price:.4f}" if result.trades else "N/A"
        print(
            f"  Hour {hour:2d}: "
            f"Trades={len(result.trades):2d}, "
            f"Volume={result.total_traded_kwh:6.2f}kWh, "
            f"Price={price_str}, "
            f"Unmet={unmet:.2f}kWh, "
            f"Curtailed={curtailed:.2f}kWh"
        )
