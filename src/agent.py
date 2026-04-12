"""
Energy agents: Prosumers and Consumers.

Combines:
- Qiu et al. POMG observation/action structure (Section 3.3)
- ADD Bitirme MB/MC curve-based pricing logic (Section 3.1.2)

Each agent:
1. Observes: inflexible load, battery state, market signals, hour
2. Decides: battery action (charge/discharge/store) + market order (price, quantity)
3. Uses MB/MC curves to determine willingness to pay / minimum ask
4. Has emergency bidding when battery drops below starvation threshold
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math

from src.battery import Battery, BatteryState
from src.config import AgentMBParams, AgentMCParams, SimConfig
from src.market import Order


@dataclass
class Observation:
    """Agent's private observation at time step t.
    
    From paper: o_n,t = [P^inf_n,t, E^es_n,t, λ^b_t, λ^s_t]
    Extended with hour info and battery state.
    """
    inflexible_load: float    # P^inf: demand - PV (positive=needs, negative=surplus)
    battery_state: BatteryState
    best_bid: float           # λ^b: current best bid in market
    best_ask: float           # λ^s: current best ask in market
    last_avg_price: float     # Average clearing price from last period
    hour: int                 # Current hour (0-23)
    pv_generation: float      # This hour's PV generation
    load_demand: float        # This hour's load demand


@dataclass
class Action:
    """Agent's action at time step t.
    
    From paper: a_n,t = [a^p, a^q]
    a^p: price decision (determined by MB/MC curves)
    a^q: battery decision (charge/discharge ratio)
    """
    order: Optional[Order]     # Market order (None if agent doesn't trade)
    battery_charge: float      # kW to charge (positive) or discharge (negative)


class EnergyAgent:
    """Base class for energy trading agents.
    
    Supports both prosumer and consumer roles with heuristic strategy:
    - Prosumer: Has PV generation, can be buyer or seller depending on net position
    - Consumer: Only has load, primarily buyer but can sell from battery
    """
    
    def __init__(
        self,
        agent_id: int,
        agent_type: str,       # "prosumer" or "consumer"
        mb_params: AgentMBParams,
        mc_params: AgentMCParams,
        battery: Battery,
        config: SimConfig,
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.mb_params = mb_params
        self.mc_params = mc_params
        self.battery = battery
        self.config = config
        
        # Tracking
        self.total_cost = 0.0           # Cumulative energy cost ($)
        self.total_traded_kwh = 0.0     # Cumulative traded energy
        self.unmet_demand = 0.0         # Cumulative unmet demand (kWh)
        self.curtailed_energy = 0.0     # Cumulative curtailed surplus (kWh)
        self.starvation_count = 0       # Number of starvation events
        self.hourly_log = []            # Detailed per-hour records
    
    def compute_mb(self, quantity: float) -> float:
        """Marginal Benefit: MB(q) = alpha * exp(-beta * q)
        
        The value (willingness to pay) for the q-th unit of energy.
        Decreases with quantity (diminishing marginal utility).
        """
        q = max(0.0, quantity)
        return self.mb_params.alpha * math.exp(-self.mb_params.beta * q)
    
    def compute_mc(self, quantity: float) -> float:
        """Marginal Cost: MC(q) = gamma * q + delta
        
        The cost of producing/releasing the q-th unit of energy.
        Increases with quantity (increasing marginal cost).
        """
        q = max(0.0, quantity)
        return self.mc_params.gamma * q + self.mc_params.delta
    
    def decide_action(self, obs: Observation) -> Action:
        """Heuristic decision-making based on observation.
        
        Logic:
        1. Determine net energy position (inflexible load)
        2. Decide battery action (charge/discharge/store)
        3. Determine market quantity and role (buyer/seller)
        4. Set price using MB/MC curves
        5. Apply emergency bidding if near starvation
        
        Returns:
            Action with order and battery decision.
        """
        dt = self.config.delta_t
        inf_load = obs.inflexible_load  # positive=needs, negative=surplus
        bat_state = obs.battery_state
        
        # --- Step 1: Battery decision ---
        battery_action = 0.0  # kW (positive=charge, negative=discharge)
        
        if inf_load > 0:
            # Agent needs energy — consider discharging battery
            max_discharge = self.battery.max_discharge_power(dt)
            
            if bat_state.soc_fraction > self.config.starvation_threshold / (bat_state.capacity_max - bat_state.capacity_min + 1e-9):
                # Battery has enough charge — discharge to cover part of demand
                # Keep some reserve (don't drain below starvation threshold)
                reserve_energy = self.config.starvation_threshold
                available_above_reserve = max(0.0, self.battery.energy - bat_state.capacity_min - reserve_energy)
                safe_discharge = min(max_discharge, available_above_reserve * self.battery.eta_discharge / dt)
                
                # Discharge up to the inflexible load
                battery_action = -min(safe_discharge, inf_load)
            else:
                # Battery low — don't discharge, go to market with full demand
                battery_action = 0.0
        
        elif inf_load < 0:
            # Agent has surplus energy — decide: store or sell?
            surplus = abs(inf_load)
            max_charge = self.battery.max_charge_power(dt)
            
            if bat_state.soc_fraction < self.config.battery_store_threshold:
                # Battery is relatively empty — prefer storing
                store_amount = min(max_charge, surplus)
                battery_action = store_amount
            else:
                # Battery is sufficiently charged — sell surplus to market
                # Still charge a small portion if possible
                store_fraction = 0.2  # Store 20% of surplus if space available
                store_amount = min(max_charge, surplus * store_fraction)
                battery_action = store_amount
        
        # --- Step 2: Compute market quantity ---
        # Net market quantity = inflexible load + battery action
        # (battery_action positive = charging = consuming more energy from perspective of market)
        # (battery_action negative = discharging = providing energy)
        market_quantity = inf_load + battery_action
        
        # --- Step 3: Determine role and set price ---
        order = None
        
        if market_quantity > 0.01:
            # BUYER — needs to buy from market
            bid_price = self.compute_mb(market_quantity)
            
            # Emergency bidding: if battery critically low
            if self.battery.energy <= self.config.starvation_threshold + self.battery.capacity_min:
                bid_price *= self.config.emergency_price_multiplier
                is_emergency = True
            else:
                is_emergency = False
            
            order = Order(
                agent_id=self.agent_id,
                price=bid_price,
                quantity=market_quantity,
                is_buy=True,
                is_emergency=is_emergency,
            )
        
        elif market_quantity < -0.01:
            # SELLER — has surplus to sell
            sell_quantity = abs(market_quantity)
            ask_price = self.compute_mc(sell_quantity)
            
            order = Order(
                agent_id=self.agent_id,
                price=ask_price,
                quantity=sell_quantity,
                is_buy=False,
                is_emergency=False,
            )
        
        # else: market_quantity ≈ 0 → no market participation this hour
        
        return Action(order=order, battery_charge=battery_action)
    
    def apply_battery_action(self, battery_charge: float, dt: float) -> float:
        """Execute the battery charge/discharge decision.
        
        Returns actual power applied (may differ from requested due to limits).
        """
        if battery_charge > 0:
            return self.battery.charge(battery_charge, dt)
        elif battery_charge < 0:
            return -self.battery.discharge(abs(battery_charge), dt)
        return 0.0
    
    def process_trade_results(
        self,
        trades_as_buyer: list,
        trades_as_seller: list,
        unmet_demand: float,
        curtailed_surplus: float,
        hour: int,
    ):
        """Process the clearing results for this agent.
        
        Updates internal state and tracking variables.
        """
        # Calculate costs from trades
        buy_cost = sum(t.price * t.quantity for t in trades_as_buyer)
        sell_income = sum(t.price * t.quantity for t in trades_as_seller)
        bought_kwh = sum(t.quantity for t in trades_as_buyer)
        sold_kwh = sum(t.quantity for t in trades_as_seller)
        
        net_cost = buy_cost - sell_income
        
        # Starvation penalty
        penalty = 0.0
        if unmet_demand > 0.01:
            penalty = unmet_demand * self.config.starvation_penalty
            self.starvation_count += 1
        
        # Update tracking
        self.total_cost += net_cost + penalty
        self.total_traded_kwh += bought_kwh + sold_kwh
        self.unmet_demand += unmet_demand
        self.curtailed_energy += curtailed_surplus
        
        # Log
        self.hourly_log.append({
            "hour": hour,
            "buy_cost": buy_cost,
            "sell_income": sell_income,
            "net_cost": net_cost,
            "penalty": penalty,
            "bought_kwh": bought_kwh,
            "sold_kwh": sold_kwh,
            "unmet_demand": unmet_demand,
            "curtailed": curtailed_surplus,
            "battery_energy": self.battery.energy,
            "battery_soc": self.battery.soc_fraction,
        })
    
    def get_reward(self, hour: int) -> float:
        """Get the reward for the last time step.
        
        Reward = -(energy cost) - starvation_penalty
        
        Designed for future RL integration.
        """
        if not self.hourly_log:
            return 0.0
        last = self.hourly_log[-1]
        return -(last["net_cost"] + last["penalty"])
    
    def __repr__(self) -> str:
        return (
            f"EnergyAgent(id={self.agent_id}, type={self.agent_type}, "
            f"battery={self.battery}, cost={self.total_cost:.3f}$)"
        )
