"""
Simulation configuration and hyperparameters.

Combines parameters from:
- Qiu et al. (IJCAI-21): ES operating parameters, time structure
- ADD Bitirme: MB/MC curve parameters, starvation thresholds
"""

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np


@dataclass
class BatteryConfig:
    """Energy Storage (ES) operating parameters (Table 1 from paper)."""
    capacity_min: float = 2.0       # E_es_min (kWh)
    capacity_max: float = 10.0      # E_es_max (kWh)
    power_max: float = 2.0          # P_es_max (kW)
    eta_charge: float = 0.95        # η_esc - charging efficiency
    eta_discharge: float = 0.95     # η_esd - discharging efficiency


@dataclass
class AgentMBParams:
    """Marginal Benefit curve: MB(q) = alpha * exp(-beta * q)
    
    Used by buyers to determine their willingness to pay.
    alpha: base value of the good (higher = more valuable energy)
    beta: rate of satiation (higher = utility drops faster with quantity)
    """
    alpha: float = 0.20   # $/kWh base valuation
    beta: float = 0.3     # satiation rate


@dataclass
class AgentMCParams:
    """Marginal Cost curve: MC(q) = gamma * q + delta
    
    Used by sellers to determine their minimum ask price.
    gamma: production difficulty (higher = cost rises faster)
    delta: base cost (minimum cost of first unit)
    """
    gamma: float = 0.02   # $/kWh per unit
    delta: float = 0.04   # $/kWh base cost


@dataclass
class SimConfig:
    """Master simulation configuration."""
    
    # --- Agent population ---
    n_producers: int = 4        # Agents with PV + battery + load
    n_consumers: int = 4        # Agents with only load + battery
    
    # --- Time structure ---
    t_hours: int = 48           # Simulation day length (steps)
    delta_t: float = 0.5        # Time step (hours)
    
    # --- Battery defaults ---
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    
    # --- Starvation prevention ---
    starvation_threshold: float = 0.5       # kWh - below this → emergency bid
    emergency_price_multiplier: float = 3.0  # Emergency bid = MB * this multiplier
    starvation_penalty: float = 10.0         # Reward penalty for unmet demand
    
    # --- MB/MC parameters per agent (overridable) ---
    # Producer MB params (when buying)
    producer_mb: AgentMBParams = field(default_factory=lambda: AgentMBParams(
        alpha=0.18, beta=0.25
    ))
    # Producer MC params (when selling)
    producer_mc: AgentMCParams = field(default_factory=lambda: AgentMCParams(
        gamma=0.03, delta=0.08
    ))
    # Consumer MB params (when buying — typically higher willingness to pay)
    consumer_mb: AgentMBParams = field(default_factory=lambda: AgentMBParams(
        alpha=0.22, beta=0.20
    ))
    # Consumer MC params (consumers rarely sell, but may discharge battery)
    consumer_mc: AgentMCParams = field(default_factory=lambda: AgentMCParams(
        gamma=0.025, delta=0.06
    ))
    
    # --- Battery storage heuristic thresholds ---
    # AI-generated: lower threshold so agents sell sooner instead of over-storing.
    battery_store_threshold: float = 0.35  # If battery SoC < this fraction, prefer storing
    
    # --- Iterative Double Auction parameters ---
    # Pseudocode Step 6: "Tüm agent'lar mb curvelerinin yüzde x altından ilk tekliflerini verirler"
    initial_shout_margin: float = 0.15   # Buyers start at MB*(1-margin), sellers at MC*(1+margin)
    # "agentlar satış fiyatını alfa değerine göre değiştirir (en iyi alış/satışın tam ortası)"
    alpha: float = 0.5                   # Offer convergence speed per round (0=static, 1=jump to midpoint)
    max_auction_rounds: int = 50         # Maximum bidding rounds per hour before market closes
    unit_size: float = 0.5              # kWh per discrete tradeable unit

    # --- Random seed for reproducibility ---
    seed: int = 123

    # --- Seller flat-block pricing (randomness) ---
    # With probability `seller_flat_prob`, a seller prices a random-sized
    # contiguous block of units (starting from unit 1) all at the same flat
    # price — their MC(0) — instead of using the strictly-increasing MC curve
    # for every unit.  This models real-world behaviour where a producer offers
    # a fixed tariff for a batch of kWh.
    #
    # Examples:
    #   seller_flat_prob = 0.0   → always MC-curve pricing (old behaviour)
    #   seller_flat_prob = 1.0   → every seller always uses flat pricing
    #   seller_flat_prob = 0.5   → ~50 % chance per seller per hour
    #
    # Block size is drawn uniformly from
    #   [seller_flat_min_units, min(seller_flat_max_units, n_seller_units)]
    seller_flat_prob: float = 0.5        # 0 = never flat, 1 = always flat
    seller_flat_min_units: int = 2       # min contiguous flat-priced units
    seller_flat_max_units: int = 5       # max contiguous flat-priced units

    # --- Buyer flat-block pricing (randomness) ---
    # Symmetric to seller_flat: with probability `buyer_flat_prob`, a consumer
    # bids the same flat price — their MB(0) — for a contiguous prefix of units
    # instead of using the diminishing MB curve.  This models a buyer who has a
    # strong, inelastic need for the first few kWh (e.g. must-have appliances).
    buyer_flat_prob: float = 0.0         # 0 = never flat, 1 = always flat (DISABLED)
    buyer_flat_min_units: int = 2        # min contiguous flat-bid units
    buyer_flat_max_units: int = 5        # max contiguous flat-bid units

    # --- Data Paths ---
    mb_data_path: str = "src/data/mb_hourly_params.json"

    @property
    def n_agents(self) -> int:
        return self.n_producers + self.n_consumers
