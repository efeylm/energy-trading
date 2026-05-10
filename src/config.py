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
class AgentMBParams:
    """Marginal Benefit curve parameters."""
    alpha: float = 0.20
    beta: float = 0.3


@dataclass
class AgentMCParams:
    """Marginal Cost curve parameters."""
    gamma: float = 0.00
    delta: float = 0.04


@dataclass
class SimConfig:
    """Master simulation configuration."""
    
    # --- Agent population ---
    n_producers: int = 4        # Agents with PV + load
    n_consumers: int = 4        # Agents with only load
    
    # --- Time structure ---
    t_hours: int = 48           # Simulation day length (steps)
    delta_t: float = 0.5        # Time step (hours)
    
    # --- MB/MC parameters per agent (overridable) ---
    # Producer MB params (when buying)
    producer_mb: AgentMBParams = field(default_factory=lambda: AgentMBParams(
        alpha=0.18, beta=0.25
    ))
    # Producer MC params (when selling)
    producer_mc: AgentMCParams = field(default_factory=lambda: AgentMCParams(
        gamma=0.03, delta=0.08
    ))
    # Consumer MB params (when buying)
    consumer_mb: AgentMBParams = field(default_factory=lambda: AgentMBParams(
        alpha=0.22, beta=0.20
    ))
    # Consumer MC params (rarely used if no battery, but kept for consistency)
    consumer_mc: AgentMCParams = field(default_factory=lambda: AgentMCParams(
        gamma=0.025, delta=0.06
    ))
    
    # --- Iterative Double Auction parameters ---
    initial_shout_margin: float = 0.15   # Buyers start at MB*(1-margin), sellers at MC*(1+margin)
    alpha: float = 0.5                   # Offer convergence speed per round
    max_auction_rounds: int = 50         # Maximum bidding rounds per hour
    unit_size: float = 0.5              # kWh per discrete tradeable unit

    # --- Random seed for reproducibility ---
    seed: int = 1

    # --- Seller flat-block pricing (randomness) ---
    seller_flat_prob: float = 0.5        
    seller_flat_min_units: int = 2       
    seller_flat_max_units: int = 5       

    # --- Buyer flat-block pricing (randomness) ---
    buyer_flat_prob: float = 0.0         

    # --- Data Paths ---
    mb_data_path: str = "src/data/mb_hourly_params.json"

    @property
    def n_agents(self) -> int:
        return self.n_producers + self.n_consumers
