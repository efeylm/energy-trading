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
    n_prosumers: int = 4        # Agents with PV + battery + load
    n_consumers: int = 4        # Agents with only load + battery
    
    # --- Time structure ---
    t_hours: int = 24           # Simulation day length (hours)
    delta_t: float = 1.0        # Time step (hours)
    
    # --- Battery defaults ---
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    
    # --- Starvation prevention ---
    starvation_threshold: float = 0.5       # kWh - below this → emergency bid
    emergency_price_multiplier: float = 3.0  # Emergency bid = MB * this multiplier
    starvation_penalty: float = 10.0         # Reward penalty for unmet demand
    
    # --- MB/MC parameters per agent (overridable) ---
    # Prosumer MB params (when buying)
    prosumer_mb: AgentMBParams = field(default_factory=lambda: AgentMBParams(
        alpha=0.18, beta=0.25
    ))
    # Prosumer MC params (when selling)
    prosumer_mc: AgentMCParams = field(default_factory=lambda: AgentMCParams(
        gamma=0.015, delta=0.04
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
    battery_store_threshold: float = 0.5  # If battery SoC < this fraction, prefer storing
    
    # --- Random seed for reproducibility ---
    seed: int = 42

    @property
    def n_agents(self) -> int:
        return self.n_prosumers + self.n_consumers
