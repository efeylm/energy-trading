"""
Synthetic energy profiles for prosumers and consumers.

Generates daily PV generation and load demand profiles inspired by the
Australian Ausgrid dataset used in Qiu et al. (IJCAI-21).

PV Generation: Gaussian bell curve peaking at solar noon (hours 10-14).
Load Demand: Double-peaked profile (morning 7-9, evening 17-20).
Initial Battery: Truncated normal distribution around mid-SoC.
"""

import numpy as np
from typing import List, Tuple


def _gaussian(x: np.ndarray, mu: float, sigma: float, amplitude: float) -> np.ndarray:
    """Gaussian function for profile shaping."""
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def generate_pv_profile(
    agent_id: int,
    peak_kw: float = 5.0,
    noise_std: float = 0.3,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Generate 24-hour PV generation profile for a prosumer.
    
    Args:
        agent_id: Used to create slight variation between agents.
        peak_kw: Peak generation power (kW) at solar noon.
        noise_std: Standard deviation of random noise.
        rng: Random number generator.
    
    Returns:
        Array of shape (24,) with PV generation in kW for each hour.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    hours = np.arange(24, dtype=float)
    
    # Solar peak varies slightly per agent (between 11.5 and 12.5)
    peak_hour = 12.0 + (agent_id % 4 - 1.5) * 0.3
    
    # Peak power varies per agent (±20%)
    agent_peak = peak_kw * (0.8 + 0.1 * (agent_id % 4))
    
    # Gaussian bell curve for solar generation
    profile = _gaussian(hours, mu=peak_hour, sigma=2.5, amplitude=agent_peak)
    
    # Zero out nighttime (before 6, after 19)
    profile[hours < 6] = 0.0
    profile[hours > 19] = 0.0
    
    # Add small positive noise (generation can't be negative)
    noise = rng.normal(0, noise_std, size=24)
    profile = np.maximum(0.0, profile + noise)
    
    return profile


def generate_load_profile(
    agent_id: int,
    base_kw: float = 2.0,
    peak_kw: float = 6.0,
    noise_std: float = 0.4,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Generate 24-hour load demand profile.
    
    Double-peaked profile: morning rush (7-9) and evening peak (17-20).
    
    Args:
        agent_id: Used to create variation between agents.
        base_kw: Baseline load (always-on devices).
        peak_kw: Peak demand during rush hours.
        noise_std: Standard deviation of random noise.
        rng: Random number generator.
    
    Returns:
        Array of shape (24,) with load demand in kW for each hour.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    hours = np.arange(24, dtype=float)
    
    # Agent-specific variation
    agent_factor = 0.8 + 0.15 * (agent_id % 8)
    
    # Base load (always present)
    profile = np.full(24, base_kw * agent_factor)
    
    # Morning peak (7-9)
    morning = _gaussian(hours, mu=8.0, sigma=1.2, amplitude=peak_kw * 0.6 * agent_factor)
    
    # Evening peak (17-20) — typically stronger
    evening = _gaussian(hours, mu=18.5, sigma=1.5, amplitude=peak_kw * agent_factor)
    
    # Midday bump (12-14) — smaller
    midday = _gaussian(hours, mu=13.0, sigma=1.0, amplitude=peak_kw * 0.3 * agent_factor)
    
    profile += morning + evening + midday
    
    # Add noise (demand can't be negative)
    noise = rng.normal(0, noise_std, size=24)
    profile = np.maximum(0.5, profile + noise)
    
    return profile


def generate_initial_battery_energy(
    agent_id: int,
    capacity_min: float,
    capacity_max: float,
    rng: np.random.Generator = None,
) -> float:
    """Generate initial battery energy level.
    
    Truncated normal around 50% SoC with some agent variation.
    
    Returns:
        Initial energy in kWh, clamped to [capacity_min, capacity_max].
    """
    if rng is None:
        rng = np.random.default_rng()
    
    mid_point = (capacity_min + capacity_max) / 2.0
    spread = (capacity_max - capacity_min) * 0.2  # ±20% of range
    
    # Slight agent variation
    offset = (agent_id % 4 - 1.5) * 0.5
    
    energy = rng.normal(mid_point + offset, spread)
    return float(np.clip(energy, capacity_min, capacity_max))


class ProfileManager:
    """Manages all agent profiles for a simulation day."""
    
    def __init__(
        self,
        n_prosumers: int,
        n_consumers: int,
        battery_config,
        seed: int = 42,
    ):
        self.n_prosumers = n_prosumers
        self.n_consumers = n_consumers
        self.n_agents = n_prosumers + n_consumers
        self.battery_config = battery_config
        self.rng = np.random.default_rng(seed)
        
        # Generate profiles
        self.pv_profiles = {}    # agent_id -> (24,) array
        self.load_profiles = {}  # agent_id -> (24,) array
        self.initial_battery = {}  # agent_id -> float
        
        self._generate_all_profiles()
    
    def _generate_all_profiles(self):
        """Generate profiles for all agents."""
        for i in range(self.n_agents):
            # PV generation: only prosumers have PV panels
            if i < self.n_prosumers:
                self.pv_profiles[i] = generate_pv_profile(i, rng=self.rng)
            else:
                self.pv_profiles[i] = np.zeros(24)  # Consumers have no PV
            
            # Load demand: all agents have load
            self.load_profiles[i] = generate_load_profile(i, rng=self.rng)
            
            # Initial battery energy
            self.initial_battery[i] = generate_initial_battery_energy(
                i, 
                self.battery_config.capacity_min,
                self.battery_config.capacity_max,
                rng=self.rng,
            )
    
    def get_pv_generation(self, agent_id: int, hour: int) -> float:
        """Get PV generation (kW) for agent at given hour."""
        return float(self.pv_profiles[agent_id][hour])
    
    def get_load_demand(self, agent_id: int, hour: int) -> float:
        """Get load demand (kW) for agent at given hour."""
        return float(self.load_profiles[agent_id][hour])
    
    def get_initial_battery(self, agent_id: int) -> float:
        """Get initial battery energy (kWh) for agent."""
        return self.initial_battery[agent_id]
    
    def get_inflexible_load(self, agent_id: int, hour: int) -> float:
        """Get inflexible (net) load: demand - PV generation.
        
        Positive = needs energy (net consumer)
        Negative = has surplus energy (net producer)
        """
        return self.get_load_demand(agent_id, hour) - self.get_pv_generation(agent_id, hour)
    
    def summary(self) -> str:
        """Print summary of generated profiles."""
        lines = ["=== Profile Summary ==="]
        for i in range(self.n_agents):
            agent_type = "Prosumer" if i < self.n_prosumers else "Consumer"
            total_pv = self.pv_profiles[i].sum()
            total_load = self.load_profiles[i].sum()
            init_bat = self.initial_battery[i]
            lines.append(
                f"  Agent {i} ({agent_type}): "
                f"PV={total_pv:.1f}kWh, Load={total_load:.1f}kWh, "
                f"Net={total_pv - total_load:.1f}kWh, "
                f"Battery₀={init_bat:.1f}kWh"
            )
        return "\n".join(lines)
