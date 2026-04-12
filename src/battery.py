"""
Energy Storage (Battery) model.

Implements equations (1), (2), (3) from Qiu et al. (IJCAI-21):
    C_n,t = min(power, (E_max - E_current) / (eta_c * dt))        -- eq(1)
    D_n,t = min(power, (E_current - E_min) * eta_d / dt)           -- eq(2)
    E_n,t+1 = E_n,t + C_n,t * eta_c * dt + D_n,t * dt / eta_d     -- eq(3)

Note: In eq(3), charging adds energy and discharging removes energy.
We store D as a positive quantity internally but subtract it from SoC.
"""

from dataclasses import dataclass


@dataclass
class BatteryState:
    """Snapshot of battery state for observation."""
    current_energy: float   # kWh
    capacity_min: float
    capacity_max: float
    soc_fraction: float     # (current - min) / (max - min)


class Battery:
    """Energy Storage System for a single agent."""
    
    def __init__(
        self,
        capacity_min: float,
        capacity_max: float,
        power_max: float,
        eta_charge: float,
        eta_discharge: float,
        initial_energy: float,
    ):
        self.capacity_min = capacity_min
        self.capacity_max = capacity_max
        self.power_max = power_max
        self.eta_charge = eta_charge
        self.eta_discharge = eta_discharge
        self.energy = initial_energy
        
        # Clamp initial energy
        self.energy = max(self.capacity_min, min(self.capacity_max, self.energy))
    
    @property
    def soc_fraction(self) -> float:
        """State of Charge as fraction [0, 1]."""
        if self.capacity_max == self.capacity_min:
            return 0.0
        return (self.energy - self.capacity_min) / (self.capacity_max - self.capacity_min)
    
    def max_charge_power(self, dt: float) -> float:
        """Maximum power (kW) that can be charged this timestep.
        
        Equation (1): C_n,t = min(P_max, (E_max - E_current) / (eta_c * dt))
        """
        available_capacity = self.capacity_max - self.energy
        if available_capacity <= 0:
            return 0.0
        max_from_capacity = available_capacity / (self.eta_charge * dt)
        return min(self.power_max, max_from_capacity)
    
    def max_discharge_power(self, dt: float) -> float:
        """Maximum power (kW) that can be discharged this timestep.
        
        Equation (2): D_n,t = min(P_max, (E_current - E_min) * eta_d / dt)
        """
        available_energy = self.energy - self.capacity_min
        if available_energy <= 0:
            return 0.0
        max_from_energy = available_energy * self.eta_discharge / dt
        return min(self.power_max, max_from_energy)
    
    def charge(self, power: float, dt: float) -> float:
        """Charge the battery at given power for dt hours.
        
        Returns the actual power charged (may be capped by capacity).
        """
        max_power = self.max_charge_power(dt)
        actual_power = min(abs(power), max_power)
        # Energy gained = power * eta * dt
        energy_added = actual_power * self.eta_charge * dt
        self.energy += energy_added
        self.energy = min(self.energy, self.capacity_max)
        return actual_power
    
    def discharge(self, power: float, dt: float) -> float:
        """Discharge the battery at given power for dt hours.
        
        Returns the actual power discharged (may be capped by available energy).
        """
        max_power = self.max_discharge_power(dt)
        actual_power = min(abs(power), max_power)
        # Energy removed = power * dt / eta
        energy_removed = actual_power * dt / self.eta_discharge
        self.energy -= energy_removed
        self.energy = max(self.energy, self.capacity_min)
        return actual_power
    
    def get_state(self) -> BatteryState:
        """Return current battery state for observation."""
        return BatteryState(
            current_energy=self.energy,
            capacity_min=self.capacity_min,
            capacity_max=self.capacity_max,
            soc_fraction=self.soc_fraction,
        )
    
    def __repr__(self) -> str:
        return (
            f"Battery(energy={self.energy:.2f}kWh, "
            f"SoC={self.soc_fraction:.1%}, "
            f"range=[{self.capacity_min}, {self.capacity_max}])"
        )
