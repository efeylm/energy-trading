"""
Metrics collector and analysis for the P2P energy trading simulation.

Tracks per-hour and per-agent metrics including:
- Market clearing prices and volumes
- Agent energy bills
- Battery utilization  
- Starvation / curtailment rates
- Allocative efficiency
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


@dataclass
class HourlyMetrics:
    """Metrics collected at the end of each auction period (hour)."""
    hour: int
    average_price: float            # $/kWh average clearing price
    total_traded_kwh: float         # Total energy traded in DA
    n_trades: int                   # Number of individual trades
    total_unmet_demand: float       # kWh of demand that couldn't be met
    total_curtailed: float          # kWh of surplus that was curtailed
    n_buyers: int                   # Number of agents that submitted buy orders
    n_sellers: int                  # Number of agents that submitted sell orders


class MetricsCollector:
    """Collects and aggregates simulation metrics."""
    
    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.hourly: List[HourlyMetrics] = []
        
        # Per-agent cumulative tracking
        self.agent_total_cost: Dict[int, float] = {i: 0.0 for i in range(n_agents)}
        self.agent_total_bought: Dict[int, float] = {i: 0.0 for i in range(n_agents)}
        self.agent_total_sold: Dict[int, float] = {i: 0.0 for i in range(n_agents)}
        self.agent_total_unmet: Dict[int, float] = {i: 0.0 for i in range(n_agents)}
        self.agent_total_curtailed: Dict[int, float] = {i: 0.0 for i in range(n_agents)}
    
    def record_hour(self, metrics: HourlyMetrics):
        """Record metrics for one hour."""
        self.hourly.append(metrics)
    
    def record_agent_hour(
        self,
        agent_id: int,
        cost: float,
        bought: float,
        sold: float,
        unmet: float,
        curtailed: float,
    ):
        """Record per-agent results for one hour."""
        self.agent_total_cost[agent_id] += cost
        self.agent_total_bought[agent_id] += bought
        self.agent_total_sold[agent_id] += sold
        self.agent_total_unmet[agent_id] += unmet
        self.agent_total_curtailed[agent_id] += curtailed
    
    # ---- Aggregate metrics ----
    
    def get_hourly_prices(self) -> np.ndarray:
        """Array of average clearing prices per hour."""
        return np.array([h.average_price for h in self.hourly])
    
    def get_hourly_volumes(self) -> np.ndarray:
        """Array of total traded volumes per hour."""
        return np.array([h.total_traded_kwh for h in self.hourly])
    
    def get_hourly_unmet(self) -> np.ndarray:
        """Array of total unmet demand per hour."""
        return np.array([h.total_unmet_demand for h in self.hourly])
    
    def get_hourly_curtailed(self) -> np.ndarray:
        """Array of total curtailed surplus per hour."""
        return np.array([h.total_curtailed for h in self.hourly])
    
    def total_internal_trading(self) -> float:
        """Total kWh traded through P2P market across all hours."""
        return sum(h.total_traded_kwh for h in self.hourly)
    
    def total_unmet_demand(self) -> float:
        """Total kWh of demand that couldn't be satisfied."""
        return sum(h.total_unmet_demand for h in self.hourly)
    
    def total_curtailment(self) -> float:
        """Total kWh of surplus energy that was curtailed."""
        return sum(h.total_curtailed for h in self.hourly)
    
    def average_clearing_price(self) -> float:
        """Volume-weighted average clearing price across all hours."""
        total_value = sum(h.average_price * h.total_traded_kwh for h in self.hourly)
        total_volume = self.total_internal_trading()
        if total_volume == 0:
            return 0.0
        return total_value / total_volume
    
    def get_agent_bills(self) -> Dict[int, float]:
        """Get final energy bills for all agents."""
        return dict(self.agent_total_cost)
    
    def summary(self) -> str:
        """Generate a text summary of the simulation results."""
        lines = [
            "=" * 60,
            "       SIMULATION RESULTS SUMMARY",
            "=" * 60,
            "",
            f"  Total hours simulated:       {len(self.hourly)}",
            f"  Total internal trading:      {self.total_internal_trading():.2f} kWh",
            f"  Total unmet demand:          {self.total_unmet_demand():.2f} kWh",
            f"  Total curtailment:           {self.total_curtailment():.2f} kWh",
            f"  Average clearing price:      ${self.average_clearing_price():.4f}/kWh",
            "",
            "  --- Agent Energy Bills ---",
        ]
        
        for agent_id in sorted(self.agent_total_cost.keys()):
            cost = self.agent_total_cost[agent_id]
            bought = self.agent_total_bought[agent_id]
            sold = self.agent_total_sold[agent_id]
            unmet = self.agent_total_unmet[agent_id]
            lines.append(
                f"    Agent {agent_id}: "
                f"Bill=${cost:.4f}, "
                f"Bought={bought:.2f}kWh, "
                f"Sold={sold:.2f}kWh, "
                f"Unmet={unmet:.2f}kWh"
            )
        
        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)
