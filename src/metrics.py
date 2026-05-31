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
    total_traded_kwh: float         # Total energy traded in P2P
    n_trades: int                   # Number of individual trades
    total_unmet_demand: float       # kWh of demand unmet after grid fallback (0 if grid_fallback=True)
    total_curtailed: float          # kWh of surplus curtailed after grid fallback (0 if grid_fallback=True)
    n_buyers: int                   # Number of agents that submitted buy orders
    n_sellers: int                  # Number of agents that submitted sell orders
    total_grid_purchased: float = 0.0  # kWh bought from grid at ToU
    total_grid_sold: float = 0.0       # kWh sold to grid at FiT


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
        self.agent_grid_purchased: Dict[int, float] = {i: 0.0 for i in range(n_agents)}
        self.agent_grid_sold: Dict[int, float] = {i: 0.0 for i in range(n_agents)}
        # Gelir/maliyet ayrıştırması (hocanın istediği)
        self.agent_p2p_sell_income: Dict[int, float] = {i: 0.0 for i in range(n_agents)}
        self.agent_p2p_buy_cost: Dict[int, float] = {i: 0.0 for i in range(n_agents)}
        self.agent_grid_sell_income: Dict[int, float] = {i: 0.0 for i in range(n_agents)}
        self.agent_grid_buy_cost: Dict[int, float] = {i: 0.0 for i in range(n_agents)}
    
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
        grid_purchased: float = 0.0,
        grid_sold: float = 0.0,
        p2p_sell_income: float = 0.0,
        p2p_buy_cost: float = 0.0,
        grid_sell_income: float = 0.0,
        grid_buy_cost: float = 0.0,
    ):
        """Record per-agent results for one hour."""
        self.agent_total_cost[agent_id] += cost
        self.agent_total_bought[agent_id] += bought
        self.agent_total_sold[agent_id] += sold
        self.agent_total_unmet[agent_id] += unmet
        self.agent_total_curtailed[agent_id] += curtailed
        self.agent_grid_purchased[agent_id] += grid_purchased
        self.agent_grid_sold[agent_id] += grid_sold
        self.agent_p2p_sell_income[agent_id] += p2p_sell_income
        self.agent_p2p_buy_cost[agent_id] += p2p_buy_cost
        self.agent_grid_sell_income[agent_id] += grid_sell_income
        self.agent_grid_buy_cost[agent_id] += grid_buy_cost
    
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

    def total_grid_purchased(self) -> float:
        """Total kWh purchased from grid at ToU (grid fallback)."""
        return sum(h.total_grid_purchased for h in self.hourly)

    def total_grid_sold(self) -> float:
        """Total kWh sold to grid at FiT (grid fallback)."""
        return sum(h.total_grid_sold for h in self.hourly)

    def p2p_ratio(self) -> float:
        """Fraction of total energy flow handled by P2P (vs grid fallback).

        P2P ratio = P2P_volume / (P2P_volume + grid_purchased + grid_sold)
        Higher = more efficient P2P market.
        """
        p2p = self.total_internal_trading()
        grid = self.total_grid_purchased() + self.total_grid_sold()
        total = p2p + grid
        return p2p / total if total > 0 else 0.0
    
    def total_seller_revenue(self, n_producers: int) -> float:
        """Toplam satıcı geliri = P2P geliri + grid FiT geliri (üretici ajanlar)."""
        return sum(
            self.agent_p2p_sell_income[i] + self.agent_grid_sell_income[i]
            for i in range(n_producers)
        )

    def total_buyer_cost(self, n_producers: int) -> float:
        """Toplam alıcı maliyeti = P2P alım maliyeti + grid ToU maliyeti (tüketici ajanlar)."""
        return sum(
            self.agent_p2p_buy_cost[i] + self.agent_grid_buy_cost[i]
            for i in range(n_producers, self.n_agents)
        )

    def total_p2p_sell_income(self, n_producers: int) -> float:
        return sum(self.agent_p2p_sell_income[i] for i in range(n_producers))

    def total_grid_sell_income(self, n_producers: int) -> float:
        return sum(self.agent_grid_sell_income[i] for i in range(n_producers))

    def total_p2p_buy_cost(self, n_producers: int) -> float:
        return sum(self.agent_p2p_buy_cost[i] for i in range(n_producers, self.n_agents))

    def total_grid_buy_cost(self, n_producers: int) -> float:
        return sum(self.agent_grid_buy_cost[i] for i in range(n_producers, self.n_agents))

    def get_agent_bills(self) -> Dict[int, float]:
        """Get final energy bills for all agents."""
        return dict(self.agent_total_cost)
    
    def summary(self) -> str:
        """Generate a text summary of the simulation results."""
        grid_p = self.total_grid_purchased()
        grid_s = self.total_grid_sold()
        lines = [
            "=" * 60,
            "       SIMULATION RESULTS SUMMARY",
            "=" * 60,
            "",
            f"  Total hours simulated:       {len(self.hourly)}",
            f"  P2P Trading:                 {self.total_internal_trading():.2f} kWh",
            f"  Grid Purchased (ToU):        {grid_p:.2f} kWh",
            f"  Grid Sold (FiT):             {grid_s:.2f} kWh",
            f"  P2P Ratio:                   {self.p2p_ratio()*100:.1f}%",
            f"  Average P2P Price:           ${self.average_clearing_price():.4f}/kWh",
            "",
            "  --- Agent Energy Bills (P2P + Grid) ---",
        ]

        for agent_id in sorted(self.agent_total_cost.keys()):
            cost    = self.agent_total_cost[agent_id]
            bought  = self.agent_total_bought[agent_id]
            sold    = self.agent_total_sold[agent_id]
            g_buy   = self.agent_grid_purchased[agent_id]
            g_sell  = self.agent_grid_sold[agent_id]
            lines.append(
                f"    Agent {agent_id}: "
                f"Bill=${cost:.4f}, "
                f"P2P Buy={bought:.2f}kWh, P2P Sell={sold:.2f}kWh, "
                f"Grid Buy={g_buy:.2f}kWh, Grid Sell={g_sell:.2f}kWh"
            )
        
        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def summary_welfare(self, n_producers: int) -> str:
        """Hocanın istediği: satıcı geliri ve alıcı maliyet ayrıştırması."""
        p2p_sell  = self.total_p2p_sell_income(n_producers)
        grid_sell = self.total_grid_sell_income(n_producers)
        p2p_buy   = self.total_p2p_buy_cost(n_producers)
        grid_buy  = self.total_grid_buy_cost(n_producers)
        tot_sell  = p2p_sell + grid_sell
        tot_buy   = p2p_buy + grid_buy

        lines = [
            "=" * 60,
            "  SATICI GELİRİ & ALICI MALİYET ANALİZİ",
            "=" * 60,
            "",
            f"  --- Satıcı Geliri (Producer agents 0-{n_producers-1}) ---",
            f"  P2P Geliri:           ${p2p_sell:>10.4f}",
            f"  Grid FiT Geliri:      ${grid_sell:>10.4f}",
            f"  TOPLAM SATICI GELİRİ: ${tot_sell:>10.4f}",
            "",
            f"  --- Alıcı Maliyeti (Consumer agents {n_producers}-{self.n_agents-1}) ---",
            f"  P2P Alım Maliyeti:    ${p2p_buy:>10.4f}",
            f"  Grid ToU Maliyeti:    ${grid_buy:>10.4f}",
            f"  TOPLAM ALICI MALİYETİ:${tot_buy:>10.4f}",
            "",
            f"  P2P Zero-Sum Kontrolü (P2P gelir - P2P maliyet): ${p2p_sell - p2p_buy:.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)
