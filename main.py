"""
Main entry point for the P2P Energy Trading Simulation.

Runs a 24-hour simulation day and generates visualization plots:
1. Hourly average clearing price
2. Hourly trading volume
3. Battery SoC timeseries for all agents
4. Agent energy bills (bar chart)
5. Hourly unmet demand and curtailment
6. PV generation and load demand profiles
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures

from src.config import SimConfig
from src.environment import EnergyTradingEnv
from src.metrics import MetricsCollector


def plot_results(metrics: MetricsCollector, env: EnergyTradingEnv, save_dir: str = "."):
    """Generate comprehensive visualization of simulation results."""
    
    hours = np.arange(len(metrics.hourly))
    n_prosumers = env.config.n_prosumers
    n_agents = env.config.n_agents
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("P2P Energy Trading — POMG Simulation Results", fontsize=16, fontweight='bold')
    
    # --- 1. Hourly Average Clearing Price ---
    ax = axes[0, 0]
    prices = metrics.get_hourly_prices()
    # Replace zero prices (no-trade hours) with NaN for display
    prices_display = np.where(prices > 0, prices, np.nan)
    ax.plot(hours, prices_display, 'b-o', markersize=4, linewidth=1.5, label='Clearing Price')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Price ($/kWh)')
    ax.set_title('Hourly Average Clearing Price')
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # --- 2. Hourly Trading Volume ---
    ax = axes[0, 1]
    volumes = metrics.get_hourly_volumes()
    unmet = metrics.get_hourly_unmet()
    curtailed = metrics.get_hourly_curtailed()
    
    ax.bar(hours - 0.2, volumes, width=0.4, color='steelblue', label='Traded', alpha=0.8)
    ax.bar(hours + 0.2, unmet, width=0.4, color='crimson', label='Unmet Demand', alpha=0.8)
    ax.bar(hours + 0.2, curtailed, width=0.4, bottom=unmet, color='orange', 
           label='Curtailed', alpha=0.8)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Energy (kWh)')
    ax.set_title('Hourly Trading Volume & Losses')
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # --- 3. Battery SoC Timeseries ---
    ax = axes[1, 0]
    colors_prosumer = plt.cm.Blues(np.linspace(0.4, 0.9, n_prosumers))
    colors_consumer = plt.cm.Reds(np.linspace(0.4, 0.9, n_agents - n_prosumers))
    
    for i in range(n_agents):
        soc = metrics.get_battery_soc_timeseries(i)
        if i < n_prosumers:
            ax.plot(hours, soc, color=colors_prosumer[i], linewidth=1.5, 
                    label=f'P{i+1}', linestyle='-')
        else:
            ax.plot(hours, soc, color=colors_consumer[i - n_prosumers], linewidth=1.5,
                    label=f'C{i - n_prosumers + 1}', linestyle='--')
    
    ax.axhline(y=0.2, color='gray', linestyle=':', alpha=0.5, label='Starvation zone')
    ax.set_xlabel('Hour')
    ax.set_ylabel('State of Charge (%)')
    ax.set_title('Battery SoC Over Time')
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=3)
    
    # --- 4. Agent Energy Bills ---
    ax = axes[1, 1]
    bills = metrics.get_agent_bills()
    agent_ids = sorted(bills.keys())
    bill_values = [bills[i] for i in agent_ids]
    labels = [f'P{i+1}' if i < n_prosumers else f'C{i - n_prosumers + 1}' 
              for i in agent_ids]
    colors = ['steelblue' if i < n_prosumers else 'crimson' for i in agent_ids]
    
    bars = ax.bar(labels, bill_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Agent')
    ax.set_ylabel('Net Energy Cost ($)')
    ax.set_title('Agent Energy Bills (negative = income)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    # Add value labels on bars
    for bar, val in zip(bars, bill_values):
        y_pos = bar.get_height() if val >= 0 else bar.get_height() - 0.01
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f'${val:.3f}',
                ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)
    
    # --- 5. PV Generation & Load Profiles ---
    ax = axes[2, 0]
    for i in range(n_agents):
        pv = np.array([env.profiles.get_pv_generation(i, h) for h in range(24)])
        load = np.array([env.profiles.get_load_demand(i, h) for h in range(24)])
        
        if i < n_prosumers:
            if i == 0:
                ax.plot(hours, pv, color='gold', alpha=0.6, linewidth=1, label='PV (Prosumers)')
                ax.plot(hours, load, color='steelblue', alpha=0.4, linewidth=1, label='Load (Prosumers)')
            else:
                ax.plot(hours, pv, color='gold', alpha=0.6, linewidth=1)
                ax.plot(hours, load, color='steelblue', alpha=0.4, linewidth=1)
        else:
            if i == n_prosumers:
                ax.plot(hours, load, color='crimson', alpha=0.4, linewidth=1, label='Load (Consumers)')
            else:
                ax.plot(hours, load, color='crimson', alpha=0.4, linewidth=1)
    
    ax.set_xlabel('Hour')
    ax.set_ylabel('Power (kW)')
    ax.set_title('Daily PV Generation & Load Profiles')
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # --- 6. Cumulative metrics text ---
    ax = axes[2, 1]
    ax.axis('off')
    
    summary_text = (
        f"Total Internal Trading:  {metrics.total_internal_trading():.2f} kWh\n"
        f"Total Unmet Demand:      {metrics.total_unmet_demand():.2f} kWh\n"
        f"Total Curtailment:       {metrics.total_curtailment():.2f} kWh\n"
        f"Avg Clearing Price:      ${metrics.average_clearing_price():.4f}/kWh\n"
        f"Survival Rate:           {metrics.survival_rate():.1%}\n"
        f"\n"
        f"Total Agent Bills:\n"
    )
    for i in agent_ids:
        label = f'P{i+1}' if i < n_prosumers else f'C{i - n_prosumers + 1}'
        summary_text += f"  {label}: ${bills[i]:.4f}\n"
    
    total_bills = sum(bills.values())
    summary_text += f"\n  TOTAL: ${total_bills:.4f}"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title('Summary Statistics')
    
    plt.tight_layout()
    filepath = f"{save_dir}/simulation_results.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {filepath}")
    return filepath


def main():
    """Run the P2P energy trading simulation."""
    
    # Create configuration
    config = SimConfig()
    
    # Create environment
    env = EnergyTradingEnv(config)
    
    # Run simulation
    metrics = env.run_day()
    
    # Generate plots
    plot_results(metrics, env, save_dir=".")
    
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
