"""
Main entry point for the P2P Energy Trading Simulation.

Runs a 24-hour simulation day and generates visualization plots:
1. Hourly average clearing price
2. Hourly trading volume
3. Battery SoC timeseries for all agents (bu artık yok battery sistemi kaldırıldı)
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
    
    hours = np.arange(len(metrics.hourly)) * env.config.delta_t
    n_producers = env.config.n_producers
    n_agents = env.config.n_agents
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    fig.suptitle("P2P Energy Trading — POMG Simulation Results", fontsize=16, fontweight='bold')
    
    # --- 1. Hourly Average Clearing Price ---
    ax = axes[0, 0]
    prices = metrics.get_hourly_prices()
    # Replace zero prices (no-trade hours) with NaN for display
    prices_display = np.where(prices > 0, prices, np.nan)
    ax.plot(hours, prices_display, 'b-o', markersize=4, linewidth=1.5, label='Clearing Price')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Price ($/kWh)')
    ax.set_title('Hourly Average Clearing Price')
    ax.set_xlim(-0.5, 24.0)
    ax.set_xticks(range(0, 25, 2))
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # --- 2. Hourly Trading Volume ---
    ax = axes[0, 1]
    volumes = metrics.get_hourly_volumes()
    unmet = metrics.get_hourly_unmet()
    curtailed = metrics.get_hourly_curtailed()
    
    ax.bar(hours - 0.1, volumes, width=0.2, color='steelblue', label='Traded', alpha=0.8)
    ax.bar(hours + 0.1, unmet, width=0.2, color='crimson', label='Unmet Demand', alpha=0.8)
    ax.bar(hours + 0.1, curtailed, width=0.2, bottom=unmet, color='orange', 
           label='Curtailed', alpha=0.8)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Energy (kWh)')
    ax.set_title('Hourly Trading Volume & Losses')
    ax.set_xlim(-0.5, 24.0)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # --- 3. PV Generation & Load Profiles ---
    ax = axes[1, 0]
    for i in range(n_agents):
        pv = np.array([env.profiles.get_pv_generation(i, h) for h in range(env.config.t_hours)])
        load = np.array([env.profiles.get_load_demand(i, h) for h in range(env.config.t_hours)])

        if i < n_producers:
            if i == 0:
                ax.plot(hours, pv,   color='gold',      alpha=0.6, linewidth=1, label='PV (Producers)')
                ax.plot(hours, load, color='steelblue', alpha=0.4, linewidth=1, label='Load (Producers)')
            else:
                ax.plot(hours, pv,   color='gold',      alpha=0.6, linewidth=1)
                ax.plot(hours, load, color='steelblue', alpha=0.4, linewidth=1)
        else:
            if i == n_producers:
                ax.plot(hours, load, color='crimson', alpha=0.4, linewidth=1, label='Load (Consumers)')
            else:
                ax.plot(hours, load, color='crimson', alpha=0.4, linewidth=1)
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Power (kW)')
    ax.set_title('Daily PV Generation & Load Profiles')
    ax.set_xlim(-0.5, 24.0)
    ax.set_xticks(range(0, 25, 2))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # --- 4. Agent Energy Bills ---
    ax = axes[1, 1]
    bills = metrics.get_agent_bills()
    agent_ids = sorted(bills.keys())
    bill_values = [bills[i] for i in agent_ids]
    labels = [f'Pr{i+1}' if i < n_producers else f'C{i - n_producers + 1}'
              for i in agent_ids]
    colors = ['forestgreen' if i < n_producers else 'crimson' for i in agent_ids]
    
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

    # --- 5. Cumulative metrics text ---
    ax = axes[2, 0]
    ax.axis('off')
    
    summary_text = (
        "===== SIMULATION SUMMARY =====\n\n"
        f"Total Trading:  {metrics.total_internal_trading():.2f} kWh\n"
        f"Avg Price:      ${metrics.average_clearing_price():.4f}/kWh\n"
        f"Total Unmet:    {metrics.total_unmet_demand():.2f} kWh\n"
        f"Total Curtail:  {metrics.total_curtailment():.2f} kWh\n\n"
        "Agent Bills:\n"
    )
    for i in agent_ids:
        label = f'Pr{i+1}' if i < n_producers else f'C{i - n_producers + 1}'
        summary_text += f"  {label}: ${bills[i]:.4f}\n"
    
    total_bills = sum(bills.values())
    summary_text += f"\n  TOTAL: ${total_bills:.4f}"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title('Summary Statistics')

    # --- 6. Empty Axis ---
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    filepath = f"{save_dir}/simulation_results.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {filepath}")

    # ------------------------------------------------------------------ #
    # Separate figure: per-agent bid / ask scatter charts                 #
    # ------------------------------------------------------------------ #
    _plot_bids_asks(env, save_dir=save_dir)

    return filepath


def _plot_bids_asks(env, save_dir: str = "."):
    """Generate two separate Price×Quantity charts from env.bid_ask_log.

    Chart 1 — Alici Teklifleri: all buy orders globally sorted across peak hour.
    Chart 2 — Satici Teklifleri: all sell orders globally sorted across peak hour.
    Chart 3 — Combined bids and asks.
    """
    bid_ask_log = env.bid_ask_log   # List[Dict[agent_id, Order]]
    n_agents = env.config.n_agents
    n_producers = env.config.n_producers

    # Distinct colour per agent (tab10 gives 10 well-separated colours)
    cmap = plt.colormaps.get_cmap('tab10')
    agent_colors = {i: cmap(i / max(n_agents, 10)) for i in range(n_agents)}

    def agent_label(i):
        if i < n_producers:
            return f'Pr{i+1} (Producer)'
        return f'C{i - n_producers + 1} (Consumer)'

    # ------------------------------------------------------------------ #
    # Find the single representative hour (most agents with orders)        #
    # ------------------------------------------------------------------ #
    hour_agent_count = {}
    for hour, hour_orders in enumerate(bid_ask_log):
        hour_agent_count[hour] = len(hour_orders)
    peak_hour = max(hour_agent_count, key=hour_agent_count.get)

    # ------------------------------------------------------------------ #
    # Collect all unit-orders for the peak hour                          #
    # ------------------------------------------------------------------ #
    all_buys = []
    all_sells = []
    for agent_id, unit_orders in bid_ask_log[peak_hour].items():
        for u in unit_orders:
            if u.is_buy:
                all_buys.append(u)
            else:
                all_sells.append(u)
                
    # Sort buys descending by price
    all_buys.sort(key=lambda o: o.price, reverse=True)
    # Sort sells ascending by price
    all_sells.sort(key=lambda o: o.price, reverse=False)

    # ------------------------------------------------------------------ #
    # Plot buyer bids                                                    #
    # ------------------------------------------------------------------ #
    fig_b, ax_buy = plt.subplots(figsize=(10, 6))
    fig_b.suptitle(
        f'Alıcı Teklifleri — Fiyat × Kümülatif Miktar  (Saat {peak_hour})\n'
        'Teklifler en yüksekten en düşüğe sıralanmıştır',
        fontsize=13, fontweight='bold')

    cum_x = 0.0
    agents_plotted = set()
    for o in all_buys:
        color = agent_colors[o.agent_id]
        label = f'A{o.agent_id} {agent_label(o.agent_id)}' if o.agent_id not in agents_plotted else None
        agents_plotted.add(o.agent_id)
        ax_buy.plot([cum_x, cum_x + o.quantity], [o.price, o.price], 
                    color=color, linewidth=3.0, alpha=0.9, label=label, solid_capstyle='butt')
        cum_x += o.quantity

    ax_buy.set_xlabel('Kümülatif Miktar (kWh)', fontsize=11)
    ax_buy.set_ylabel('Teklif Fiyatı ($/kWh)',   fontsize=11)
    ax_buy.set_title('Alıcı Teklifleri (Buy Bids)', fontsize=12, fontweight='bold')
    ax_buy.grid(True, alpha=0.3)
    if all_buys:
        ax_buy.legend(fontsize=8, loc='upper right')
    else:
        ax_buy.text(0.5, 0.5, 'Alıcı teklifi yok',
                    transform=ax_buy.transAxes, ha='center', va='center',
                    fontsize=12, color='gray')

    fig_b.tight_layout()
    buy_path = f"{save_dir}/buyer_bids_agents.png"
    fig_b.savefig(buy_path, dpi=150, bbox_inches='tight')
    plt.close(fig_b)
    print(f"Buyer bids plot saved to: {buy_path}")

    # ------------------------------------------------------------------ #
    # Plot seller asks                                                   #
    # ------------------------------------------------------------------ #
    fig_s, ax_sell = plt.subplots(figsize=(10, 6))
    fig_s.suptitle(
        f'Satıcı Teklifleri — Fiyat × Kümülatif Miktar  (Saat {peak_hour})\n'
        'Teklifler en düşükten en yükseğe sıralanmıştır',
        fontsize=13, fontweight='bold')

    cum_x = 0.0
    agents_plotted_sell = set()
    for o in all_sells:
        color = agent_colors[o.agent_id]
        label = f'A{o.agent_id} {agent_label(o.agent_id)}' if o.agent_id not in agents_plotted_sell else None
        agents_plotted_sell.add(o.agent_id)
        ax_sell.plot([cum_x, cum_x + o.quantity], [o.price, o.price], 
                     color=color, linewidth=3.0, alpha=0.9, label=label, solid_capstyle='butt')
        cum_x += o.quantity

    ax_sell.set_xlabel('Kümülatif Miktar (kWh)', fontsize=11)
    ax_sell.set_ylabel('Teklif Fiyatı ($/kWh)',   fontsize=11)
    ax_sell.set_title('Satıcı Teklifleri (Sell Asks)', fontsize=12, fontweight='bold')
    ax_sell.grid(True, alpha=0.3)
    if all_sells:
        ax_sell.legend(fontsize=8, loc='upper left')
    else:
        ax_sell.text(0.5, 0.5, 'Satıcı teklifi yok',
                     transform=ax_sell.transAxes, ha='center', va='center',
                     fontsize=12, color='gray')

    fig_s.tight_layout()
    sell_path = f"{save_dir}/seller_asks_agents.png"
    fig_s.savefig(sell_path, dpi=150, bbox_inches='tight')
    plt.close(fig_s)
    print(f"Seller asks plot saved to: {sell_path}")

    # ------------------------------------------------------------------ #
    # Combined chart                                                     #
    # ------------------------------------------------------------------ #
    fig_c, ax_c = plt.subplots(figsize=(13, 7))
    fig_c.suptitle(
        f'Alıcı & Satıcı Teklifleri — Fiyat × Kümülatif Miktar  (Saat {peak_hour})\n'
        'Tüm çizgiler düz çizilmiştir | Renk = Agent',
        fontsize=13, fontweight='bold')

    cum_x_buy = 0.0
    agents_plotted_combined = set()
    
    # Track legends manually to organize them well
    import matplotlib.lines as mlines
    agent_handles = []
    
    for o in all_buys:
        color = agent_colors[o.agent_id]
        label = f'A{o.agent_id} {agent_label(o.agent_id)}'
        if o.agent_id not in agents_plotted_combined:
            agents_plotted_combined.add(o.agent_id)
            agent_handles.append(mlines.Line2D([], [], color=color, linewidth=2, label=label))
            
        ax_c.plot([cum_x_buy, cum_x_buy + o.quantity], [o.price, o.price], 
                  color=color, linewidth=3.0, alpha=0.9, linestyle='-', solid_capstyle='butt')
        
        # Add small text for agent ID above/below line
        # ax_c.text(cum_x_buy + o.quantity/2, o.price, f"A{o.agent_id}", ha='center', va='bottom', fontsize=7, color=color)
        
        cum_x_buy += o.quantity

    cum_x_sell = 0.0
    for o in all_sells:
        color = agent_colors[o.agent_id]
        label = f'A{o.agent_id} {agent_label(o.agent_id)}'
        if o.agent_id not in agents_plotted_combined:
            agents_plotted_combined.add(o.agent_id)
            agent_handles.append(mlines.Line2D([], [], color=color, linewidth=2, label=label))
            
        ax_c.plot([cum_x_sell, cum_x_sell + o.quantity], [o.price, o.price], 
                  color=color, linewidth=3.0, alpha=0.9, linestyle='-', solid_capstyle='butt')
                  
        cum_x_sell += o.quantity

    ax_c.set_xlabel('Kümülatif Miktar (kWh)', fontsize=11)
    ax_c.set_ylabel('Teklif Fiyatı ($/kWh)', fontsize=11)
    ax_c.set_title('Birleşik Teklif Grafiği', fontsize=12, fontweight='bold')
    ax_c.grid(True, alpha=0.3)

    # Sort agent handles by the agent ID in their label text
    agent_handles.sort(key=lambda h: int(h.get_label().split()[0][1:]))

    leg1 = ax_c.legend(handles=agent_handles,
                       fontsize=8, loc='upper right',
                       title='Agent', title_fontsize=9)
    ax_c.add_artist(leg1)

    fig_c.tight_layout()
    combined_path = f"{save_dir}/combined_bids_asks.png"
    fig_c.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close(fig_c)
    print(f"Combined bids/asks plot saved to: {combined_path}")



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
