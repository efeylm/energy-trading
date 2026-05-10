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
    
    hours = np.arange(len(metrics.hourly)) * env.config.delta_t
    n_producers = env.config.n_producers
    n_agents = env.config.n_agents
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
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
    
    # --- 3. Battery SoC Timeseries ---
    ax = axes[1, 0]
    colors_producer = plt.cm.Blues(np.linspace(0.4, 0.9, n_producers))
    colors_consumer = plt.cm.Reds(np.linspace(0.4, 0.9, n_agents - n_producers))

    for i in range(n_agents):
        soc = metrics.get_battery_soc_timeseries(i)
        if i < n_producers:
            ax.plot(hours, soc, color=colors_producer[i], linewidth=1.5,
                    label=f'Pr{i+1}', linestyle='-')
        else:
            ax.plot(hours, soc, color=colors_consumer[i - n_producers], linewidth=1.5,
                    label=f'C{i - n_producers + 1}', linestyle='--')
    
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
    
    # --- 5. PV Generation & Load Profiles ---
    ax = axes[2, 0]
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
        label = f'Pr{i+1}' if i < n_producers else f'C{i - n_producers + 1}'
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

    # ------------------------------------------------------------------ #
    # Separate figure: per-agent bid / ask scatter charts                 #
    # ------------------------------------------------------------------ #
    _plot_bids_asks(env, save_dir=save_dir)

    return filepath


def _plot_bids_asks(env, save_dir: str = "."):
    """Generate two separate Price×Quantity charts from env.bid_ask_log.

    Chart 1 — Alici Teklifleri: all buy orders across all 24 hours,
               each agent in a distinct colour.
    Chart 2 — Satici Teklifleri: all sell orders across all 24 hours,
               each agent in a distinct colour.
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
    # Collect unit-order data from bid_ask_log                            #
    # bid_ask_log[hour] = {agent_id: [unit_Order, ...]}                  #
    # ------------------------------------------------------------------ #
    #
    # For each agent we build two views:
    #   "by_hour"  – one (sorted) step-curve per active hour
    #   "all"      – every unit across all hours (aggregate scatter)
    #
    BuyData  = {i: {'by_hour': {}, 'all_prices': [], 'all_cumqtys': []}
                for i in range(n_agents)}
    SellData = {i: {'by_hour': {}, 'all_prices': [], 'all_cumqtys': []}
                for i in range(n_agents)}

    for hour, hour_orders in enumerate(bid_ask_log):
        for agent_id, unit_orders in hour_orders.items():
            if not unit_orders:
                continue
            is_buy = unit_orders[0].is_buy
            store  = BuyData[agent_id] if is_buy else SellData[agent_id]

            # Sort units by price (desc for buyers, asc for sellers)
            sorted_units = sorted(unit_orders,
                                  key=lambda o: o.price,
                                  reverse=is_buy)

            # Build step-curve for this hour.
            # Rule: each x value appears EXACTLY ONCE.
            #   xs[i] = left edge of step i  (= cumulative qty before step i)
            #   ys[i] = price for step i
            # The LAST point (xs[-1], ys[-1]) is a sentinel: same price as the
            # previous step but shifted right by its quantity so the final
            # horizontal segment is visible.  No duplicate x values.
            # matplotlib's drawstyle='steps-post' draws the horizontal bar
            # from xs[i] to xs[i+1] at height ys[i], with a vertical drop at
            # each boundary — the vertical is drawn by matplotlib, NOT stored
            # in the data, so the data remains a proper function (x → y).
            cum = 0.0
            xs, ys = [], []
            for u in sorted_units:
                xs.append(cum)        # left edge
                ys.append(u.price)
                cum += u.quantity
            # Sentinel: extend the last step to its right edge
            if xs:
                xs.append(cum)
                ys.append(ys[-1])

            store['by_hour'][hour] = (xs, ys)
            # Aggregate scatter: midpoint of each step
            mid = 0.0
            for u in sorted_units:
                mid += u.quantity / 2
                store['all_prices'].append(u.price)
                store['all_cumqtys'].append(mid)
                mid += u.quantity / 2

    # ------------------------------------------------------------------ #
    # Find the single representative hour (most agents with orders)        #
    # ------------------------------------------------------------------ #
    hour_agent_count = {}
    for hour, hour_orders in enumerate(bid_ask_log):
        hour_agent_count[hour] = len(hour_orders)
    peak_hour = max(hour_agent_count, key=hour_agent_count.get)

    # ------------------------------------------------------------------ #
    # Plot buyer bids  (one curve per agent, representative hour)          #
    # ------------------------------------------------------------------ #
    fig_b, ax_buy = plt.subplots(figsize=(10, 6))
    fig_b.suptitle(
        f'Alıcı Teklifleri — Fiyat × Kümülatif Miktar  (Saat {peak_hour})\n'
        'Her renk bir agent; eğri o saatin talep çizelgesi',
        fontsize=13, fontweight='bold')

    any_buyer = False

    for agent_id in range(n_agents):
        data  = BuyData[agent_id]
        color = agent_colors[agent_id]
        label = f'A{agent_id} {agent_label(agent_id)}'

        result = data['by_hour'].get(peak_hour)
        if result is None:
            continue
        xs, ys = result
        ln, = ax_buy.plot(xs, ys,
                          color=color, linewidth=2.0, alpha=0.9,
                          drawstyle='steps-post', label=label)
        any_buyer = True

    ax_buy.set_xlabel('Kümülatif Miktar (kWh)', fontsize=11)
    ax_buy.set_ylabel('Teklif Fiyatı ($/kWh)',   fontsize=11)
    ax_buy.set_title('Alıcı Teklifleri (Buy Bids)', fontsize=12, fontweight='bold')
    ax_buy.grid(True, alpha=0.3)
    if any_buyer:
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
    # Plot seller asks  (one curve per agent, representative hour)         #
    # ------------------------------------------------------------------ #
    fig_s, ax_sell = plt.subplots(figsize=(10, 6))
    fig_s.suptitle(
        f'Satıcı Teklifleri — Fiyat × Kümülatif Miktar  (Saat {peak_hour})\n'
        'Her renk bir agent; eğri o saatin arz çizelgesi',
        fontsize=13, fontweight='bold')

    any_seller = False

    for agent_id in range(n_agents):
        data  = SellData[agent_id]
        color = agent_colors[agent_id]
        label = f'A{agent_id} {agent_label(agent_id)}'

        result = data['by_hour'].get(peak_hour)
        if result is None:
            continue
        xs, ys = result
        ln, = ax_sell.plot(xs, ys,
                           color=color, linewidth=2.0, alpha=0.9,
                           drawstyle='steps-post', label=label)
        any_seller = True

    ax_sell.set_xlabel('Kümülatif Miktar (kWh)', fontsize=11)
    ax_sell.set_ylabel('Teklif Fiyatı ($/kWh)',   fontsize=11)
    ax_sell.set_title('Satıcı Teklifleri (Sell Asks)', fontsize=12, fontweight='bold')
    ax_sell.grid(True, alpha=0.3)
    if any_seller:
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
    # Combined chart  (one curve per agent, representative hour)           #
    # ------------------------------------------------------------------ #
    fig_c, ax_c = plt.subplots(figsize=(13, 7))
    fig_c.suptitle(
        f'Alıcı & Satıcı Teklifleri — Fiyat × Kümülatif Miktar  (Saat {peak_hour})\n'
        'Düz çizgi = Alıcı Bid  |  Kesikli çizgi = Satıcı Ask  |  Renk = Agent',
        fontsize=13, fontweight='bold')

    legend_handles = {}

    for agent_id in range(n_agents):
        color = agent_colors[agent_id]
        label = f'A{agent_id} {agent_label(agent_id)}'
        buy_d  = BuyData[agent_id]
        sell_d = SellData[agent_id]

        # Buyer curve — solid
        b_result = buy_d['by_hour'].get(peak_hour)
        if b_result is not None:
            xs, ys = b_result
            ln, = ax_c.plot(xs, ys,
                            color=color, linewidth=2.0, alpha=0.9,
                            linestyle='-', drawstyle='steps-post', label=label)
            legend_handles.setdefault(agent_id, {})['buy'] = ln

        # Seller curve — dashed
        s_result = sell_d['by_hour'].get(peak_hour)
        if s_result is not None:
            xs, ys = s_result
            ln, = ax_c.plot(xs, ys,
                            color=color, linewidth=2.0, alpha=0.9,
                            linestyle='--', drawstyle='steps-post')
            legend_handles.setdefault(agent_id, {})['sell'] = ln
            if agent_id not in legend_handles or 'buy' not in legend_handles[agent_id]:
                ln.set_label(label)

    ax_c.set_xlabel('Kümülatif Miktar (kWh)', fontsize=11)
    ax_c.set_ylabel('Teklif Fiyatı ($/kWh)', fontsize=11)
    ax_c.set_title('Birleşik Teklif Grafiği', fontsize=12, fontweight='bold')
    ax_c.grid(True, alpha=0.3)

    # --- Legend: two sections ---
    # Section 1: one coloured entry per agent (from buyer solid lines)
    agent_handles = [
        legend_handles[aid].get('buy') or legend_handles[aid].get('sell')
        for aid in sorted(legend_handles)
        if ('buy' in legend_handles[aid] or 'sell' in legend_handles[aid])
    ]
    # Section 2: two style proxy lines for buyer vs seller
    import matplotlib.lines as mlines
    style_buy  = mlines.Line2D([], [], color='black', linewidth=1.5,
                               linestyle='-',  label='Alıcı Bid (solid)')
    style_sell = mlines.Line2D([], [], color='black', linewidth=1.5,
                               linestyle='--', label='Satıcı Ask (dashed)')

    leg1 = ax_c.legend(handles=agent_handles,
                       fontsize=8, loc='upper right',
                       title='Agent', title_fontsize=9)
    ax_c.add_artist(leg1)
    ax_c.legend(handles=[style_buy, style_sell],
                fontsize=9, loc='upper center',
                title='Çizgi Stili', title_fontsize=9)

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
