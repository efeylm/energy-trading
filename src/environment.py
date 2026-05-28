"""
POMG Environment for P2P Energy Trading.

Implements the Partially Observable Markov Game (POMG) from
Qiu et al. (IJCAI-21) Section 3.3:

    POMG = (N, S, {O_n}, {A_n}, T, {R_n})

- N agents (producers + consumers) with pre-determined daily profiles
- Each hour is one auction period with batch clearing
- No grid connection: unmatched energy is curtailed (starvation possible)
- Emergency bidding for agents near starvation threshold

The environment orchestrates:
1. Profile-based generation/consumption per hour
2. Agent observations and decisions
3. Market order submission
4. Batch clearing (Algorithm 1)
5. Trade settlement
6. Reward computation
"""

from typing import List, Dict, Tuple, Optional
import math
# pyrefly: ignore [missing-import]
import numpy as np

from src.config import SimConfig
from src.agent import EnergyAgent, Observation, Action
from src.market import (
    DoubleAuctionMarket, IterativeDoubleAuction, Order, ClearingResult,
    PartialMatchDoubleAuction, PartialClearingResult, PartialTrade,
)
from src.profiles import ProfileManager
from src.metrics import MetricsCollector, HourlyMetrics


class EnergyTradingEnv:
    """POMG environment for P2P energy trading simulation.
    
    Runs a 24-hour simulation day where agents trade energy
    through a batch double auction market.
    """
    
    def __init__(self, config: SimConfig):
        self.config = config
        self.market = PartialMatchDoubleAuction()  # Partial-match DA with order persistence
        self.metrics = MetricsCollector(config.n_agents)
        self.profiles: Optional[ProfileManager] = None
        self.agents: List[EnergyAgent] = []
        self.current_hour = 0
        # Per-hour order log: list of {agent_id: Order} dicts (one entry per hour)
        self.bid_ask_log: List[Dict] = []
        
    def reset(self, seed: Optional[int] = None) -> Dict:
        """Initialize/reset the environment for a new simulation day.
        
        Creates agents, generates profiles, resets market and metrics.
        
        Returns:
            Initial observations for all agents.
        """
        actual_seed = seed if seed is not None else self.config.seed
        
        # Generate daily profiles
        self.profiles = ProfileManager(
            n_producers=self.config.n_producers,
            n_consumers=self.config.n_consumers,
            seed=actual_seed,
        )
        
        # Random number generator for parameter variation
        rng = np.random.default_rng(actual_seed + 1)
        
        # Create agents
        self.agents = []
        for i in range(self.config.n_agents):
            if i < self.config.n_producers:
                agent_type = "producer"   # PV owner, sell-only (no buy orders)
                base_mb = self.config.producer_mb
                base_mc = self.config.producer_mc
            else:
                agent_type = "consumer"
                base_mb = self.config.consumer_mb
                base_mc = self.config.consumer_mc
            
            # Add ±15% variation so same-type agents don't have identical curves
            from src.config import AgentMBParams, AgentMCParams
            mb_params = AgentMBParams(
                alpha=base_mb.alpha * rng.uniform(0.85, 1.15),
                beta=base_mb.beta * rng.uniform(0.85, 1.15)
            )
            mc_params = AgentMCParams(
                gamma=base_mc.gamma * rng.uniform(0.85, 1.15),
                delta=base_mc.delta * rng.uniform(0.85, 1.15)
            )
            
            agent = EnergyAgent(
                agent_id=i,
                agent_type=agent_type,
                mb_params=mb_params,
                mc_params=mc_params,
                config=self.config,
            )
            self.agents.append(agent)
        
        # Reset market and metrics — fresh PartialMatchDoubleAuction each day
        self.market = PartialMatchDoubleAuction()
        self.metrics = MetricsCollector(self.config.n_agents)
        self.current_hour = 0
        self.bid_ask_log = []

        # Dedicated RNG for per-step stochastic decisions (flat-block pricing, etc.).
        # Uses seed+2 so it is independent of the profile/agent RNG streams.
        self._step_rng = np.random.default_rng(actual_seed + 2)
        
        # Print profile summary
        print(self.profiles.summary())
        print()
        
        return self._get_all_observations()
    
    def _get_all_observations(self) -> Dict[int, Observation]:
        """Generate observations for all agents at current hour."""
        market_info = self.market.get_market_info()  # works for both market types
        observations = {}
        
        for agent in self.agents:
            obs = Observation(
                inflexible_load=self.profiles.get_inflexible_load(
                    agent.agent_id, self.current_hour
                ),
                best_bid=market_info["best_bid"],
                best_ask=market_info["best_ask"],
                last_avg_price=market_info["last_avg_price"],
                hour=self.current_hour,
            )
            observations[agent.agent_id] = obs
        
        return observations
    
    def step(self) -> Tuple[Dict, Dict[int, float], bool, Dict]:
        """Execute one hour (auction period) of the simulation.

        Steps:
        1. Generate observations for all agents
        2. Each agent decides action (market order)
        3. Submit orders to PartialMatchDoubleAuction
        4. Clear the market (partial matching, unmatched orders persist)
        5. Settle trades and handle unmatched orders
        6. Compute rewards and metrics

        Returns:
            (observations, rewards, done, info)
        """
        hour = self.current_hour

        # --- Step 1: Observations ---
        observations = self._get_all_observations()

        # --- Step 2: Agent decisions ---
        actions: Dict[int, Action] = {}
        for agent in self.agents:
            obs = observations[agent.agent_id]
            action = agent.decide_action(obs)
            actions[agent.agent_id] = action

        # --- Step 3: Submit unit-by-unit orders priced by MB/MC curves ---
        # ─────────────────────────────────────────────────────────────────
        # Each agent's total market quantity is split into chunks of
        # unit_size kWh.  Every chunk gets its own price:
        #
        #   Buyers  → price_k = MB(k * unit_size)   (decreasing in k)
        #   Sellers → price_k = MC(k * unit_size)   (increasing in k)
        #
        # The market collects all unit orders from ALL agents, sorts
        # buyers descending and sellers ascending, then matches them
        # one unit at a time.  This mirrors a proper Walrasian auction
        # where each unit's value is evaluated independently.
        # ─────────────────────────────────────────────────────────────────
        n_buyers = 0
        n_sellers = 0
        agent_requested: Dict[int, Order] = {}  # original order per agent
        hour_bid_ask: Dict[int, list] = {}       # unit orders per agent (for plotting)

        # Flush stale pending orders from the previous hour.
        self.market._buy_book = []
        self.market._sell_book = []

        unit_size = self.config.unit_size

        # Flat-block pricing config
        s_flat_prob = self.config.seller_flat_prob
        s_flat_min  = self.config.seller_flat_min_units
        s_flat_max  = self.config.seller_flat_max_units
        flat_sellers = []   # for logging

        for agent in self.agents:
            order = actions[agent.agent_id].order
            if order is None or order.quantity <= 1e-6:
                continue

            agent_requested[agent.agent_id] = order  # store for unmet tracking
            if order.is_buy:
                n_buyers += 1
            else:
                n_sellers += 1

            # Split into unit orders, each priced by MB or MC at cumulative qty
            n_units = max(1, math.ceil(order.quantity / unit_size))

            # ── Flat-block pricing (SELLERS only) ────────────────────────────
            # first `flat_block` units all priced at MC(0).
            # ────────────────────────────────────────────────────────────────
            flat_block     = 0      # seller: units priced at MC(0)
            flat_price     = 0.0

            if not order.is_buy and n_units >= 2 and self._step_rng.random() < s_flat_prob:
                hi = min(s_flat_max, n_units)
                lo = min(s_flat_min, hi)
                flat_block = int(self._step_rng.integers(lo, hi + 1))
                flat_price = agent.compute_mc(0.0)
                flat_sellers.append(f"A{agent.agent_id}({flat_block}×${flat_price:.4f})")

            unit_orders = []

            # ── Order submission strategy ────────────────────────────────────
            # Flat block  → ONE aggregate order (combined quantity at flat price)
            # Remaining   → individual unit orders (each at its own MB/MC price)
            #
            # Submitting the flat block as a single order means the market
            # sees exactly ONE bid/ask at that price level per agent, so it
            # cannot match the same agent with itself or create duplicate
            # (buyer, seller, price) trade records within the flat range.
            # ────────────────────────────────────────────────────────────────

            # ── Order submission ─────────────────────────────────────────────
            # We calculate a 'multiplier' to shift the agent's base curve 
            # by their strategic decision (from Q-Learning or Baseline).
            # This preserves the 'unit-by-unit' architecture while making
            # actions effective in the market.
            
            if order.is_buy:
                if order.use_flat_price:
                    # Q-Learning alıcı: flat fiyatlı tek toplu emir (satıcıyla simetrik)
                    agg_buy_order = Order(
                        agent_id=agent.agent_id,
                        price=float(np.clip(order.price, 0.001, 2.0)),
                        quantity=order.quantity,
                        is_buy=True,
                        is_emergency=order.is_emergency,
                    )
                    self.market.submit_order(agg_buy_order)
                    unit_orders.append(agg_buy_order)
                else:
                    # Heuristik alıcı: MB eğrisi ile ünite başı emir
                    base_price = agent.compute_mb(order.quantity, hour=self.current_hour)
                    multiplier = order.price / base_price if base_price > 1e-6 else 1.0

                    for k in range(n_units):
                        cum_qty    = k * unit_size
                        actual_qty = min(unit_size, order.quantity - cum_qty)
                        if actual_qty <= 1e-9:
                            break

                        raw_u_price = agent.compute_mb(cum_qty, hour=self.current_hour)
                        u_price = float(np.clip(raw_u_price * multiplier, 0.001, 2.0))

                        ind_order = Order(
                            agent_id=agent.agent_id,
                            price=u_price,
                            quantity=actual_qty,
                            is_buy=True,
                            is_emergency=order.is_emergency,
                        )
                        self.market.submit_order(ind_order)
                        unit_orders.append(ind_order)
            else:
                # SELLERS: Currently using a flat price for total quantity.
                # We use the agent's decided price (which includes their strategy).
                flat_price = order.price
                
                agg_order = Order(
                    agent_id=agent.agent_id,
                    price=flat_price,
                    quantity=order.quantity,
                    is_buy=False,
                    is_emergency=order.is_emergency,
                )
                self.market.submit_order(agg_order)
                unit_orders.append(agg_order)
                flat_sellers.append(f"A{agent.agent_id}({order.quantity:.1f}kWh@${flat_price:.4f})")

            hour_bid_ask[agent.agent_id] = unit_orders


        # Log flat-block pricing this hour
        if flat_sellers:
            print(f"    [FLAT-S] Sabit fiyat kullanan satıcılar: {', '.join(flat_sellers)}")

        # Snapshot this hour's unit orders for post-simulation plotting
        self.bid_ask_log.append(hour_bid_ask)


        # --- Step 5: Clear market (PartialMatchDoubleAuction) ---
        clearing_result: PartialClearingResult = self.market.clear_period()

        # Build pending lookup: agent_id → remaining unmatched quantity
        # (Buyers submit one order per unit; sum all pending rows per agent.)
        pending_buy_qty: Dict[int, float] = {}
        for po in clearing_result.pending_buy_orders:
            pending_buy_qty[po.agent_id] = (
                pending_buy_qty.get(po.agent_id, 0.0) + po.remaining_quantity
            )
        pending_sell_qty: Dict[int, float] = {}
        for po in clearing_result.pending_sell_orders:
            pending_sell_qty[po.agent_id] = (
                pending_sell_qty.get(po.agent_id, 0.0) + po.remaining_quantity
            )

        # Print per-trade info (mirrors the old iterative auction verbose output)
        for t in clearing_result.trades:
            print(
                f"    TRADE  A{t.buyer_id}(bid={t.buyer_bid:.4f}) <-> "
                f"A{t.seller_id}(ask={t.seller_ask:.4f}) "
                f"qty={t.quantity:.3f} kWh @ ${t.price:.4f}/kWh"
            )

        # --- Step 6: Settle trades ---
        rewards = {}
        total_unmet = 0.0
        total_curtailed = 0.0
        n_starvation = 0
        agent_battery_soc = {}

        for agent in self.agents:
            agent_id = agent.agent_id

            # Trades involving this agent in this period
            trades_as_buyer = [
                t for t in clearing_result.trades if t.buyer_id == agent_id
            ]
            trades_as_seller = [
                t for t in clearing_result.trades if t.seller_id == agent_id
            ]

            bought_kwh = sum(t.quantity for t in trades_as_buyer)
            sold_kwh   = sum(t.quantity for t in trades_as_seller)

            # Unmet demand  = quantity this hour that is still pending in the buy book
            # Curtailed     = quantity this hour that is still pending in the sell book
            # (Unmatched orders remain in the book for future periods, but we
            #  charge the cost this hour so the agent receives the signal.)
            unmet_demand      = 0.0
            curtailed_surplus = 0.0

            if agent_id in agent_requested:
                req = agent_requested[agent_id]
                if req.is_buy:
                    unmet_demand = pending_buy_qty.get(agent_id, 0.0)
                else:
                    curtailed_surplus = pending_sell_qty.get(agent_id, 0.0)

            if unmet_demand > 0.01:
                n_starvation += 1

            total_unmet      += unmet_demand
            total_curtailed  += curtailed_surplus

            # PartialTrade has the same fields as Trade used by process_trade_results
            agent.process_trade_results(
                trades_as_buyer=trades_as_buyer,
                trades_as_seller=trades_as_seller,
                unmet_demand=unmet_demand,
                curtailed_surplus=curtailed_surplus,
                hour=hour,
            )

            rewards[agent_id] = agent.get_reward(hour)

            buy_cost    = sum(t.price * t.quantity for t in trades_as_buyer)
            sell_income = sum(t.price * t.quantity for t in trades_as_seller)
            self.metrics.record_agent_hour(
                agent_id=agent_id,
                cost=buy_cost - sell_income,
                bought=bought_kwh,
                sold=sold_kwh,
                unmet=unmet_demand,
                curtailed=curtailed_surplus,
            )

        # --- Step 7: Record hourly metrics ---
        hourly = HourlyMetrics(
            hour=hour,
            average_price=clearing_result.average_price,
            total_traded_kwh=clearing_result.total_traded_kwh,
            n_trades=len(clearing_result.trades),
            total_unmet_demand=total_unmet,
            total_curtailed=total_curtailed,
            n_buyers=n_buyers,
            n_sellers=n_sellers,
        )
        self.metrics.record_hour(hourly)

        self._print_hour_summary(hour, clearing_result, total_unmet, total_curtailed)

        # Advance time
        self.current_hour += 1
        done = self.current_hour >= self.config.t_hours

        next_obs = self._get_all_observations() if not done else {}

        info = {
            "clearing_result": clearing_result,
            "total_unmet_demand": total_unmet,
            "total_curtailed": total_curtailed,
            "n_starvation_events": n_starvation,
        }

        return next_obs, rewards, done, info
    
    def run_day(self) -> MetricsCollector:
        """Run a complete 24-hour simulation day.
        
        Returns:
            MetricsCollector with all recorded data.
        """
        self.reset()
        
        print("=" * 60)
        print("  Starting 24-hour P2P Energy Trading Simulation")
        print("=" * 60)
        print()
        
        done = False
        while not done:
            _, _, done, _ = self.step()
        
        print()
        print(self.metrics.summary())
        
        return self.metrics
    
    def _print_hour_summary(
        self,
        hour: int,
        result: PartialClearingResult,
        unmet: float,
        curtailed: float,
    ):
        """Print a concise summary of one hour's activity."""
        price_str = f"${result.average_price:.4f}" if result.trades else "N/A"
        pending_buy  = sum(po.remaining_quantity for po in result.pending_buy_orders)
        pending_sell = sum(po.remaining_quantity for po in result.pending_sell_orders)
        pending_str  = (
            f"  [pending buy={pending_buy:.2f} sell={pending_sell:.2f} kWh]"
            if (pending_buy + pending_sell) > 1e-6 else ""
        )
        print(
            f"  Hour {hour:2d}: "
            f"Trades={len(result.trades):2d}, "
            f"Volume={result.total_traded_kwh:6.2f}kWh, "
            f"Price={price_str}, "
            f"Unmet={unmet:.2f}kWh, "
            f"Curtailed={curtailed:.2f}kWh"
            f"{pending_str}"
        )
