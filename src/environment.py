"""
POMG Environment for P2P Energy Trading.

Implements the Partially Observable Markov Game (POMG) from
Qiu et al. (IJCAI-21) Section 3.3:

    POMG = (N, S, {O_n}, {A_n}, T, {R_n})

- N agents (producers + consumers) with pre-determined daily profiles
- Each half-hour slot is one auction period with batch clearing
- Grid fallback (config.grid_fallback, varsayılan): P2P'de eşleşmeyen alıcı
  şebekeden ToU'dan alır, eşleşmeyen satıcı şebekeye FiT'ten satar

The environment orchestrates:
1. Profile-based generation/consumption per hour
2. Agent observations and decisions
3. Market order submission
4. Batch clearing (Algorithm 1)
5. Trade settlement
6. Reward computation
"""

from typing import List, Dict, Tuple, Optional
# pyrefly: ignore [missing-import]
import numpy as np

from src.config import SimConfig
from src.agent import EnergyAgent, Observation, Action
from src.market import Order, PartialMatchDoubleAuction, PartialClearingResult
from src.profiles import ProfileManager
from src.metrics import MetricsCollector, HourlyMetrics


class EnergyTradingEnv:
    """POMG environment for P2P energy trading simulation.
    
    Runs a 24-hour simulation day (48 half-hour slots) where agents trade
    energy through a batch double auction market. Ajanlar reset()'te taban
    EnergyAgent olarak kurulur; alt sınıflar (RandomBaselineEnv, QLearningEnv,
    REINFORCEEnv) onları kendi ajanlarıyla değiştirir.
    """
    
    def __init__(self, config: SimConfig):
        self.config = config
        self.market = PartialMatchDoubleAuction()
        self.metrics = MetricsCollector(config.n_agents)
        self.profiles: Optional[ProfileManager] = None
        self.agents: List[EnergyAgent] = []
        self.current_hour = 0
        
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
            # Yalnızca talep şoku deneyi için; varsayılan 1.0 = değişiklik yok.
            demand_multiplier=getattr(self.config, "demand_multiplier", 1.0),
            # Yalnızca arz şoku deneyi için; varsayılan 1.0 = değişiklik yok.
            supply_multiplier=getattr(self.config, "supply_multiplier", 1.0),
        )
        
        # Create agents
        self.agents = []
        for i in range(self.config.n_agents):
            agent_type = "producer" if i < self.config.n_producers else "consumer"
            agent = EnergyAgent(
                agent_id=i,
                agent_type=agent_type,
                config=self.config,
            )
            self.agents.append(agent)
        
        # Reset market and metrics — fresh PartialMatchDoubleAuction each day
        self.market = PartialMatchDoubleAuction()
        self.metrics = MetricsCollector(self.config.n_agents)
        self.current_hour = 0

        # Print profile summary
        print(self.profiles.summary())
        print()
        
        return self._get_all_observations()
    
    def _get_all_observations(self) -> Dict[int, Observation]:
        """Generate observations for all agents at current hour.

        Makale Eq. (6): o_n,t = [P^inf_n,t, λ^avg_t, t]

        best_bid / best_ask ARTIK GÖZLEME DAHİL DEĞİL: PartialMatchDoubleAuction
        clear_period() sonunda emir defterlerini boşalttığı için bu iki değer
        adımlar arasında her zaman 0.0 oluyordu (bkz. src/agent.py Observation).
        """
        market_info = self.market.get_market_info()
        observations = {}

        for agent in self.agents:
            obs = Observation(
                inflexible_load=self.profiles.get_inflexible_load(
                    agent.agent_id, self.current_hour
                ),
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
        4. Clear the market (partial matching; unmatched remainder → grid)
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

        # --- Step 3: Submit orders ---
        # Her ajan slot başına TEK toplu emir verir (fiyat ajanın stratejisinden:
        # Random / QL / NN). Alıcı fiyatı [0.001, 2.0] aralığına kırpılır.
        n_buyers = 0
        n_sellers = 0
        agent_requested: Dict[int, Order] = {}  # original order per agent

        # Flush stale pending orders from the previous hour.
        self.market._buy_book = []
        self.market._sell_book = []

        for agent in self.agents:
            order = actions[agent.agent_id].order
            if order is None or order.quantity <= 1e-6:
                continue

            agent_requested[agent.agent_id] = order  # store for unmet tracking
            if order.is_buy:
                n_buyers += 1
                self.market.submit_order(Order(
                    agent_id=agent.agent_id,
                    price=float(np.clip(order.price, 0.001, 2.0)),
                    quantity=order.quantity,
                    is_buy=True,
                ))
            else:
                n_sellers += 1
                self.market.submit_order(Order(
                    agent_id=agent.agent_id,
                    price=order.price,
                    quantity=order.quantity,
                    is_buy=False,
                ))

        # --- Step 4: Clear market (PartialMatchDoubleAuction) ---
        clearing_result: PartialClearingResult = self.market.clear_period()

        # Build pending lookup: agent_id → remaining unmatched quantity
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

        # --- Step 5: Settle trades ---
        rewards = {}
        total_unmet = 0.0
        total_curtailed = 0.0
        total_grid_purchased = 0.0
        total_grid_sold = 0.0

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

            # Unmet demand  = quantity this hour left unmatched in the buy book
            # Curtailed     = quantity this hour left unmatched in the sell book
            # (Defterler her periyot sonunda boşaltılır; artık grid fallback'e gider.)
            unmet_demand      = 0.0
            curtailed_surplus = 0.0

            if agent_id in agent_requested:
                req = agent_requested[agent_id]
                if req.is_buy:
                    unmet_demand = pending_buy_qty.get(agent_id, 0.0)
                else:
                    curtailed_surplus = pending_sell_qty.get(agent_id, 0.0)

            # --- Grid fallback ---
            # P2P'de eşleşemeyen alıcı → grid'den ToU fiyatıyla alır
            # P2P'de eşleşemeyen satıcı → grid'e FiT fiyatıyla satar
            if self.config.grid_fallback:
                grid_purchased    = unmet_demand
                grid_sold         = curtailed_surplus
                unmet_demand      = 0.0   # grid tamamen karşıladı
                curtailed_surplus = 0.0   # grid tamamen aldı
            else:
                grid_purchased = 0.0
                grid_sold      = 0.0

            total_unmet     += unmet_demand
            total_curtailed += curtailed_surplus
            total_grid_purchased += grid_purchased
            total_grid_sold      += grid_sold

            agent.process_trade_results(
                trades_as_buyer=trades_as_buyer,
                trades_as_seller=trades_as_seller,
                unmet_demand=unmet_demand,
                curtailed_surplus=curtailed_surplus,
                hour=hour,
                grid_purchased=grid_purchased,
                grid_sold=grid_sold,
            )

            rewards[agent_id] = agent.get_reward(hour)

            fit_price        = self.config.fit_price
            tou_price        = self.config.tou_price
            buy_cost         = sum(t.price * t.quantity for t in trades_as_buyer)
            sell_income      = sum(t.price * t.quantity for t in trades_as_seller)
            grid_sell_income = grid_sold * fit_price
            grid_buy_cost    = grid_purchased * tou_price
            net_cost         = buy_cost + grid_buy_cost - sell_income - grid_sell_income
            self.metrics.record_agent_hour(
                agent_id=agent_id,
                cost=net_cost,
                bought=bought_kwh,
                sold=sold_kwh,
                unmet=unmet_demand,
                curtailed=curtailed_surplus,
                grid_purchased=grid_purchased,
                grid_sold=grid_sold,
                p2p_sell_income=sell_income,
                p2p_buy_cost=buy_cost,
                grid_sell_income=grid_sell_income,
                grid_buy_cost=grid_buy_cost,
            )

        # --- Step 6: Record hourly metrics ---
        hourly = HourlyMetrics(
            hour=hour,
            average_price=clearing_result.average_price,
            total_traded_kwh=clearing_result.total_traded_kwh,
            n_trades=len(clearing_result.trades),
            total_unmet_demand=total_unmet,
            total_curtailed=total_curtailed,
            n_buyers=n_buyers,
            n_sellers=n_sellers,
            total_grid_purchased=total_grid_purchased,
            total_grid_sold=total_grid_sold,
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
        }

        return next_obs, rewards, done, info
    
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
