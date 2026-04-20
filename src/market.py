"""
Double Auction Market implementations.

DoubleAuctionMarket  — Batch clearing (Algorithm 1, Qiu et al. IJCAI-21).
IterativeDoubleAuction — Price-discovery auction (ADD Bitirme pseudocode Step 6):
    1. All agents shout initial offers X% away from their MB/MC curves.
    2. If best_bid >= best_ask → trade at midpoint.
    3. Unmatched agents converge toward midpoint via alpha parameter.
    4. Repeat until no more trades or convergence.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, TYPE_CHECKING
import copy

if TYPE_CHECKING:
    from src.agent import EnergyAgent


@dataclass
class Order:
    """A single buy or sell order."""
    agent_id: int
    price: float        # $/kWh willingness price
    quantity: float     # kWh (positive for both buy and sell)
    is_buy: bool        # True = buy order, False = sell order
    is_emergency: bool = False  # Emergency bid from starvation mechanism


@dataclass
class Trade:
    """Record of a completed trade."""
    buyer_id: int
    seller_id: int
    price: float        # $/kWh trade price (mid-price)
    quantity: float     # kWh traded
    buyer_bid: float    # Original bid price
    seller_ask: float   # Original ask price


@dataclass
class ClearingResult:
    """Result of the batch clearing process for one auction period."""
    trades: List[Trade] = field(default_factory=list)
    unmatched_buy_orders: List[Order] = field(default_factory=list)
    unmatched_sell_orders: List[Order] = field(default_factory=list)
    total_traded_kwh: float = 0.0
    average_price: float = 0.0
    
    def compute_stats(self):
        """Compute aggregate statistics from trades."""
        if self.trades:
            self.total_traded_kwh = sum(t.quantity for t in self.trades)
            self.average_price = (
                sum(t.price * t.quantity for t in self.trades) / self.total_traded_kwh
            )
        else:
            self.total_traded_kwh = 0.0
            self.average_price = 0.0


class DoubleAuctionMarket:
    """Batch Double Auction market for P2P energy trading.
    
    Each auction period:
    1. Agents submit orders (one per agent)
    2. Orders are sorted into buy/sell order books
    3. Batch clearing matches orders using Algorithm 1
    4. Results are returned to agents
    """
    
    def __init__(self):
        self.buy_orders: List[Order] = []
        self.sell_orders: List[Order] = []
        self.trade_history: List[List[Trade]] = []  # Per-period trade lists
        self.clearing_results: List[ClearingResult] = []
    
    def reset(self):
        """Reset the market for a new simulation day."""
        self.buy_orders = []
        self.sell_orders = []
        self.trade_history = []
        self.clearing_results = []
    
    def submit_order(self, order: Order):
        """Submit a buy or sell order to the market."""
        if order.quantity <= 0:
            return  # Skip zero-quantity orders
        
        if order.is_buy:
            self.buy_orders.append(order)
        else:
            self.sell_orders.append(order)
    
    def clear(self) -> ClearingResult:
        """Execute batch clearing (Algorithm 1 from paper).
        
        Matches buy orders (sorted by descending price) with sell orders
        (sorted by ascending price) using mid-price clearing.
        
        Returns:
            ClearingResult with all trades and unmatched orders.
        """
        result = ClearingResult()
        
        if not self.buy_orders or not self.sell_orders:
            # No trades possible — all orders are unmatched
            result.unmatched_buy_orders = copy.deepcopy(self.buy_orders)
            result.unmatched_sell_orders = copy.deepcopy(self.sell_orders)
            self._finalize_period(result)
            return result
        
        # Step 1: Sort order books
        # Buy: highest price first (most willing buyers matched first)
        buy_book = sorted(self.buy_orders, key=lambda o: -o.price)
        # Sell: lowest price first (cheapest sellers matched first)
        sell_book = sorted(self.sell_orders, key=lambda o: o.price)
        
        # Step 2: Make working copies of quantities
        buy_remaining = [o.quantity for o in buy_book]
        sell_remaining = [o.quantity for o in sell_book]
        
        # Step 3: Matching loop (Algorithm 1)
        i = 0  # Buy index
        j = 0  # Sell index
        
        while i < len(buy_book) and j < len(sell_book):
            buy_order = buy_book[i]
            sell_order = sell_book[j]
            
            # Check if buy price >= sell price (trade possible)
            if buy_order.price < sell_order.price:
                break  # No more matches possible
            
            # Trade quantity = min of remaining quantities
            trade_qty = min(buy_remaining[i], sell_remaining[j])
            
            # Trade price = mid-price (average of bid and ask)
            trade_price = (buy_order.price + sell_order.price) / 2.0
            
            # Record the trade
            trade = Trade(
                buyer_id=buy_order.agent_id,
                seller_id=sell_order.agent_id,
                price=trade_price,
                quantity=trade_qty,
                buyer_bid=buy_order.price,
                seller_ask=sell_order.price,
            )
            result.trades.append(trade)
            
            # Update remaining quantities
            buy_remaining[i] -= trade_qty
            sell_remaining[j] -= trade_qty
            
            # Move to next order if current is fully matched
            if buy_remaining[i] <= 1e-9:
                i += 1
            if sell_remaining[j] <= 1e-9:
                j += 1
        
        # Step 4: Collect unmatched orders
        for k in range(i, len(buy_book)):
            if buy_remaining[k] > 1e-9:
                unmatched = copy.deepcopy(buy_book[k])
                unmatched.quantity = buy_remaining[k]
                result.unmatched_buy_orders.append(unmatched)
        
        for k in range(j, len(sell_book)):
            if sell_remaining[k] > 1e-9:
                unmatched = copy.deepcopy(sell_book[k])
                unmatched.quantity = sell_remaining[k]
                result.unmatched_sell_orders.append(unmatched)
        
        self._finalize_period(result)
        return result
    
    def _finalize_period(self, result: ClearingResult):
        """Compute stats and record period results, then reset order books."""
        result.compute_stats()
        self.trade_history.append(result.trades)
        self.clearing_results.append(result)
        
        # Reset order books for next period
        self.buy_orders = []
        self.sell_orders = []
    
    def get_market_info(self) -> dict:
        """Return public market information (for agent observations).
        
        Returns the best bid and best ask from current order book,
        plus summary of last period's clearing.
        """
        # Current order book info
        best_bid = max((o.price for o in self.buy_orders), default=0.0)
        best_ask = min((o.price for o in self.sell_orders), default=float('inf'))
        
        # Last period summary
        last_avg_price = 0.0
        last_volume = 0.0
        if self.clearing_results:
            last = self.clearing_results[-1]
            last_avg_price = last.average_price
            last_volume = last.total_traded_kwh
        
        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "last_avg_price": last_avg_price,
            "last_volume": last_volume,
            "n_periods_completed": len(self.clearing_results),
        }
    
    def get_agent_trades(self, agent_id: int, period: int = -1) -> List[Trade]:
        """Get all trades for a specific agent in a given period."""
        if not self.trade_history:
            return []
        
        period_trades = self.trade_history[period]
        return [
            t for t in period_trades
            if t.buyer_id == agent_id or t.seller_id == agent_id
        ]
    
    def get_agent_net_cost(self, agent_id: int, period: int = -1) -> float:
        """Get net cost for an agent in a given period.
        
        Positive = net spending (buyer), Negative = net income (seller).
        """
        trades = self.get_agent_trades(agent_id, period)
        net = 0.0
        for t in trades:
            if t.buyer_id == agent_id:
                net += t.price * t.quantity   # Pays
            if t.seller_id == agent_id:
                net -= t.price * t.quantity   # Receives
        return net


class IterativeDoubleAuction:
    """Price-discovery double auction (ADD Bitirme pseudocode Step 6).

    Each auction period proceeds in rounds:
      Round start : All agents hold a current offer (bid or ask).
      Match check : If best_bid >= best_ask → trade at midpoint, record Trade.
      Convergence : Otherwise every unmatched agent moves their offer toward
                    midpoint by alpha factor (buyer up, seller down), clamped
                    to their own MB/MC so they never trade at a loss.
      Termination : No new trade for 10 consecutive rounds, or max_rounds hit.

    Agents must call setup_auction() before each period so their auction state
    (_auction_role, _auction_units_to_trade, etc.) is initialised.
    """

    def __init__(self):
        self.trade_history: List[List[Trade]] = []
        self.clearing_results: List[ClearingResult] = []

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run_period(
        self,
        buyers: List,       # EnergyAgent instances with _auction_role == 'buyer'
        sellers: List,      # EnergyAgent instances with _auction_role == 'seller'
        margin: float,      # config.initial_shout_margin
        alpha: float,       # config.alpha
        unit_size: float,   # config.unit_size
        max_rounds: int,    # config.max_auction_rounds
        verbose: bool = False,
    ) -> ClearingResult:
        """Run one hourly iterative auction. Returns a ClearingResult."""
        result = ClearingResult()

        if not buyers or not sellers:
            self._finalize(result)
            return result

        # Step 1 — Initialize all offers (X% from MB/MC)
        for b in buyers:
            b.auction_initial_offer(margin)
        for s in sellers:
            s.auction_initial_offer(margin)

        no_trade_streak = 0

        for rnd in range(max_rounds):
            # Gather active participants
            active_b = [
                b for b in buyers
                if b.auction_units_remaining() > 0 and b._auction_current_offer is not None
            ]
            active_s = [
                s for s in sellers
                if s.auction_units_remaining() > 0 and s._auction_current_offer is not None
            ]

            if not active_b or not active_s:
                break

            best_buyer = max(active_b, key=lambda b: b._auction_current_offer)
            best_seller = min(active_s, key=lambda s: s._auction_current_offer)

            best_bid = best_buyer._auction_current_offer
            best_ask = best_seller._auction_current_offer

            if best_bid >= best_ask:
                # Step 2 — Match → trade at midpoint
                price = round((best_bid + best_ask) / 2.0, 6)
                trade = Trade(
                    buyer_id=best_buyer.agent_id,
                    seller_id=best_seller.agent_id,
                    price=price,
                    quantity=unit_size,
                    buyer_bid=best_bid,
                    seller_ask=best_ask,
                )
                result.trades.append(trade)

                best_buyer.auction_record_trade()
                best_seller.auction_record_trade()

                # Initialise offers for the next unit
                best_buyer.auction_initial_offer(margin)
                best_seller.auction_initial_offer(margin)

                no_trade_streak = 0

                if verbose:
                    print(
                        f"    [r{rnd+1:02d}] TRADE "
                        f"A{best_buyer.agent_id}(bid={best_bid:.4f}) ↔ "
                        f"A{best_seller.agent_id}(ask={best_ask:.4f}) "
                        f"@ ${price:.4f}/kWh"
                    )

            else:
                # Step 3 — No match: converge toward midpoint
                no_trade_streak += 1
                if no_trade_streak >= 10:
                    if verbose:
                        print(
                            f"    [r{rnd+1:02d}] Converged "
                            f"(best_bid={best_bid:.4f}, best_ask={best_ask:.4f}, "
                            f"streak={no_trade_streak})"
                        )
                    break

                for b in active_b:
                    b.auction_update_offer(best_bid, best_ask, alpha)
                for s in active_s:
                    s.auction_update_offer(best_bid, best_ask, alpha)

        # Record unmatched orders for settlement reference
        for b in buyers:
            remaining = b.auction_units_remaining()
            if remaining > 0:
                result.unmatched_buy_orders.append(Order(
                    agent_id=b.agent_id,
                    price=b._auction_current_offer or 0.0,
                    quantity=remaining * unit_size,
                    is_buy=True,
                ))
        for s in sellers:
            remaining = s.auction_units_remaining()
            if remaining > 0:
                result.unmatched_sell_orders.append(Order(
                    agent_id=s.agent_id,
                    price=s._auction_current_offer or 0.0,
                    quantity=remaining * unit_size,
                    is_buy=False,
                ))

        self._finalize(result)
        return result

    def get_agent_trades(self, agent_id: int, period: int = -1) -> List[Trade]:
        if not self.trade_history:
            return []
        return [
            t for t in self.trade_history[period]
            if t.buyer_id == agent_id or t.seller_id == agent_id
        ]

    def get_market_info(self) -> dict:
        """Public market info consumed by agent observations between hours."""
        last_avg_price = 0.0
        last_volume = 0.0
        if self.clearing_results:
            last = self.clearing_results[-1]
            last_avg_price = last.average_price
            last_volume = last.total_traded_kwh
        return {
            "best_bid": 0.0,        # Dynamic during auction; 0 between hours
            "best_ask": 0.0,
            "last_avg_price": last_avg_price,
            "last_volume": last_volume,
            "n_periods_completed": len(self.clearing_results),
        }

    # ------------------------------------------------------------------ #

    def _finalize(self, result: ClearingResult):
        result.compute_stats()
        self.trade_history.append(result.trades)
        self.clearing_results.append(result)
