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


# ============================================================
# Partial-Match Double Auction (Yeni Sistem)
# ============================================================

@dataclass
class PendingOrder:
    """An order that may be partially or fully unmatched across periods.

    Wraps a base Order with a mutable remaining_quantity so partially-
    matched orders survive into the next clearing period.
    """
    order: Order                    # Original order (immutable)
    remaining_quantity: float       # kWh still to be matched
    period_submitted: int = 0       # Which clearing period this was submitted in

    # ---- convenience passthroughs ----
    @property
    def agent_id(self) -> int:
        return self.order.agent_id

    @property
    def price(self) -> float:
        return self.order.price

    @property
    def is_buy(self) -> bool:
        return self.order.is_buy

    @property
    def mb_value(self) -> float:
        """Marginal-benefit / ask price used for sorting.

        For buyers  this is the bid price (proxy for MB at the submitted qty).
        For sellers this is the ask price (proxy for MC at the submitted qty).
        """
        return self.order.price


@dataclass
class PartialTrade:
    """Record of a single partial (or full) trade within PartialMatchDoubleAuction."""
    buyer_id: int
    seller_id: int
    price: float            # $/kWh — mid-price of this specific unit-pair
    quantity: float         # kWh actually traded
    buyer_bid: float        # Buyer's original bid
    seller_ask: float       # Seller's original ask
    period: int = 0         # Clearing period index


@dataclass
class PartialClearingResult:
    """Result of one PartialMatchDoubleAuction clearing period."""
    trades: List[PartialTrade] = field(default_factory=list)
    pending_buy_orders: List[PendingOrder] = field(default_factory=list)   # Still in book
    pending_sell_orders: List[PendingOrder] = field(default_factory=list)  # Still in book
    total_traded_kwh: float = 0.0
    average_price: float = 0.0
    period: int = 0

    def compute_stats(self):
        if self.trades:
            self.total_traded_kwh = sum(t.quantity for t in self.trades)
            self.average_price = (
                sum(t.price * t.quantity for t in self.trades)
                / self.total_traded_kwh
            )
        else:
            self.total_traded_kwh = 0.0
            self.average_price = 0.0


class PartialMatchDoubleAuction:
    """Double Auction with partial matching and order-book persistence.

    Rules implemented:
    1. Sellers enter a fixed-quantity sell offer.
    2. Buyers are sorted by marginal benefit (bid price descending), with
       ties broken by quantity descending so larger buyers go first.
    3. Each seller's quantity is distributed across sorted buyers; a buyer
       and seller CAN trade at different prices than other buyer-seller pairs
       (price discrimination per unit, using mid-price of that specific pair).
    4. Any unmatched quantity from a sell offer stays in the order book and
       waits for a future buyer in the next clearing period.
    5. Buyers whose demand is not fully met by the current best seller are
       matched against additional sellers (partial cross-seller matching).

    Usage::

        market = PartialMatchDoubleAuction()
        market.submit_order(Order(agent_id=0, price=0.18, quantity=2.0, is_buy=False))
        market.submit_order(Order(agent_id=1, price=0.20, quantity=1.0, is_buy=True))
        market.submit_order(Order(agent_id=2, price=0.19, quantity=1.5, is_buy=True))
        result = market.clear_period()
    """

    def __init__(self):
        # Active order books (persist across periods until matched or reset)
        self._buy_book: List[PendingOrder] = []
        self._sell_book: List[PendingOrder] = []

        # Historical records
        self.clearing_results: List[PartialClearingResult] = []
        self.all_trades: List[PartialTrade] = []   # Flat list of every trade ever

        self._period: int = 0   # Current clearing period counter

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def submit_order(self, order: Order):
        """Add a buy or sell order to the book for the current period.

        Orders with non-positive quantity are silently ignored.
        """
        if order.quantity <= 1e-9:
            return

        pending = PendingOrder(
            order=order,
            remaining_quantity=order.quantity,
            period_submitted=self._period,
        )
        if order.is_buy:
            self._buy_book.append(pending)
        else:
            self._sell_book.append(pending)

    def clear_period(self) -> PartialClearingResult:
        """Match current order books and return the clearing result.

        Algorithm:
        1. Sort buyers: descending bid price (MB proxy), ties → descending qty.
        2. Sort sellers: ascending ask price (MC proxy).
        3. For each seller (cheapest first):
           a. Walk through sorted buyers.
           b. If buyer.bid >= seller.ask → trade at mid-price for min(remaining) qty.
           c. Continue until seller quantity exhausted or no more willing buyers.
        4. Unmatched remainders stay in the order books.
        5. Compute stats and return result.
        """
        result = PartialClearingResult(period=self._period)

        # Step 1 & 2: Sort books
        # Pure Global Price Priority: 
        # Buyers: highest bid first. 
        # Sellers: lowest ask first.
        # This matches the "sort all bids on the chart" requirement.
        
        buy_sorted = sorted(
            self._buy_book,
            key=lambda o: (-o.price, -o.remaining_quantity),
        )
        sell_sorted = sorted(
            self._sell_book,
            key=lambda o: (o.price, -o.remaining_quantity),
        )

        # Step 3: Global Price-Priority Matching loop
        # We always match the CURRENT best buyer with the CURRENT best seller.
        # This ensures that A5 (highest bid) always gets the absolute best
        # available deal before anyone else (like A4) even starts.
        
        b_idx = 0
        s_idx = 0
        
        while b_idx < len(buy_sorted) and s_idx < len(sell_sorted):
            buyer = buy_sorted[b_idx]
            seller = sell_sorted[s_idx]
            
            if buyer.remaining_quantity <= 1e-9:
                b_idx += 1
                continue
            if seller.remaining_quantity <= 1e-9:
                s_idx += 1
                continue
                
            # Prevent self-trading
            if buyer.agent_id == seller.agent_id:
                # If they are the same agent, this buyer must look at the NEXT seller
                # or this seller must look at the NEXT buyer. 
                # For simplicity in batch clearing, we skip this pair.
                # In a real market, you'd look for the next best price.
                s_idx += 1 
                continue

            # Price check
            if buyer.price < seller.price:
                break # No more matches possible (books are sorted)

            # Match them
            trade_qty = min(buyer.remaining_quantity, seller.remaining_quantity)
            trade_price = (buyer.price + seller.price) / 2.0
            
            trade = PartialTrade(
                buyer_id=buyer.agent_id,
                seller_id=seller.agent_id,
                price=trade_price,
                quantity=trade_qty,
                buyer_bid=buyer.price,
                seller_ask=seller.price,
                period=self._period,
            )
            result.trades.append(trade)
            self.all_trades.append(trade)
            
            buyer.remaining_quantity -= trade_qty
            seller.remaining_quantity -= trade_qty
            
            # If one side is exhausted, the loop will move to the next index 
            # in the next iteration via the <= 1e-9 checks above.


        # Step 4: Collect pending (unmatched) orders that remain in book
        # Prune fully-matched entries from the books
        self._buy_book = [o for o in self._buy_book if o.remaining_quantity > 1e-9]
        self._sell_book = [o for o in self._sell_book if o.remaining_quantity > 1e-9]

        result.pending_buy_orders = list(self._buy_book)
        result.pending_sell_orders = list(self._sell_book)

        # Step 5: Finalize
        result.compute_stats()
        self.clearing_results.append(result)
        self._period += 1

        return result

    def reset(self):
        """Clear order books and reset period counter (daily reset)."""
        self._buy_book = []
        self._sell_book = []
        self.clearing_results = []
        self.all_trades = []
        self._period = 0

    def get_pending_orders(self) -> Tuple[List[PendingOrder], List[PendingOrder]]:
        """Return (buy_book, sell_book) — orders still waiting to be matched."""
        return list(self._buy_book), list(self._sell_book)

    def get_market_info(self) -> dict:
        """Public market snapshot for agent observations."""
        best_bid = max((o.price for o in self._buy_book), default=0.0)
        best_ask = min((o.price for o in self._sell_book), default=float("inf"))

        last_avg_price = 0.0
        last_volume = 0.0
        if self.clearing_results:
            last = self.clearing_results[-1]
            last_avg_price = last.average_price
            last_volume = last.total_traded_kwh

        return {
            "best_bid": best_bid,
            "best_ask": best_ask if best_ask != float("inf") else 0.0,
            "last_avg_price": last_avg_price,
            "last_volume": last_volume,
            "n_periods_completed": len(self.clearing_results),
        }

    def get_agent_trades(
        self, agent_id: int, period: Optional[int] = None
    ) -> List[PartialTrade]:
        """Retrieve trades for a specific agent.

        If ``period`` is None, return all historical trades.
        If ``period`` is an integer index (may be negative), filter by that period.
        """
        if period is None:
            source = self.all_trades
        else:
            if not self.clearing_results:
                return []
            source = self.clearing_results[period].trades

        return [
            t for t in source
            if t.buyer_id == agent_id or t.seller_id == agent_id
        ]
