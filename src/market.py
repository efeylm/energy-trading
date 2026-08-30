"""
Partial Match Double Auction — makalede kullanılan tek piyasa mekanizması.

PartialMatchDoubleAuction (makale Bölüm 3.2):
    1. Alıcı teklifleri fiyata göre azalan, satıcı teklifleri artan sıralanır.
    2. En yüksek teklif ile en düşük istek eşleşir; takas fiyatı Eq. (2) uyarınca
       ikisinin tam orta noktasıdır.
    3. Bir istek birden çok teklife bölünebilir (kısmi eşleşme) ve tersi de geçerlidir.
    4. En iyi teklif en iyi isteğin altına düştüğünde eşleştirme durur.
    5. Eşleşmeyen artık miktar defterden düşer ve şebeke üzerinden (ToU / FiT) kapanır.

NOT: Bu dosyada daha önce iki ölü sınıf daha vardı — DoubleAuctionMarket
(Qiu et al. batch clearing) ve IterativeDoubleAuction (ADD Bitirme fiyat keşfi).
Simülasyon boru hattının hiçbir yerinden çağrılmıyorlardı; sadece hangi
mekanizmanın gerçekten kullanıldığı konusunda karışıklık yaratıyorlardı
(ör. artık gözlemde yer almayan best_bid / best_ask alanları). Kaldırıldılar.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Order:
    """A single buy or sell order."""
    agent_id: int
    price: float        # $/kWh willingness price
    quantity: float     # kWh (positive for both buy and sell)
    is_buy: bool        # True = buy order, False = sell order
    is_emergency: bool = False  # Emergency bid from starvation mechanism
    use_flat_price: bool = False  # True → tek toplu emir, MB eğrisi uygulanmaz


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
    """Double Auction with partial matching (No order-book persistence).

    Rules implemented:
    1. Sellers enter a fixed-quantity sell offer.
    2. Buyers are sorted by marginal benefit (bid price descending).
    3. Each seller's quantity is distributed across sorted buyers.
    4. Any unmatched quantity (buy or sell) is DISCARDED at the end of the period.
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


        # Step 4: Record unmatched orders for metrics (optional) but CLEAR the actual books
        # After matching, any leftover quantity is DISCARDED (No persistence/order book)
        result.pending_buy_orders = [o for o in self._buy_book if o.remaining_quantity > 1e-9]
        result.pending_sell_orders = [o for o in self._sell_book if o.remaining_quantity > 1e-9]
        
        self._buy_book = []
        self._sell_book = []

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
