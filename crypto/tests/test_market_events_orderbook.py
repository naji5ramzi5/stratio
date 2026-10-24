import unittest

from market_events.model import (
    EventType,
    make_book_snapshot,
    make_book_update,
)
from market_events.orderbook import (
    OrderBook,
    OrderBookCrossedError,
    OrderBookSequenceError,
    OrderBookValidationError,
)

EXCH = "binance_futures_um"
SYM = "BTCUSDT"


def snap(seq, bids, asks, partial=False, ts=1_000_000):
    return make_book_snapshot(EXCH, SYM, ts, bids, asks,
                              first_update_id=seq, final_update_id=seq,
                              receive_timestamp=2_000_000, partial=partial)


def upd(seq, bid_u, ask_u, ts=1_000_000):
    return make_book_update(EXCH, SYM, ts, bid_u, ask_u,
                            first_update_id=seq, final_update_id=seq,
                            receive_timestamp=2_000_000)


class TestOrderBookFullSnapshots(unittest.TestCase):
    def test_snapshot_replaces_state(self):
        book = OrderBook(EXCH, SYM)
        book.apply(snap(1, [(100.0, 5.0)], [(101.0, 5.0)]))
        self.assertEqual(book.best_bid, 100.0)
        self.assertEqual(book.best_ask, 101.0)
        self.assertEqual(book.mid, 100.5)
        self.assertAlmostEqual(book.spread_bps, 99.5, delta=0.5)
        book.apply(snap(2, [(200.0, 5.0)], [(201.0, 5.0)]))
        self.assertEqual(book.best_bid, 200.0)
        self.assertEqual(book.best_ask, 201.0)
        self.assertEqual(len(book), 2)

    def test_top_n_and_size(self):
        book = OrderBook(EXCH, SYM)
        book.apply(snap(1, [(100.0, 1.0), (99.9, 2.0), (99.8, 3.0)],
                        [(100.1, 1.0), (100.2, 2.0)]))
        self.assertEqual(book.top_n("bid", 2), [(100.0, 1.0), (99.9, 2.0)])
        self.assertEqual(book.top_n("ask", 2), [(100.1, 1.0), (100.2, 2.0)])
        self.assertEqual(book.size_at(100.0, "bid"), 1.0)
        self.assertEqual(book.size_at(99.0, "bid"), 0.0)

    def test_zero_qty_removes_level(self):
        book = OrderBook(EXCH, SYM)
        book.apply(snap(1, [(100.0, 5.0)], [(101.0, 5.0)]))
        book.apply(snap(2, [(100.0, 0.0)], [(101.0, 5.0)]))
        self.assertEqual(book.best_bid, None)
        self.assertEqual(book.size_at(100.0, "bid"), 0.0)

    def test_crossed_snapshot_rejected(self):
        book = OrderBook(EXCH, SYM)
        with self.assertRaises(OrderBookCrossedError):
            book.apply(snap(1, [(101.0, 5.0)], [(100.0, 5.0)]))


class TestOrderBookPartial(unittest.TestCase):
    def test_partial_replaces_levels_in_event(self):
        book = OrderBook(EXCH, SYM)
        book.apply(snap(1, [(100.0, 1.0), (99.0, 1.0)], [(101.0, 1.0), (102.0, 1.0)]))
        book.apply(snap(2, [(100.0, 2.0)], [(101.0, 2.0)], partial=True))
        self.assertEqual(book.size_at(100.0, "bid"), 2.0)
        self.assertEqual(book.size_at(101.0, "ask"), 2.0)
        # levels NOT in the partial event keep their previous state
        self.assertEqual(book.size_at(99.0, "bid"), 1.0)
        self.assertEqual(book.size_at(102.0, "ask"), 1.0)

    def test_partial_drops_levels_better_than_new_top(self):
        """A level better than the new visible top cannot exist after a
        top-N replacement — the exchange would have shown it."""
        book = OrderBook(EXCH, SYM)
        book.apply(snap(1, [(100.0, 1.0), (99.0, 1.0), (98.0, 1.0)],
                        [(101.0, 1.0), (102.0, 1.0), (103.0, 1.0)]))
        book.apply(snap(2, [(99.0, 2.0)], [(102.0, 2.0)], partial=True))
        self.assertEqual(book.size_at(100.0, "bid"), 0.0)  # stale better bid pruned
        self.assertEqual(book.size_at(99.0, "bid"), 2.0)
        self.assertEqual(book.size_at(98.0, "bid"), 1.0)
        self.assertEqual(book.size_at(101.0, "ask"), 0.0)  # stale better ask pruned
        self.assertEqual(book.size_at(102.0, "ask"), 2.0)
        self.assertEqual(book.size_at(103.0, "ask"), 1.0)


class TestOrderBookUpdates(unittest.TestCase):
    def test_diff_update(self):
        book = OrderBook(EXCH, SYM)
        book.apply(snap(1, [(100.0, 1.0), (99.0, 2.0)], [(101.0, 1.0), (102.0, 2.0)]))
        book.apply(upd(2, [(100.0, 3.0), (99.5, 1.0)], [(102.0, 0.0)]))
        self.assertEqual(book.size_at(100.0, "bid"), 3.0)
        self.assertEqual(book.size_at(99.5, "bid"), 1.0)
        self.assertEqual(book.size_at(102.0, "ask"), 0.0)
        self.assertEqual(book.best_ask, 101.0)

    def test_bad_price_rejected(self):
        book = OrderBook(EXCH, SYM)
        with self.assertRaises(OrderBookValidationError):
            book.apply(upd(1, [(-5.0, 1.0)], []))

    def test_bad_qty_rejected(self):
        book = OrderBook(EXCH, SYM)
        with self.assertRaises(OrderBookValidationError):
            book.apply(upd(1, [(100.0, -1.0)], []))

    def test_sequence_rewind_rejected(self):
        book = OrderBook(EXCH, SYM)
        book.apply(snap(10, [(100.0, 1.0)], [(101.0, 1.0)]))
        with self.assertRaises(OrderBookSequenceError):
            book.apply(snap(9, [(100.0, 1.0)], [(101.0, 1.0)]))

    def test_duplicate_sequence_rejected(self):
        book = OrderBook(EXCH, SYM)
        book.apply(snap(10, [(100.0, 1.0)], [(101.0, 1.0)]))
        with self.assertRaises(OrderBookSequenceError):
            book.apply(snap(10, [(100.0, 1.0)], [(101.0, 1.0)]))

    def test_wrong_symbol_rejected(self):
        book = OrderBook(EXCH, SYM)
        ev = make_book_snapshot(EXCH, "ETHUSDT", 1_000_000, [(1.0, 1.0)], [(2.0, 1.0)],
                                first_update_id=1, final_update_id=1,
                                receive_timestamp=2_000_000)
        with self.assertRaises(OrderBookValidationError):
            book.apply(ev)

    def test_trade_event_rejected(self):
        from market_events.model import make_trade
        book = OrderBook(EXCH, SYM)
        with self.assertRaises(OrderBookValidationError):
            book.apply(make_trade(EXCH, SYM, 1_000_000, 100.0, 1.0, False,
                                  trade_id=1, receive_timestamp=2_000_000))


class TestDeterminism(unittest.TestCase):
    def _build(self, events):
        book = OrderBook(EXCH, SYM)
        for ev in events:
            book.apply(ev)
        return book

    def test_same_events_same_state(self):
        events = [
            snap(1, [(100.0, 1.0), (99.0, 2.0)], [(101.0, 1.0), (102.0, 2.0)]),
            upd(2, [(100.0, 3.0)], [(102.0, 0.0)]),
            snap(3, [(100.0, 4.0), (99.5, 1.0)], [(101.5, 1.0)], partial=True),
            upd(4, [(99.0, 5.0)], []),
        ]
        b1 = self._build(events)
        b2 = self._build(events)
        self.assertEqual(b1.snapshot(), b2.snapshot())
        self.assertEqual(b1.top_n("bid", 5), b2.top_n("bid", 5))
        self.assertEqual(b1.top_n("ask", 5), b2.top_n("ask", 5))

    def test_version_increments(self):
        book = OrderBook(EXCH, SYM)
        v0 = book.version
        book.apply(snap(1, [(100.0, 1.0)], [(101.0, 1.0)]))
        self.assertEqual(book.version, v0 + 1)


if __name__ == "__main__":
    unittest.main()
