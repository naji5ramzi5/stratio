import unittest

from market_events.features import (
    EXAMPLE_CALCULATORS,
    MarketState,
    compute_all,
)
from market_events.model import make_book_snapshot, make_trade
from market_events.orderbook import OrderBook
from market_events.replay import ReplayEngine

EXCH = "binance_futures_um"
SYM = "BTCUSDT"


class TestMarketState(unittest.TestCase):
    def test_from_book(self):
        ev = make_book_snapshot(EXCH, SYM, 1_000_000,
                                [(100.0, 1.0), (99.0, 2.0)],
                                [(101.0, 1.0)], first_update_id=1, final_update_id=1,
                                receive_timestamp=2_000_000)
        book = OrderBook(EXCH, SYM)
        book.apply(ev)
        state = MarketState.from_book(book, 1_000_000, ev.event_type.value)
        self.assertEqual(state.best_bid, 100.0)
        self.assertEqual(state.best_ask, 101.0)
        self.assertEqual(state.event_time_us, 1_000_000)

    def test_from_none_book(self):
        state = MarketState.from_book(None, 1_000_000, "TRADE")
        self.assertIsNone(state.mid)
        self.assertIsNone(state.spread_bps)


class TestCalculators(unittest.TestCase):
    def setUp(self):
        self.book = OrderBook(EXCH, SYM)
        self.book.apply(make_book_snapshot(
            EXCH, SYM, 1_000_000,
            [(100.0, 1.0), (99.0, 2.0)], [(101.0, 1.0), (102.0, 2.0)],
            first_update_id=1, final_update_id=1, receive_timestamp=2_000_000))
        self.state = MarketState.from_book(self.book, 1_000_000, "BOOK")

    def test_mid(self):
        calc = EXAMPLE_CALCULATORS[0]
        self.assertEqual(calc.calculate(self.state)["mid"], 100.5)

    def test_spread_bps(self):
        calc = EXAMPLE_CALCULATORS[1]
        self.assertAlmostEqual(calc.calculate(self.state)["spread_bps"],
                               99.5, delta=0.5)

    def test_imbalance(self):
        calc = EXAMPLE_CALCULATORS[2]
        out = calc.calculate(self.state)["book_imbalance"]
        self.assertGreater(out, 0.0)
        self.assertLess(out, 1.0)

    def test_nan_when_no_book(self):
        state = MarketState.from_book(None, 1_000_000, "TRADE")
        for calc in EXAMPLE_CALCULATORS:
            for v in calc.calculate(state).values():
                self.assertTrue(v != v)  # NaN

    def test_compute_all_keys(self):
        out = compute_all(self.state)
        self.assertIsInstance(out, dict)
        self.assertEqual(set(out.keys()),
                         {"mid.mid", "spread_bps.spread_bps", "book_imbalance.book_imbalance"})


class TestFeaturePipeline(unittest.TestCase):
    def test_replay_feature_end_to_end(self):
        events = [
            make_trade(EXCH, SYM, 1_000_000, 100.0, 1.0, False, trade_id=1,
                       receive_timestamp=1_000_100),
            make_book_snapshot(EXCH, SYM, 2_000_000,
                               [(100.0, 1.0)], [(101.0, 1.0)],
                               first_update_id=1, final_update_id=1,
                               receive_timestamp=2_000_100),
        ]
        eng = ReplayEngine(events, build_book=True)
        eng.advance_to(2_000_000)
        state = MarketState.from_book(eng.current_market_state,
                                      eng.current_event_time, "BOOK")
        out = compute_all(state)
        self.assertEqual(out["mid.mid"], 100.5)
        # consumer sees only what the clock has reached:
        self.assertEqual(eng.event_sequence, 2)
        self.assertEqual(state.event_time_us, 2_000_000)


if __name__ == "__main__":
    unittest.main()
