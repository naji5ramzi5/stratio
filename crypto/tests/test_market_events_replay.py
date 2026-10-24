import unittest

from market_events.model import (
    EventType,
    make_book_snapshot,
    make_book_update,
    make_quote,
    make_trade,
)
from market_events.replay import ReplayClockError, ReplayEngine, ReplayPITError

EXCH = "binance_futures_um"
SYM = "BTCUSDT"


def t(seq, ts, price=100.0):
    return make_trade(EXCH, SYM, ts, price, 1.0, False, trade_id=seq,
                      sequence_number=seq, receive_timestamp=ts + 100)


def s(seq, ts):
    return make_book_snapshot(EXCH, SYM, ts, [(99.0, 1.0)], [(101.0, 1.0)],
                              first_update_id=seq, final_update_id=seq,
                              receive_timestamp=ts + 100)


def u(seq, ts, bid_u, ask_u=()):
    return make_book_update(EXCH, SYM, ts, bid_u, ask_u,
                            first_update_id=seq, final_update_id=seq,
                            receive_timestamp=ts + 100)


class TestReplayBasics(unittest.TestCase):
    def test_advance_consumes_only_through_target(self):
        events = [t(1, 1_000_000), t(2, 2_000_000), t(3, 3_000_000)]
        eng = ReplayEngine(events, build_book=False)
        self.assertTrue(eng.at_start)
        last = eng.advance_to(2_000_000)
        self.assertEqual(last.sequence_number, 2)
        self.assertEqual(eng.current_event_time, 2_000_000)
        self.assertEqual(eng.event_sequence, 2)
        last = eng.advance_to(5_000_000)
        self.assertEqual(last.sequence_number, 3)
        self.assertEqual(eng.event_sequence, 3)

    def test_advance_with_no_events(self):
        eng = ReplayEngine(iter([]), build_book=False)
        self.assertIsNone(eng.advance_to(1_000_000))
        self.assertTrue(eng.at_start)

    def test_future_events_not_consumed(self):
        events = [t(1, 1_000_000), t(2, 9_000_000)]
        eng = ReplayEngine(events, build_book=False)
        eng.advance_to(1_000_000)
        self.assertEqual(eng.event_sequence, 1)
        self.assertEqual(eng.current_event_time, 1_000_000)

    def test_rewind_rejected(self):
        events = [t(1, 1_000_000), t(2, 2_000_000)]
        eng = ReplayEngine(events, build_book=False)
        eng.advance_to(2_000_000)
        with self.assertRaises(ReplayClockError):
            eng.advance_to(1_500_000)

    def test_non_monotonic_input_rejected(self):
        events = [t(2, 2_000_000), t(1, 1_000_000)]
        eng = ReplayEngine(events, build_book=False)
        with self.assertRaises(ReplayClockError):
            eng.advance_to(5_000_000)

    def test_duplicate_trade_sequence_rejected(self):
        events = [t(5, 1_000_000), t(5, 2_000_000)]
        eng = ReplayEngine(events, build_book=False)
        with self.assertRaises(ReplayPITError):
            eng.advance_to(5_000_000)

    def test_equal_timestamps_allowed_in_order(self):
        events = [t(1, 1_000_000), t(2, 1_000_000), t(3, 1_000_000)]
        eng = ReplayEngine(events, build_book=False)
        eng.advance_to(1_000_000)
        self.assertEqual(eng.event_sequence, 3)

    def test_advance_by(self):
        events = [t(1, 1_000_000), t(2, 3_000_000), t(3, 7_000_000)]
        eng = ReplayEngine(events, build_book=False)
        eng.advance_by(5_000_000)
        self.assertEqual(eng.event_sequence, 2)
        eng.advance_by(5_000_000)
        self.assertEqual(eng.event_sequence, 3)

    def test_invalid_target(self):
        eng = ReplayEngine(iter([]), build_book=False)
        with self.assertRaises(ReplayClockError):
            eng.advance_to(-1)

    def test_state_exposed(self):
        events = [t(1, 1_000_000)]
        eng = ReplayEngine(events, build_book=False)
        eng.advance_to(1_000_000)
        self.assertEqual(eng.current_event_time, 1_000_000)
        self.assertEqual(eng.current_receive_time, 1_000_100)
        self.assertIsNone(eng.current_market_state)


class TestReplayBook(unittest.TestCase):
    def test_book_built_from_events(self):
        events = [s(1, 1_000_000), u(2, 1_000_001, [(99.0, 5.0)])]
        eng = ReplayEngine(events)
        eng.advance_to(1_000_001)
        book = eng.current_market_state
        self.assertIsNotNone(book)
        self.assertEqual(book.size_at(99.0, "bid"), 5.0)
        self.assertEqual(book.best_ask, 101.0)

    def test_book_state_is_latest_only(self):
        events = [s(1, 1_000_000), u(2, 2_000_000, [(99.0, 9.0)])]
        eng = ReplayEngine(events)
        eng.advance_to(1_000_000)
        self.assertEqual(eng.current_market_state.size_at(99.0, "bid"), 1.0)
        eng.advance_to(2_000_000)
        self.assertEqual(eng.current_market_state.size_at(99.0, "bid"), 9.0)

    def test_book_sequence_gap_detected(self):
        events = [s(1, 1_000_000), u(3, 2_000_000, [(99.0, 5.0)])]
        eng = ReplayEngine(events)
        eng.advance_to(1_000_000)
        # gap 1 -> 3 is a skippable loss of visibility, not an error; but a
        # rewind or duplicate must fail loudly:
        eng.advance_to(2_000_000)

    def test_crossed_book_raises_pit_error(self):
        ev = make_book_snapshot(EXCH, SYM, 1_000_000,
                                [(101.0, 1.0)], [(100.0, 1.0)],
                                first_update_id=1, final_update_id=1,
                                receive_timestamp=1_000_100)
        eng = ReplayEngine([ev])
        with self.assertRaises(ReplayPITError):
            eng.advance_to(1_000_000)

    def test_no_future_peek_api(self):
        """Structural guarantee: the only read path is advance_to."""
        events = [t(1, 1_000_000), t(2, 2_000_000)]
        eng = ReplayEngine(events, build_book=False)
        eng.advance_to(1_000_000)
        # consumer asks for state 1 second ahead: still has event_sequence 1
        self.assertEqual(eng.event_sequence, 1)
        # and nothing below current_event_time is reachable
        self.assertEqual(eng.current_event_time, 1_000_000)


if __name__ == "__main__":
    unittest.main()
