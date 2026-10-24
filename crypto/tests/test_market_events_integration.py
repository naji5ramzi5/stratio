"""Integration tests: the event layer against REAL downloaded data.

Skips cleanly when the arbitrage-layer datasets are not present, so the
suite still passes in a fresh checkout.
"""

import os
import unittest

from market_events.adapters import (
    FUTURES_EXCHANGE,
    SPOT_EXCHANGE,
    aggtrade_events_from_zip,
    day_aggtrade_events,
)
from market_events.model import EventType
from market_events.replay import ReplayClockError, ReplayEngine
from market_events.store import DEFAULT_ROOT, EventStore
from market_events.validation import validate

ARB_DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "arbitrage", "data",
)
SPOT_ZIP = os.path.join(ARB_DATA, "spot", "aggTrades", "BTCUSDT", "2026-08-09.zip")
FUT_DIR = os.path.join(ARB_DATA, "futures_um", "aggTrades")
DEPTH_JSONL = os.path.join(ARB_DATA, "orderbook", "2026-08-10", "btcusdt_depth.jsonl")


def _fut_zip():
    if not os.path.isdir(FUT_DIR):
        return None
    for sym in os.listdir(FUT_DIR):
        d = os.path.join(FUT_DIR, sym)
        if os.path.isdir(d):
            files = [f for f in os.listdir(d) if f.endswith(".zip")]
            if files:
                return os.path.join(d, sorted(files)[0])
    return None


@unittest.skipUnless(os.path.exists(SPOT_ZIP), "spot aggTrades data not present")
class TestRealTrades(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = list(aggtrade_events_from_zip(
            SPOT_ZIP, SPOT_EXCHANGE, "BTCUSDT", receive_timestamp=1))
        cls.rows = len(cls.events)

    def test_all_events_valid(self):
        for ev in self.events:
            r = validate(ev)
            self.assertTrue(r.valid, f"{r} at {ev}")
        self.assertGreater(self.rows, 100_000)

    def test_monotonic_timestamps(self):
        ts = [e.event_timestamp for e in self.events]
        for a, b in zip(ts, ts[1:]):
            self.assertLessEqual(a, b)

    def test_trade_semantics(self):
        for ev in self.events:
            self.assertIs(ev.event_type, EventType.TRADE)
            self.assertEqual(ev.exchange, "binance_spot")
            self.assertEqual(ev.symbol, "BTCUSDT")
            self.assertGreater(ev.payload.price, 0)
            self.assertGreater(ev.payload.quantity, 0)

    def test_replay_over_real_day(self):
        eng = ReplayEngine(iter(self.events), build_book=False)
        first = eng.advance_to(self.events[0].event_timestamp)
        self.assertIsNotNone(first)
        eng.advance_to(self.events[-1].event_timestamp)
        self.assertEqual(eng.event_sequence, len(self.events))
        self.assertEqual(eng.current_event_time, self.events[-1].event_timestamp)

    def test_replay_bar_walk(self):
        """Walk the day in 1-minute bars; state must stay PIT-correct."""
        eng = ReplayEngine(iter(self.events), build_book=False)
        t0 = self.events[0].event_timestamp
        bar = t0
        prev_events = 0
        for _ in range(60):
            last = eng.advance_to(bar + 60_000_000)
            self.assertIsNotNone(last)
            self.assertLessEqual(last.event_timestamp, bar + 60_000_000)
            self.assertGreaterEqual(eng.event_sequence, prev_events)
            prev_events = eng.event_sequence
            bar += 60_000_000
        # clock may never move backwards:
        with self.assertRaises(ReplayClockError):
            eng.advance_to(bar - 60_000_000)


@unittest.skipUnless(_fut_zip(), "futures aggTrades data not present")
class TestRealFutures(unittest.TestCase):
    def test_ms_header_parse(self):
        zip_path = _fut_zip()
        sym = zip_path.split(os.sep)[-2]
        events = list(aggtrade_events_from_zip(
            zip_path, FUTURES_EXCHANGE, sym, receive_timestamp=1))
        self.assertGreater(len(events), 1000)
        # ms->µs conversion sanity: future data must land in 2026
        ts0 = events[0].event_timestamp
        self.assertGreater(ts0, 1_750_000_000_000_000)  # > 2025-06 in µs
        self.assertLess(ts0, 1_800_000_000_000_000)  # < 2027-01 in µs
        for ev in events:
            self.assertTrue(validate(ev).valid)


@unittest.skipUnless(os.path.exists(DEPTH_JSONL), "recorder depth data not present")
class TestRealDepth(unittest.TestCase):
    def test_replay_recorder_day(self):
        import json

        from market_events.adapters import recorder_depth_row_to_event

        with open(DEPTH_JSONL, encoding="utf-8") as f:
            events = [recorder_depth_row_to_event(json.loads(line),
                                                  FUTURES_EXCHANGE, "BTCUSDT",
                                                  receive_timestamp=1)
                      for line in f]
        self.assertGreater(len(events), 100)
        for ev in events:
            self.assertTrue(validate(ev).valid)
        eng = ReplayEngine(events, build_book=True)
        eng.advance_to(events[-1].event_timestamp)
        book = eng.current_market_state
        self.assertIsNotNone(book)
        self.assertGreater(book.best_bid, 0)
        self.assertGreater(book.best_ask, book.best_bid)
        self.assertEqual(eng.event_sequence, len(events))
        # determinism: replay again into a fresh engine
        eng2 = ReplayEngine(iter(events), build_book=True)
        eng2.advance_to(events[-1].event_timestamp)
        self.assertEqual(eng.current_market_state.snapshot(),
                         eng2.current_market_state.snapshot())


@unittest.skipUnless(os.path.exists(SPOT_ZIP), "spot aggTrades data not present")
class TestRealStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import shutil
        import tempfile

        cls.tmp = tempfile.mkdtemp()
        cls.store = EventStore(os.path.join(cls.tmp, "cache"))
        cls.store.ingest_trade_day(SPOT_ZIP, "spot", "BTCUSDT", "2026-08-09",
                                   receive_timestamp=1)

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.tmp)

    def test_store_replay_matches_direct(self):
        direct = list(aggtrade_events_from_zip(
            SPOT_ZIP, SPOT_EXCHANGE, "BTCUSDT", receive_timestamp=1))
        eng = self.store.replay("spot", "BTCUSDT", "2026-08-09",
                                receive_timestamp=1)
        eng.advance_to(direct[-1].event_timestamp)
        self.assertEqual(eng.event_sequence, len(direct))

    def test_store_range_query(self):
        data = self.store.load_day("spot", "BTCUSDT", "2026-08-09")
        mid_ts = int(data["ts_us"][len(data["ts_us"]) // 2])
        r = self.store.range_query("spot", "BTCUSDT", "2026-08-09",
                                   start_us=mid_ts)
        self.assertEqual(int(r["ts_us"][0]), mid_ts)
        self.assertLessEqual(len(r["ts_us"]), len(data["ts_us"]))


if __name__ == "__main__":
    unittest.main()
