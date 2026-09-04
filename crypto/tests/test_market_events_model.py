import unittest

from market_events import model, timestamps
from market_events.model import (
    EventType,
    make_book_snapshot,
    make_book_update,
    make_quote,
    make_trade,
)


class TestTimestamps(unittest.TestCase):
    def test_normalize_identity(self):
        self.assertEqual(timestamps.normalize(1_234_567), 1_234_567)

    def test_normalize_ms(self):
        self.assertEqual(timestamps.normalize(1_234, "ms"), 1_234_000)

    def test_normalize_s(self):
        self.assertEqual(timestamps.normalize(1_234, "s"), 1_234_000_000)

    def test_normalize_none_stays_none(self):
        self.assertIsNone(timestamps.normalize(None, "ms"))

    def test_bad_unit_raises(self):
        with self.assertRaises(ValueError):
            timestamps.normalize(1, "days")

    def test_us_from_ms_lossless(self):
        self.assertEqual(timestamps.us_from_ms(1786394553105), 1786394553105000)


class TestModel(unittest.TestCase):
    def test_make_trade(self):
        ev = make_trade(
            "binance_spot", "BTCUSDT", 1_000_000, 64_000.5, 0.1,
            True, trade_id=42, receive_timestamp=2_000_000,
        )
        self.assertIs(ev.event_type, EventType.TRADE)
        self.assertEqual(ev.exchange, "binance_spot")
        self.assertEqual(ev.symbol, "BTCUSDT")
        self.assertEqual(ev.event_timestamp, 1_000_000)
        self.assertEqual(ev.receive_timestamp, 2_000_000)
        self.assertEqual(ev.sequence_number, 42)
        self.assertEqual(ev.payload.price, 64_000.5)
        self.assertEqual(ev.payload.quantity, 0.1)
        self.assertTrue(ev.payload.is_buyer_maker)

    def test_trade_sequence_falls_back_to_trade_id(self):
        ev = make_trade("x", "Y", 1, 2.0, 3.0, False, trade_id=7)
        self.assertEqual(ev.sequence_number, 7)

    def test_make_quote(self):
        ev = make_quote("binance_futures_um", "ETHUSDT", 1_000_000, 3_400.1, 3_400.5, 1.0, 2.0)
        self.assertIs(ev.event_type, EventType.QUOTE)
        self.assertEqual(ev.payload.bid, 3_400.1)
        self.assertEqual(ev.payload.ask, 3_400.5)
        self.assertEqual(ev.payload.bid_size, 1.0)

    def test_make_book_snapshot(self):
        ev = make_book_snapshot("binance_futures_um", "BTCUSDT", 1_000_000,
                                [(64_000.0, 1.0)], [(64_100.0, 2.0)],
                                first_update_id=10, final_update_id=20)
        self.assertEqual(ev.payload.bids, [(64_000.0, 1.0)])
        self.assertEqual(ev.sequence_number, 10)
        self.assertFalse(ev.payload.partial)

    def test_make_book_update_defaults(self):
        ev = make_book_update("binance_futures_um", "BTCUSDT", 1_000_000,
                              [(64_000.0, 0.0)], [(64_100.0, 1.0)])
        self.assertEqual(ev.payload.first_update_id, None)
        self.assertEqual(ev.payload.final_update_id, None)
        self.assertIsNone(ev.sequence_number)

    def test_event_time_alias(self):
        ev = make_trade("x", "Y", 555, 1.0, 1.0, False)
        self.assertEqual(ev.time, 555)

    def test_market_event_frozen(self):
        ev = make_trade("x", "Y", 1, 1.0, 1.0, False)
        with self.assertRaises(Exception):
            ev.symbol = "ZZZZ"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
