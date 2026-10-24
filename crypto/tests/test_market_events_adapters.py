import unittest

from market_events import model
from market_events.adapters import (
    FUTURES_EXCHANGE,
    SPOT_EXCHANGE,
    aggtrade_row_to_event,
    aggtrade_rows_from_zip,
    book_ticker_row_to_quote,
    recorder_depth_row_to_event,
)

SPOT_ROW = ["11258210202242", "64116.30", "0.001", "1", "2", "1786394553105000",
            "False", "False"]
FUTURES_ROW = ["11258210202242", "64116.30", "0.001", "1", "2", "1786394553105",
               "false"]


class TestAgTradeAdapter(unittest.TestCase):
    def test_spot_row(self):
        ev = aggtrade_row_to_event(SPOT_ROW, SPOT_EXCHANGE, "BTCUSDT",
                                   receive_timestamp=1, has_header=False)
        self.assertEqual(ev.event_type, model.EventType.TRADE)
        self.assertEqual(ev.exchange, "binance_spot")
        self.assertEqual(ev.event_timestamp, 1786394553105000)
        self.assertEqual(ev.payload.price, 64116.30)
        self.assertEqual(ev.payload.quantity, 0.001)
        self.assertFalse(ev.payload.is_buyer_maker)
        self.assertEqual(ev.sequence_number, 11258210202242)
        self.assertEqual(ev.source, "binance_vision_aggtrades")

    def test_futures_row_ms_conversion(self):
        ev = aggtrade_row_to_event(FUTURES_ROW, FUTURES_EXCHANGE, "BTCUSDT",
                                   receive_timestamp=1, has_header=True)
        self.assertEqual(ev.event_timestamp, 1786394553105000)
        self.assertEqual(ev.exchange, "binance_futures_um")

    def test_buyer_maker_true_variants(self):
        ev = aggtrade_row_to_event(SPOT_ROW, SPOT_EXCHANGE, "X", 1, has_header=False)
        self.assertFalse(ev.payload.is_buyer_maker)
        row = SPOT_ROW.copy()
        row[6] = "True"
        ev = aggtrade_row_to_event(row, SPOT_EXCHANGE, "X", 1, has_header=False)
        self.assertTrue(ev.payload.is_buyer_maker)


class TestRecorderDepthAdapter(unittest.TestCase):
    def test_row(self):
        row = {"t": 1786394553105, "U": 11258210196395, "u": 11258210202242,
               "b": [["64116.30", "23.570"], ["64116.20", "1.268"]],
               "a": [["64116.40", "1.925"]]}
        ev = recorder_depth_row_to_event(row, "binance_futures_um", "BTCUSDT",
                                         receive_timestamp=1)
        self.assertEqual(ev.event_type, model.EventType.ORDER_BOOK_SNAPSHOT)
        self.assertEqual(ev.event_timestamp, 1786394553105000)
        self.assertEqual(ev.sequence_number, 11258210196395)
        self.assertEqual(ev.payload.bids, [(64116.30, 23.570), (64116.20, 1.268)])
        self.assertEqual(ev.payload.asks, [(64116.40, 1.925)])
        self.assertTrue(ev.payload.partial)
        self.assertEqual(ev.payload.first_update_id, 11258210196395)
        self.assertEqual(ev.payload.final_update_id, 11258210202242)


class TestBookTickerAdapter(unittest.TestCase):
    def test_row(self):
        row = {"u": 123, "s": "BTCUSDT", "b": "64116.30", "B": "2.5",
               "a": "64116.40", "A": "1.1", "T": 1786394553105}
        ev = book_ticker_row_to_quote(row, "binance_futures_um",
                                      receive_timestamp=1)
        self.assertEqual(ev.event_type, model.EventType.QUOTE)
        self.assertEqual(ev.event_timestamp, 1786394553105000)
        self.assertEqual(ev.payload.bid, 64116.30)
        self.assertEqual(ev.payload.ask, 64116.40)
        self.assertEqual(ev.sequence_number, 123)

    def test_symbol_from_row(self):
        row = {"u": 1, "s": "ETHUSDT", "b": "1", "B": "1", "a": "2", "A": "1",
               "T": 1}
        ev = book_ticker_row_to_quote(row, "x")
        self.assertEqual(ev.symbol, "ETHUSDT")

    def test_missing_symbol_raises(self):
        with self.assertRaises(ValueError):
            book_ticker_row_to_quote({"u": 1, "b": "1", "B": "1", "a": "2",
                                      "A": "1", "T": 1}, "x")


if __name__ == "__main__":
    unittest.main()
