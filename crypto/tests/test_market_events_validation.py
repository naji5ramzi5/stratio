import unittest

from market_events import validation
from market_events.model import (
    make_book_snapshot,
    make_book_update,
    make_quote,
    make_trade,
)


def trade(ts=1_000_000, price=64_000.0, qty=1.0, recv=None, bm=False):
    return make_trade("binance_spot", "BTCUSDT", ts, price, qty, bm,
                      trade_id=1, receive_timestamp=recv)


class TestValidation(unittest.TestCase):
    def test_valid_trade(self):
        r = validation.validate(trade(recv=2_000_000))
        self.assertTrue(r.valid)
        self.assertEqual(r.reasons, [])

    def test_invalid_object(self):
        self.assertFalse(validation.validate("not an event").valid)

    def test_missing_receive_is_flagged_not_filled(self):
        r = validation.validate(trade())
        self.assertFalse(r.valid)
        self.assertIn(validation.REASON_RECEIVE, r.reasons)

    def test_negative_price(self):
        r = validation.validate(trade(price=-1.0, recv=2_000_000))
        self.assertIn(validation.REASON_PRICE, r.reasons)

    def test_zero_price(self):
        r = validation.validate(trade(price=0.0, recv=2_000_000))
        self.assertIn(validation.REASON_PRICE, r.reasons)

    def test_nan_price(self):
        r = validation.validate(trade(price=float("nan"), recv=2_000_000))
        self.assertIn(validation.REASON_PRICE, r.reasons)

    def test_negative_qty(self):
        r = validation.validate(trade(qty=-0.5, recv=2_000_000))
        self.assertIn(validation.REASON_QUANTITY, r.reasons)

    def test_zero_ts(self):
        r = validation.validate(trade(ts=0, recv=2_000_000))
        self.assertIn(validation.REASON_TIMESTAMP, r.reasons)

    def test_negative_ts(self):
        r = validation.validate(trade(ts=-5, recv=2_000_000))
        self.assertIn(validation.REASON_TIMESTAMP, r.reasons)

    def test_float_ts_rejected(self):
        r = validation.validate(trade(ts=1_000_000.5, recv=2_000_000))
        self.assertIn(validation.REASON_TIMESTAMP, r.reasons)

    def test_empty_symbol(self):
        ev = make_trade("binance_spot", "", 1_000_000, 1.0, 1.0, False,
                        receive_timestamp=2_000_000)
        self.assertIn(validation.REASON_ENVELOPE, validation.validate(ev).reasons)

    def test_crossed_quote(self):
        ev = make_quote("x", "Y", 1_000_000, 100.0, 99.0, receive_timestamp=2_000_000)
        r = validation.validate(ev)
        self.assertIn(validation.REASON_QUOTE_CROSSED, r.reasons)

    def test_valid_quote(self):
        ev = make_quote("x", "Y", 1_000_000, 100.0, 100.5, receive_timestamp=2_000_000)
        self.assertTrue(validation.validate(ev).valid)

    def test_book_snapshot_unsorted_bids(self):
        ev = make_book_snapshot("x", "Y", 1_000_000,
                                [(100.0, 2.0), (101.0, 1.0)],  # ascending = wrong
                                [(102.0, 1.0)], receive_timestamp=2_000_000)
        r = validation.validate(ev)
        self.assertIn(f"{validation.REASON_BOOK_SORT}:bids", r.reasons)

    def test_book_snapshot_crossed(self):
        ev = make_book_snapshot("x", "Y", 1_000_000,
                                [(105.0, 1.0)], [(104.0, 1.0)],
                                receive_timestamp=2_000_000)
        r = validation.validate(ev)
        self.assertIn(validation.REASON_BOOK_CROSSED, r.reasons)

    def test_book_snapshot_bad_level_price(self):
        ev = make_book_snapshot("x", "Y", 1_000_000,
                                [(0.0, 1.0)], [(102.0, 1.0)],
                                receive_timestamp=2_000_000)
        r = validation.validate(ev)
        self.assertTrue(any(x.startswith(validation.REASON_PRICE) for x in r.reasons))

    def test_book_update_zero_qty_valid(self):
        ev = make_book_update("x", "Y", 1_000_000,
                              [(100.0, 0.0)], [(102.0, 3.0)],
                              receive_timestamp=2_000_000)
        self.assertTrue(validation.validate(ev).valid)

    def test_negative_seq(self):
        ev = make_trade("x", "Y", 1_000_000, 1.0, 1.0, False,
                        trade_id=-3, receive_timestamp=2_000_000)
        self.assertIn(validation.REASON_SEQUENCE, validation.validate(ev).reasons)

    def test_no_silent_repair_check(self):
        ev = trade()
        r = validation.validate(ev)
        self.assertIsNone(ev.receive_timestamp)  # untouched after validation


if __name__ == "__main__":
    unittest.main()
