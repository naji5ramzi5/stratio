import io
import json
import os
import tempfile
import unittest
import zipfile

from market_events.replay import ReplayEngine
from market_events.store import EventStore

SPOT_HEADERLESS = (
    "1,64100.00,0.100,1001,2001,1786394553105000,False,False\n"
    "2,64101.00,0.200,1002,2002,1786394553106000,True,False\n"
    "3,64100.50,0.050,1003,2003,1786394553107000,False,False\n"
)
FUTURES_HEADER = (
    "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
    "10,64100.00,0.100,1,2,1786394553105,true\n"
    "11,64101.00,0.200,3,4,1786394553106,false\n"
)


def make_zip(path, content):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("day.csv", content)


class StoreTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "cache")
        self.store = EventStore(self.root)
        self.spot_zip = os.path.join(self.tmp.name, "BTCUSDT.zip")
        self.fut_zip = os.path.join(self.tmp.name, "ETHUSDT.zip")
        make_zip(self.spot_zip, SPOT_HEADERLESS)
        make_zip(self.fut_zip, FUTURES_HEADER)
        self.day = "2026-08-09"

    def tearDown(self):
        self.tmp.cleanup()


class TestIngest(StoreTestCase):
    def test_ingest_spot(self):
        m = self.store.ingest_trade_day(self.spot_zip, "spot", "BTCUSDT", self.day,
                                        receive_timestamp=1_000_000)
        self.assertEqual(m["rows"], 3)
        self.assertEqual(m["symbol"], "BTCUSDT")
        self.assertEqual(m["first_ts_us"], 1786394553105000)
        self.assertEqual(m["last_ts_us"], 1786394553107000)
        self.assertEqual(m["receive_ts_us"], 1_000_000)
        self.assertEqual(len(m["source_checksum"]), 64)
        self.assertTrue(self.store.has_day("spot", "BTCUSDT", self.day))

    def test_ingest_futures_header(self):
        m = self.store.ingest_trade_day(self.fut_zip, "futures_um", "ETHUSDT", self.day,
                                        receive_timestamp=1_000_000)
        self.assertEqual(m["rows"], 2)
        data = self.store.load_day("futures_um", "ETHUSDT", self.day)
        # ms converted to µs
        self.assertEqual(int(data["ts_us"][0]), 1786394553105000)
        self.assertTrue(bool(data["buyer_maker"][0]))
        self.assertFalse(bool(data["buyer_maker"][1]))

    def test_ingest_sorted(self):
        self.store.ingest_trade_day(self.spot_zip, "spot", "BTCUSDT", self.day,
                                    receive_timestamp=1_000_000)
        data = self.store.load_day("spot", "BTCUSDT", self.day)
        ts = data["ts_us"]
        self.assertTrue((ts[1:] >= ts[:-1]).all())

    def test_ingest_idempotent_same_checksum(self):
        m1 = self.store.ingest_trade_day(self.spot_zip, "spot", "BTCUSDT", self.day,
                                         receive_timestamp=1_000_000)
        m2 = self.store.ingest_trade_day(self.spot_zip, "spot", "BTCUSDT", self.day,
                                         receive_timestamp=1_000_000)
        self.assertEqual(m1["content_checksum"], m2["content_checksum"])
        self.assertEqual(m1["rows"], m2["rows"])

    def test_ingest_rejects_changed_source(self):
        self.store.ingest_trade_day(self.spot_zip, "spot", "BTCUSDT", self.day,
                                    receive_timestamp=1_000_000)
        make_zip(self.spot_zip, SPOT_HEADERLESS.replace("0.100", "0.999"))
        m = self.store.ingest_trade_day(self.spot_zip, "spot", "BTCUSDT", self.day,
                                        receive_timestamp=1_000_000)
        self.assertEqual(m["rows"], 3)
        data = self.store.load_day("spot", "BTCUSDT", self.day)
        self.assertEqual(float(data["qty"][0]), 0.999)

    def test_unknown_market_rejected(self):
        with self.assertRaises(ValueError):
            self.store.ingest_trade_day(self.spot_zip, "mars", "BTCUSDT", self.day)

    def test_empty_zip_rejected(self):
        empty = os.path.join(self.tmp.name, "EMPTY.zip")
        make_zip(empty, "idx\n")
        with self.assertRaises(ValueError):
            self.store.ingest_trade_day(empty, "spot", "BTCUSDT", self.day)


class TestReads(StoreTestCase):
    def setUp(self):
        super().setUp()
        self.store.ingest_trade_day(self.spot_zip, "spot", "BTCUSDT", self.day,
                                    receive_timestamp=1_000_000)

    def test_range_query_full(self):
        r = self.store.range_query("spot", "BTCUSDT", self.day)
        self.assertEqual(len(r["ts_us"]), 3)

    def test_range_query_slice(self):
        r = self.store.range_query("spot", "BTCUSDT", self.day,
                                   start_us=1786394553106000)
        self.assertEqual(len(r["ts_us"]), 2)
        r2 = self.store.range_query("spot", "BTCUSDT", self.day,
                                    start_us=1786394553106000,
                                    end_us=1786394553107000)
        self.assertEqual(len(r2["ts_us"]), 1)
        self.assertEqual(int(r2["price"][0]), 64101)
        # events exactly AT end_us are excluded ([start, end) semantics)
        r3 = self.store.range_query("spot", "BTCUSDT", self.day,
                                    start_us=1786394553107000)
        self.assertEqual(len(r3["ts_us"]), 1)
        self.assertEqual(float(r3["price"][0]), 64100.5)

    def test_range_query_empty(self):
        r = self.store.range_query("spot", "BTCUSDT", self.day,
                                   end_us=1)
        self.assertEqual(len(r["ts_us"]), 0)

    def test_load_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.store.load_day("spot", "ETHUSDT", self.day)

    def test_list_days(self):
        self.assertEqual(self.store.list_days("spot", "BTCUSDT"), [self.day])

    def test_manifest_missing(self):
        self.assertIsNone(self.store.manifest("spot", "ZZZ", self.day))

    def test_replay(self):
        eng = self.store.replay("spot", "BTCUSDT", self.day,
                                receive_timestamp=1_000_000)
        eng.advance_to(1786394553106000)
        self.assertEqual(eng.event_sequence, 2)
        self.assertEqual(eng.current_event_time, 1786394553106000)
        # duplicate trade sequences across a second replay are fine
        eng2 = self.store.replay("spot", "BTCUSDT", self.day)
        eng2.advance_to(1786394553107000)
        self.assertEqual(eng2.event_sequence, 3)


if __name__ == "__main__":
    unittest.main()
