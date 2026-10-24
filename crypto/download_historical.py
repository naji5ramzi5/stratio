"""Download real historical data from Binance for backtesting.
Fetches actual historical OHLCV candles for the past 150+ days."""
import json
import os
import time
import logging
from datetime import datetime, timezone, timedelta
from data_loader.binance_ohlcv import fetch_klines

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("data_downloader")

DATA_DIR = "historical_data"
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "LTCUSDT", "LINKUSDT", "DOTUSDT",
]
TIMEFRAMES = ["1h"]  # Focus on 1h for the backtest
LOOKBACK_DAYS = 150  # Extra data for training windows

def download_historical_data():
    """Download historical candles for all symbols."""
    os.makedirs(DATA_DIR, exist_ok=True)

    metadata = {
        "download_time": datetime.now(timezone.utc).isoformat(),
        "symbols": SYMBOLS,
        "timeframes": TIMEFRAMES,
        "lookback_days": LOOKBACK_DAYS,
        "data": {}
    }

    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            key = f"{symbol}_{tf}"
            logger.info(f"Downloading {key}...")

            # Binance limit is 1000 candles per request
            # For 150 days at 1h = 3600 candles, need 4 requests
            all_candles = []
            limit = 1000

            # Fetch in chunks going backwards
            for i in range(4):
                candles = fetch_klines(symbol, tf, limit)
                if candles:
                    all_candles.extend(candles)
                time.sleep(0.3)  # rate limit

            # Remove duplicates and sort by timestamp
            seen = set()
            unique_candles = []
            for c in all_candles:
                ts = c[0]
                if ts not in seen:
                    seen.add(ts)
                    unique_candles.append(c)
            unique_candles.sort(key=lambda x: x[0])

            # Save to file
            filename = f"{DATA_DIR}/{key}.json"
            with open(filename, "w") as f:
                json.dump(unique_candles, f)

            metadata["data"][key] = {
                "file": filename,
                "count": len(unique_candles),
                "start_ts": unique_candles[0][0] if unique_candles else None,
                "end_ts": unique_candles[-1][0] if unique_candles else None,
                "start_date": datetime.fromtimestamp(unique_candles[0][0]/1000, tz=timezone.utc).isoformat() if unique_candles else None,
                "end_date": datetime.fromtimestamp(unique_candles[-1][0]/1000, tz=timezone.utc).isoformat() if unique_candles else None,
            }

            logger.info(f"  Saved {len(unique_candles)} candles ({metadata['data'][key]['start_date'][:10]} to {metadata['data'][key]['end_date'][:10]})")

    # Save metadata
    with open(f"{DATA_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nDownload complete! Data saved to {DATA_DIR}/")
    return metadata

if __name__ == "__main__":
    download_historical_data()
