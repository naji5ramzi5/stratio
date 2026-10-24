"""
Legacy Hydra dataset-loading facade (CoinMarket / Bitmex).

Imports are intentionally lazy: this package also hosts the modern
``binance_ohlcv`` loader used by the ML/evaluation stack, and eager imports
here (``Bitmex`` needs the optional ``bitmex`` package) would break
``from data_loader.binance_ohlcv import ...`` for everyone.
"""
from datetime import datetime

DATASETS = ['CoinMarket', 'Bitmex']
DATA_TYPES = ['train', 'validation', 'test']


def get_dataset(dataset_name, start_date, end_date, args):
    assert dataset_name in DATASETS, \
        f"Dataset {dataset_name} is not available."
    if dataset_name == 'CoinMarket':
        from .CoinMarketDataset import CoinMarketDataset
        main_features = ['High', 'Volume', 'Low', 'Close', 'Open', 'Mean']

        if start_date == "-1":
            start_date = None
        else:
            start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')

        if end_date == "-1":
            end_date = None
        else:
            end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

        btc = CoinMarketDataset(main_features=main_features, start_date=start_date,
                                end_date=end_date, window_size=args.dataset_loader.window_size)
        dataset = btc.get_dataset()

    elif dataset_name == 'Bitmex':
        from .Bitmex import BitmexDataset
        btc = BitmexDataset(args)
        dataset = btc.get_dataset()

    return dataset
