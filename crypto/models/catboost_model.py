import numpy as np

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


class MyCatBoost:
    def __init__(self, args):
        self.response_col = args.response_col
        self.date_col = args.date_col
        self.regressors = []
        self.model = None

    def fit(self, data_x):
        if not HAS_CATBOOST:
            raise ImportError("catboost not installed")
        self.regressors = [col for col in data_x.columns
                           if col != self.response_col and col != self.date_col]
        train_x = data_x[self.regressors].astype(float)
        train_y = data_x[self.response_col].astype(float)
        self.model = CatBoostRegressor(
            iterations=200,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            subsample=0.8,
            random_seed=42,
            verbose=False,
            early_stopping_rounds=20,
        )
        self.model.fit(train_x, train_y, verbose=False)

    def predict(self, test_x):
        if not HAS_CATBOOST or self.model is None:
            return np.zeros(len(test_x))
        valid_x = test_x[self.regressors].astype(float)
        return self.model.predict(valid_x)