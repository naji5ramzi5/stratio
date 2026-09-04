import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


class MyLightGBM:
    def __init__(self, args):
        self.response_col = args.response_col
        self.date_col = args.date_col
        self.regressors = []
        self.model = None
        self.params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_samples': 20,
            'verbose': -1,
            'random_state': 42,
        }

    def fit(self, data_x):
        if not HAS_LGB:
            raise ImportError("lightgbm not installed")
        self.regressors = [col for col in data_x.columns
                           if col != self.response_col and col != self.date_col]
        train_x = data_x[self.regressors].astype(float)
        train_y = data_x[self.response_col].astype(float)
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(train_x, train_y)

    def predict(self, test_x):
        if not HAS_LGB or self.model is None:
            return np.zeros(len(test_x))
        valid_x = test_x[self.regressors].astype(float)
        return self.model.predict(valid_x)