import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from .xgboost import MyXGboost
from .random_forest import RandomForest

class MyEnsemble:
    def __init__(self, args):
        self.xgb = MyXGboost(args)
        self.rf = RandomForest(args)
        self.gbr = GradientBoostingRegressor(
            n_estimators=80, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        self.meta_learner = Ridge(alpha=1.0)
        self.response_col = args.response_col
        self.date_col = args.date_col
        self.regressors = []
        self.is_fitted = False

    def fit(self, data_x):
        self.regressors = [col for col in data_x.columns
                           if col != self.response_col and col != self.date_col]
        train_x = data_x[self.regressors].astype(float)
        train_y = data_x[self.response_col].astype(float)

        self.xgb.fit(data_x)
        self.rf.fit(data_x)
        self.gbr.fit(train_x, train_y)

        pred_xgb = self.xgb.predict(data_x)
        pred_rf = self.rf.predict(data_x)
        pred_gbr = self.gbr.predict(train_x)

        meta_features = np.column_stack([pred_xgb, pred_rf, pred_gbr])
        self.meta_learner.fit(meta_features, train_y)
        self.is_fitted = True

    def predict(self, test_x):
        valid_x = test_x[self.regressors].astype(float) if all(c in test_x.columns for c in self.regressors) else test_x

        pred_xgb = self.xgb.predict(test_x)
        pred_rf = self.rf.predict(test_x)
        pred_gbr = self.gbr.predict(valid_x)

        if self.is_fitted and len(pred_xgb) > 0:
            meta_features = np.column_stack([pred_xgb, pred_rf, pred_gbr])
            return self.meta_learner.predict(meta_features)

        return (pred_xgb + pred_rf + pred_gbr) / 3
