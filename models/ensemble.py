import numpy as np
import pandas as pd
from .xgboost import MyXGboost
from .random_forest import RandomForest

class MyEnsemble:
    def __init__(self, args):
        self.xgb = MyXGboost(args)
        self.rf = RandomForest(args)
        self.response_col = args.response_col
        self.date_col = args.date_col

    def fit(self, data_x):
        # Fit both models
        self.xgb.fit(data_x)
        self.rf.fit(data_x)

    def predict(self, test_x):
        pred_xgb = self.xgb.predict(test_x)
        pred_rf = self.rf.predict(test_x)
        # Average the predictions
        return (pred_xgb + pred_rf) / 2
