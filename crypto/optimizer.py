import numpy as np
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


def optimize_xgb(X, y, n_trials=30):
    if not HAS_OPTUNA or len(X) < 100:
        return {}

    try:
        from xgboost import XGBRegressor
    except ImportError:
        return {}

    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': 42,
            'verbosity': 0,
        }
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            scores.append(-mean_absolute_error(y_val, pred))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    best = study.best_params
    logger.info(f"Optuna XGB best: {best} (score={study.best_value:.4f})")
    return best


def optimize_lgb(X, y, n_trials=30):
    if not HAS_OPTUNA or len(X) < 100:
        return {}

    try:
        import lightgbm as lgb
    except ImportError:
        return {}

    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            'random_state': 42,
            'verbose': -1,
        }
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            scores.append(-mean_absolute_error(y_val, pred))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    best = study.best_params
    logger.info(f"Optuna LGB best: {best} (score={study.best_value:.4f})")
    return best


def optimize_rf(X, y, n_trials=20):
    if not HAS_OPTUNA or len(X) < 100:
        return {}

    from sklearn.ensemble import RandomForestRegressor

    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 4, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
            'n_jobs': 1,
        }
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            model = RandomForestRegressor(**params)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            scores.append(-mean_absolute_error(y_val, pred))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    best = study.best_params
    logger.info(f"Optuna RF best: {best} (score={study.best_value:.4f})")
    return best