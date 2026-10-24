"""
Hyper-parameter tuning for the stable-baselines3 PPO path, using optuna.

Splits the data into train / validation / test.  Each trial trains an sb3 PPO
on the train slice for a modest budget, evaluates excess-Sharpe on the
validation slice, and optuna picks the best hyper-parameters.  A final model is
trained on train+validation and scored on the held-out test slice.

Usage:
    from reinforcement import tune_ppo as tp
    best, report = tp.tune_ppo(price, tech, env_params, "ppo_models/tune",
                               n_trials=8, trial_steps=20000, final_steps=40000)
"""

import os

import numpy as np

try:
    from . import finrl_ppo as fp
    from . import sb3_ppo as sp
except ImportError:
    import sys
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if _pkg_dir not in sys.path:
        sys.path.insert(0, _pkg_dir)
    import finrl_ppo as fp  # noqa: F401
    import sb3_ppo as sp  # noqa: F401

__all__ = ["tune_ppo"]


def _excess_metric(account, price_test, timeframe, lookback):
    """Score an agent: excess Sharpe over equal-weight benchmark, minus a
    penalty whenever its max drawdown undercuts the benchmark's drawdown by
    more than 10 percentage points (agents that ride positions into a crash)."""
    try:
        m = fp.evaluate_ppo(account, price_test, timeframe, lookback)
        excess = float(m.get("excess_sharpe", 0.0))
        dd_bot = float(m.get("max_drawdown_bot", 0.0) or 0.0)
        dd_eqw = float(m.get("max_drawdown_eqw", 0.0) or 0.0)
        extra_dd = dd_bot - dd_eqw  # negative when bot draws down harder than market
        penalty = max(0.0, -extra_dd - 0.10) * 3.0
        return float(np.clip(excess - penalty, -5.0, 5.0)), m
    except Exception:
        return -5.0, None


def tune_ppo(price_array, tech_array, env_params, workdir, n_trials=8,
             train_frac=0.70, val_frac=0.10, trial_steps=20000,
             final_steps=40000, timeframe="1h", seed=0, verbose=0):
    """Run an optuna search over PPO hyper-parameters.

    Returns (best_params, report_dict).
    """
    import optuna
    from optuna.samplers import TPESampler

    n = len(price_array)
    split_val = int(n * (train_frac + val_frac))
    split_train = int(n * train_frac)
    price_tr, tech_tr = price_array[:split_train], tech_array[:split_train]
    price_va, tech_va = price_array[split_train:split_val], tech_array[split_train:split_val]
    price_te, tech_te = price_array[split_val:], tech_array[split_val:]
    lookback = env_params["lookback"]
    for name, arr in (("val", price_va), ("test", price_te)):
        if len(arr) < lookback + 5:
            raise ValueError(f"{name} slice too small: {len(arr)} rows")

    os.makedirs(workdir, exist_ok=True)
    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner())

    def objective(trial):
        hp = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
            "ent_coef": trial.suggest_categorical("ent_coef",
                                                  [0.0, 0.001, 0.005, 0.01, 0.03]),
            "gamma": trial.suggest_categorical("gamma", [0.98, 0.99, 0.995]),
            "gae_lambda": trial.suggest_categorical("gae_lambda", [0.90, 0.95, 0.98]),
            "net_dimension": trial.suggest_categorical("net_dimension", [64, 128, 256]),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        }
        trial_dir = os.path.join(workdir, f"trial_{trial.number}")
        sp.train_ppo_sb3(price_tr, tech_tr, env_params, trial_dir,
                         total_timesteps=trial_steps, seed=seed,
                         verbose=verbose, **hp)
        account = sp.predict_ppo_sb3(price_va, tech_va, env_params, trial_dir)
        excess, _ = _excess_metric(account, price_va, timeframe, lookback)
        return excess

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    # final model on train + validation
    final_dir = os.path.join(workdir, "final")
    price_tv, tech_tv = price_array[:split_val], tech_array[:split_val]
    res = sp.train_ppo_sb3(price_tv, tech_tv, env_params, final_dir,
                           total_timesteps=final_steps, seed=seed,
                           verbose=verbose, **best)
    account = sp.predict_ppo_sb3(price_te, tech_te, env_params, final_dir)
    _, m = _excess_metric(account, price_te, timeframe, lookback)

    report = {
        "splits": {"train": len(price_tr), "val": len(price_va), "test": len(price_te)},
        "best_params": best,
        "trials": study.trials_dataframe().to_dict("records"),
        "test_metrics": m,
        "final_dir": final_dir,
        "final_model": res,
    }
    return best, report
