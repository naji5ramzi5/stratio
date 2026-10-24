"""
SB3CryptoEnv — gymnasium adapter exposing CryptoEnvCCXT to stable-baselines3.

CryptoEnvCCXT follows the legacy 4-tuple step() convention used by the ElegantRL
fork.  stable-baselines3 requires gymnasium spaces and the 5-tuple
(obs, reward, terminated, truncated, info) API, so we wrap it here instead of
modifying the shared environment.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .environment_CCXT import CryptoEnvCCXT


class SB3CryptoEnv(gym.Env):
    """gymnasium wrapper for CryptoEnvCCXT (for stable-baselines3)."""

    def __init__(self, config, env_params, **kwargs):
        self.inner = CryptoEnvCCXT(config=config, env_params=env_params, **kwargs)
        self.reward_mode = env_params.get("reward_mode", "relative")
        self.action_dim = self.inner.action_dim
        self.norm_action = float(self.inner.norm_action)

        self.action_space = spaces.Box(
            low=-self.norm_action, high=self.norm_action,
            shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.inner.state_dim,), dtype=np.float32)
        self.metadata = {"render_modes": []}

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed % 2**32)
        return self.inner.reset().astype(np.float32), {}

    @property
    def cash(self):
        return self.inner.cash

    @property
    def stocks(self):
        return self.inner.stocks

    @property
    def price_array(self):
        return self.inner.price_array

    @property
    def time(self):
        return self.inner.time

    @property
    def total_asset(self):
        return self.inner.total_asset

    @property
    def total_asset_eqw(self):
        return self.inner.total_asset_eqw

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).copy()
        if self.reward_mode == "relative":
            prev_bot = self.inner.total_asset
            prev_eqw = self.inner.total_asset_eqw
        else:
            prev_bot = prev_eqw = None
        obs, inner_reward, done, _ = self.inner.step(action)
        if self.reward_mode == "relative":
            # return-based reward: portfolio log-return minus benchmark log-return
            if prev_bot > 0 and prev_eqw > 0:
                reward = (np.log(self.inner.total_asset / prev_bot)
                          - np.log(self.inner.total_asset_eqw / prev_eqw))
            else:
                reward = 0.0
        else:
            reward = float(inner_reward)
        return (obs.astype(np.float32), float(reward), bool(done), False, {})

    def render(self, mode="human"):
        return None

    def close(self):
        self.inner.close()


class SB3WeightEnv(gym.Env):
    """Target-weight rebalancing env for stable-baselines3 (sb3 path).

    Fixes the failure mode of SB3CryptoEnv: there the raw action is a crypto
    quantity, so a policy can express "buy a little every step" (which bleeds
    cash into a falling market) but has no clean "exit to cash".  Here the
    action is a target portfolio weight in [-1, 1] per asset (negative is
    clipped to 0 = cash; no shorting).  The portfolio is rebalanced each step
    toward the target; churn pays buy/sell costs, so holding is free and
    going to cash is explicit.  Reward is the portfolio log-return minus the
    equal-weight benchmark log-return (relative mode), same as SB3CryptoEnv.
    """

    def __init__(self, config, env_params, **kwargs):
        self.inner = CryptoEnvCCXT(config=config, env_params=env_params, **kwargs)
        self.reward_mode = env_params.get("reward_mode", "relative")
        self.action_dim = self.inner.action_dim
        self.lookback = self.inner.lookback
        self.max_step = self.inner.max_step
        self.state_dim = self.inner.state_dim
        self.initial_cash = float(self.inner.initial_cash)
        self.buy_cost_pct = self.inner.buy_cost_pct
        self.sell_cost_pct = self.inner.sell_cost_pct
        # only rebalance when the target weight differs from the current weight
        # by more than this fraction; kills hourly micro-churn / fee bleed
        self.rebalance_deadband = float(env_params.get("rebalance_deadband", 0.05))
        # reward shaping (robust mode): loss aversion + explicit churn penalty
        self.loss_penalty = float(env_params.get("loss_penalty", 0.0))
        self.churn_penalty = float(env_params.get("churn_penalty", 0.0))

        # action = target weight per asset in [-1, 1]
        self.norm_action = 1.0
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,), dtype=np.float32)
        self.metadata = {"render_modes": []}

        self.cash = self.initial_cash
        self.stocks = np.zeros(self.action_dim, dtype=np.float64)
        self.total_asset = self.initial_cash
        self.reset()

    def _mark(self, price):
        return float(self.cash + (self.stocks * price).sum())

    @property
    def price_array(self):
        return self.inner.price_array

    @property
    def time(self):
        return self.inner.time

    @property
    def total_asset_eqw(self):
        return self.inner.total_asset_eqw

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed % 2**32)
        obs = self.inner.reset().astype(np.float32)
        self.cash = self.initial_cash
        self.stocks = np.zeros(self.action_dim, dtype=np.float64)
        self.total_asset = self.initial_cash
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        prev_asset = self.total_asset
        prev_eqw = float(np.sum(self.inner.equal_weight_stock
                                * self.inner.price_array[self.inner.time]))
        if prev_asset <= 0 or prev_eqw <= 0:
            raise ValueError("portfolio/benchmark must stay positive")

        self.inner.time += 1
        price = self.inner.price_array[self.inner.time]

        # price moved first: mark current holdings at the new price
        market_value = self._mark(price)
        target_w = np.clip(action, 0.0, 1.0)
        cur_w = (self.stocks * price) / market_value
        rebalance = np.abs(target_w - cur_w) > self.rebalance_deadband
        target_shares = np.where(
            rebalance,
            target_w * market_value / np.where(price > 0, price, 1.0),
            self.stocks)

        delta = target_shares - self.stocks
        buys = delta > 0
        sells = delta < 0
        self.cash -= (delta * price * (1.0 + self.buy_cost_pct))[buys].sum()
        self.cash += (-delta * price * (1.0 - self.sell_cost_pct))[sells].sum()
        self.stocks = target_shares

        new_asset = self._mark(price)
        eqw_now = float(np.sum(self.inner.equal_weight_stock * price))
        bot_ret = float(np.log(new_asset / prev_asset))
        eqw_ret = float(np.log(eqw_now / prev_eqw))
        if self.reward_mode == "relative":
            reward = bot_ret - eqw_ret
            if self.loss_penalty or self.churn_penalty:
                churn = float(np.sum(np.abs(target_w - cur_w)))
                reward = reward + self.loss_penalty * min(0.0, bot_ret) \
                    - self.churn_penalty * churn
        else:
            reward = new_asset / prev_asset - 1.0

        self.total_asset = new_asset
        done = self.inner.time == self.max_step
        obs = self.inner.get_state().astype(np.float32)
        return obs, float(reward), bool(done), False, {}

    def render(self, mode="human"):
        return None

    def close(self):
        self.inner.close()
