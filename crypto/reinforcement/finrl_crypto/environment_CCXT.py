"""
CryptoEnvCCXT — custom RL environment for trading multiple cryptocurrencies.

Self-contained port of FinRL_Crypto's `environment_CCXT.py`
(https://github.com/AI4Finance-Foundation/FinRL_Crypto) with the CCXT / API-key
dependencies removed: orders are simulated internally (buy/sell fees), so the
same environment works for training and for backtesting on historical OHLCV.

The reward is the change in total asset MINUS the change of an equal-weight
benchmark portfolio (`delta_bot - delta_eqw`), which teaches the agent to beat
a simple buy-and-hold baseline instead of just going long the market.

State  = [cash, stocks, lookback x technical_indicator rows]
Action = crypto units per asset (continuous, normalized by order of magnitude)
"""

import math

import numpy as np

# Minimum crypto order limits (BTC, ETH, NEAR, LINK, LTC, MATIC, UNI, SOL, ...)
DEFAULT_CRYPTO_LIMITS = np.array([0.0001, 0.001, 0.01, 0.1, 0.1, 1.0, 0.1, 0.01, 0.1, 0.001])


class CryptoEnvCCXT:
    """Custom env for crypto trading (CCXT-based design, fully simulated)."""

    def __init__(self, config, env_params, initial_capital=1000000,
                 buy_cost_pct=0.001, sell_cost_pct=0.001, gamma=0.99, if_log=False,
                 exchange_name='binance'):

        self.if_log = if_log
        self.env_params = env_params
        self.lookback = env_params['lookback']
        self.initial_total_asset = initial_capital
        self.initial_cash = initial_capital
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.gamma = gamma
        self.exchange_name = exchange_name

        # In live trading this would hold a CCXT exchange client.
        self.exchange = None

        # Get initial price array to compute eqw
        self.price_array = config['price_array']
        self.prices_initial = list(self.price_array[0, :])
        self.equal_weight_stock = np.array([self.initial_cash /
                                            len(self.prices_initial) /
                                            self.prices_initial[i] for i in
                                            range(len(self.prices_initial))])

        # read normalization of cash, stocks and tech
        self.norm_cash = env_params['norm_cash']
        self.norm_stocks = env_params['norm_stocks']
        self.norm_tech = env_params['norm_tech']
        self.norm_reward = env_params['norm_reward']
        self.norm_action = env_params['norm_action']

        # Initialize constants
        self.tech_array = config['tech_array']
        self._generate_action_normalizer()
        self.crypto_num = self.price_array.shape[1]
        self.max_step = self.price_array.shape[0] - self.lookback - 1

        # reset
        self.time = self.lookback - 1
        # Ensure we don't go out of bounds
        if self.time >= self.price_array.shape[0]:
            self.time = self.price_array.shape[0] - 1
        self.cash = self.initial_cash
        self.current_price = self.price_array[self.time]
        self.current_tech = self.tech_array[self.time]
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)
        self.stocks_cooldown = None
        self.safety_factor_stock_buy = 1 - 0.05  # 5% safety factor for crypto

        self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        self.total_asset_eqw = np.sum(self.equal_weight_stock * self.price_array[self.time])

        self.episode_return = 0.0
        self.gamma_return = 0.0

        '''env information'''
        self.env_name = 'MultiCryptoEnvCCXT'

        # state_dim = cash[1,1] + stocks[1,4] + tech_array[1,44] * lookback
        self.state_dim = 1 + self.price_array.shape[1] + self.tech_array.shape[1] * self.lookback
        self.action_dim = self.price_array.shape[1]

        crypto_limits = env_params.get('crypto_limits', DEFAULT_CRYPTO_LIMITS)
        if hasattr(crypto_limits, '__len__'):
            self.minimum_qty_crypto = (np.asarray(crypto_limits)[:self.action_dim] * 1.1)  # 10% safety factor
        else:
            # Default minimum if not configured
            self.minimum_qty_crypto = np.ones(self.action_dim) * 0.001

        self.if_discrete = False
        self.target_return = 10 ** 8

        # Store ticker symbols for order book keeping
        self.ticker_list = config.get('ticker_list', [f'BTC/USDT', f'ETH/USDT'])

    def reset(self) -> np.ndarray:
        self.time = self.lookback - 1
        # Ensure we don't go out of bounds with small datasets
        if self.time >= self.price_array.shape[0]:
            self.time = min(self.lookback - 1, self.price_array.shape[0] - 1)

        self.current_price = self.price_array[self.time]
        self.current_tech = self.tech_array[self.time]
        self.cash = self.initial_cash  # reset()
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)
        self.stocks_cooldown = np.zeros_like(self.stocks)
        self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()

        state = self.get_state()
        return state

    def step(self, actions) -> (np.ndarray, float, bool, None):
        self.time += 1

        # Crypto markets are 24/7, so cooldown logic is different
        # Reduced cooldown for crypto markets
        for i in range(len(actions)):
            if self.stocks[i] > 0:
                self.stocks_cooldown[i] += 1

        price = self.price_array[self.time]
        for i in range(self.action_dim):
            norm_vector_i = self.action_norm_vector[i]
            actions[i] = actions[i] * norm_vector_i

        # Compute actions in crypto units (not dollars like in stock version)
        actions_crypto = actions  # Already in crypto units due to normalization

        # Sell
        #######################################################################################################
        #######################################################################################################
        #######################################################################################################

        for index in np.where(actions < -self.minimum_qty_crypto)[0]:

            if self.stocks[index] > 0:

                if price[index] > 0:  # Sell only if current asset is > 0
                    sell_amount = min(self.stocks[index], -actions[index])

                    assert sell_amount >= 0, "Negative sell!"

                    # In backtest mode, we simulate the sale
                    self.stocks_cooldown[index] = 0
                    self.stocks[index] -= sell_amount
                    self.cash += price[index] * sell_amount * (1 - self.sell_cost_pct)

                    # In live trading, this would be:
                    # try:
                    #     symbol = self.ticker_list[index]
                    #     order = self.exchange.create_market_sell_order(symbol, sell_amount)
                    # except Exception as e:
                    #     print(f"Failed to sell {symbol}: {e}")

        # Adaptive cooldown for crypto (shorter than stocks due to 24/7 market)
        for index in np.where(self.stocks_cooldown >= 24)[0]:  # 2 hours instead of 12 hours
            sell_amount = self.stocks[index] * 0.05  # Sell 5%
            self.stocks_cooldown[index] = 0
            self.stocks[index] -= sell_amount
            self.cash += price[index] * sell_amount * (1 - self.sell_cost_pct)

        # Buy
        #######################################################################################################
        #######################################################################################################
        #######################################################################################################

        for index in np.where(actions > self.minimum_qty_crypto)[0]:
            if price[index] > 0:  # Buy only if the price is > 0

                fee_corrected_cash = self.cash / (1 + self.buy_cost_pct)
                max_crypto_can_buy = (fee_corrected_cash / price[index]) * self.safety_factor_stock_buy
                buy_amount = min(max_crypto_can_buy, actions_crypto[index])

                if buy_amount < self.minimum_qty_crypto[index]:
                    buy_amount = 0

                self.stocks[index] += buy_amount
                self.cash -= price[index] * buy_amount * (1 + self.buy_cost_pct)

                # In live trading, this would be:
                # try:
                #     symbol = self.ticker_list[index]
                #     order = self.exchange.create_market_buy_order(symbol, buy_amount)
                # except Exception as e:
                #     print(f"Failed to buy {symbol}: {e}")

        """update time"""
        done = self.time == self.max_step
        state = self.get_state()
        next_total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        next_total_asset_eqw = np.sum(self.equal_weight_stock * self.price_array[self.time])

        # Difference in portfolio value + cooldown management
        delta_bot = next_total_asset - self.total_asset
        delta_eqw = next_total_asset_eqw - self.total_asset_eqw

        # Reward function
        reward = (delta_bot - delta_eqw) * self.norm_reward
        self.total_asset = next_total_asset
        self.total_asset_eqw = next_total_asset_eqw

        self.gamma_return = self.gamma_return * self.gamma + reward
        self.cumu_return = self.total_asset / self.initial_cash

        if done:
            reward = self.gamma_return
            self.episode_return = self.total_asset / self.initial_cash

        return state, reward, done, None

    def get_state(self):
        state = np.hstack((self.cash * self.norm_cash, self.stocks * self.norm_stocks))
        for i in range(self.lookback):
            # Ensure we don't go out of bounds
            idx = max(0, self.time - i)
            tech_i = self.tech_array[idx]
            normalized_tech_i = tech_i * self.norm_tech
            state = np.hstack((state, normalized_tech_i)).astype(np.float32)
        return state

    def close(self):
        if hasattr(self.exchange, 'close'):
            self.exchange.close()

    def _generate_action_normalizer(self):
        action_norm_vector = []
        price_0 = self.price_array[0]
        for price in price_0:
            x = math.floor(math.log(price, 10))  # the order of magnitude
            action_norm_vector.append(1 / ((10) ** x))

        action_norm_vector = np.asarray(action_norm_vector) * self.norm_action
        self.action_norm_vector = np.asarray(action_norm_vector)

    def get_portfolio_value(self):
        """Get current portfolio value in USD"""
        crypto_value = (self.stocks * self.price_array[self.time]).sum()
        return self.cash + crypto_value

    def get_positions(self):
        """Get current cryptocurrency positions"""
        positions = {}
        for i, ticker in enumerate(self.ticker_list):
            if self.stocks[i] > 0:
                positions[ticker] = {
                    'amount': self.stocks[i],
                    'value_usd': self.stocks[i] * self.price_array[self.time][i]
                }
        return positions
