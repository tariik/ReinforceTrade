from collections import deque

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from config import env_config, LOGGER
from data_pipeliine import DataPipeline
from portfolio import Portfolio
from reward import RewardClass

INVEST_VALUE = 0.05


class TradingCryptoEnv(gym.Env):
    def __init__(self, config, is_test=False, seed: int = 1):
        super(TradingCryptoEnv, self).__init__()

        # PARAMS

        self.initial_balance = config.get("initial_balance", 200)
        self.buy_fee = config.get("buy_fee", 0.001)
        self.sell_fee = config.get("sell_fee", 0.0015)
        self.borrow_interest_rate = config.get("borrow_interest_rate", 0.01)
        self.time_window = config.get("time_window", 5)
        self.window_size = config.get("window_size", 10)
        self.alpha = config.get("alpha", 0.6)
        self.train_data = config.get("train_data", '')
        self.test_data = config.get("test_data", '')
        self.symbol = config.get("symbol", '')
        self.timeframe = config.get("timeframe", '')
        self.exchanges = config.get("exchanges", [])
        self.max_steps = config.get("max_steps", [])
        self.reward_type = config.get("reward_type", 'default')
        self.reward_class = RewardClass()
        self.current_portfolio_value = self.initial_balance
        self.total_rows = 0
        self.B_t = 0.0
        self.A_t = 0.0

        self.reward = np.array([0.0], dtype=np.float32)
        self.step_reward = np.array([0.0], dtype=np.float32)

        self.done = False

        self.max_position = 1
        self.price_change = 0
        self.current_price = None
        self.data_step = 0
        self._seed = seed
        self._random_state = np.random.RandomState(seed=self._seed)
        self.prices = None
        self.normalized_data = None
        self.is_test = is_test

        self.data_pipeline = DataPipeline(
            self.is_test,
            self.train_data,
            self.test_data,
            self.symbol,
            self.timeframe,
            self.exchanges
        )

        self.entry_price = None
        # self.historical_info = []
        self.current_position = 0  # Posición actual del agente: 1 (long), 0 (wait), -1 (short)
        self.timer = 0  # Contador para el tiempo de congelación
        self.portfolio = Portfolio(asset=0, fiat=self.initial_balance)
        # buffer for appending lags
        self.data_buffer = deque(maxlen=self.window_size)
        self.data = None
        self.current_step = None
        self.state = None
        self.actions = np.eye(3, dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation, _ = self.reset()  # Reset to load observation.shape
        self.observation_space = spaces.Box(low=-10., high=10.,
                                            shape=self.observation.shape,
                                            dtype=np.float32)

    def reset(self, seed=None, options=bool):
        super().reset(seed=seed)
        self.current_step = 0
        self.prices, self.data, self.normalized_data = self.data_pipeline.load_ohcl_data()
        self.A_t, self.B_t = 0., 0.
        self.reward = 0.0
        self.done = False
        self.portfolio.reset(asset=0, fiat=self.initial_balance)
        self.data_buffer.clear()

        self.total_rows = len(self.data)
        self.data_step = np.random.randint(0, self.total_rows)
        if self.is_test:
            self.data_step = 0

        for step in range(self.window_size):
            self.current_price = self.prices[self.data_step]
            self.price_change = 0
            if self.prices[self.data_step - 1]:
                self.price_change = self.prices[self.data_step] - self.prices[self.data_step - 1]
            # Add current step's observation to the data buffer
            step_observation = self._get_step_observation(step_action=0)
            self.data_buffer.append(step_observation)

            self.data_step += 1

        state = np.asarray(self.data_buffer, dtype=np.float32)
        return state, {}

    def step(self, step_action):
        if self.current_step > self.max_steps or self.data_step + 1 > self.total_rows or self.current_portfolio_value < 100:
            self.done = True
            LOGGER.info(f" DONE")
            LOGGER.info(f"current_step: {self.current_step}")
            LOGGER.info(f"max_steps: {self.max_steps}")
            LOGGER.info(f"data_step: {self.data_step}")
            LOGGER.info(f"total_rows: {self.total_rows}")
            LOGGER.info(f"portfolio_value: {self.current_portfolio_value}")

            return self.observation, self.reward, self.done, False, {}

        LOGGER.info(f" Action taked: {step_action}")

        self.step_reward = 0.

        self.current_price = self.prices[self.data_step]
        self.price_change = self.prices[self.data_step] - self.prices[self.data_step - 1]

        investment_amount = self.portfolio.fiat * 0.05  # 5% of the balance

        if self.timer == 0:  # Si el timer es 0, el agente puede elegir una nueva acción
            
            self.timer = self.time_window  # Reiniciar el timer al tiempo de ventana
            if step_action == 1:
                self.current_price = self.data.loc[self.data_step, 'close']
                volatility = self.calculate_volatility().iloc[self.data_step]
                market_depth = self.data.loc[self.data_step, 'volume']  # Simplified proxy for market depth
                self._handle_buy_action(self.current_price, investment_amount, volatility, market_depth)
            elif step_action == 2:
                self.current_price = self.data.loc[self.data_step, 'close']
                volatility = self.calculate_volatility().iloc[self.data_step]
                market_depth = self.data.loc[self.data_step, 'volume']  # Simplified proxy for market depth
                self._handle_sell_action(self.current_price, investment_amount, volatility, market_depth)
            elif step_action == 0:
                pass  # Do nothing

            self.portfolio.update_interest(borrow_interest=self.borrow_interest_rate)
            self.current_portfolio_value = self.portfolio.valorisation(self.current_price)

            self.step_reward = self.current_portfolio_value

            # Calcular la recompensa basada en la diferencia de precios y la acción
            # if self.current_position != 0:  # Solo calcular la recompensa si hay una posición abierta
            #            price_change_reward = ((current_price - self.entry_price) / self.entry_price) * (
            #         1 if self.current_position == 1 else -1)
            # else:
            #     price_change_reward = 0
            # reward = self.combined_reward(price_change_reward, portfolio_value_reward)
            self.step_reward = self.get_reward()
        else:
            self.timer -= 1

        # Add current step's observation to the data buffer

        self.reward += self.step_reward
        # Add current step's observation to the data buffer
        step_observation = self._get_step_observation(step_action=step_action)
        self.data_buffer.append(step_observation)
        self.observation = np.asarray(self.data_buffer, dtype=np.float32)

        LOGGER.info(f"current_step reward: {self.reward}")
        self.data_step += 1
        self.current_step += 1
        return self.observation, self.reward, self.done, False, {}

    def combined_reward(self, price_change_reward, portfolio_value_reward):
        """
        Combina dos tipos de recompensas en una única métrica.

        Parámetros:
        price_change_reward (float): Recompensa basada en el cambio de precio y la acción tomada.
        portfolio_value_reward (float): Recompensa basada en el cambio del valor del portafolio.
        alpha (float): Factor de ponderación que determina la importancia relativa de cada recompensa.

        Retorna:
        float: La recompensa combinada.
        """
        return self.alpha * price_change_reward + (1 - self.alpha) * portfolio_value_reward

    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.portfolio.fiat}")
        print(f"Holdings: {self.portfolio.asset}")
        print(f"Portfolio Value: {self.portfolio.valorisation(self.state[3])}")
        print(f"Transaction History: {self.portfolio.get_transaction_history()}")

    def _handle_buy_action(self, current_price, investment_amount, volatility, market_depth):
        if not self.portfolio.is_long and not self.portfolio.is_short:
            self.portfolio.open_long(current_price, investment_amount, self.buy_fee, volatility, market_depth)
        elif self.portfolio.is_short:
            self.portfolio.close_short(current_price, self.buy_fee, volatility, market_depth)

    def _handle_sell_action(self, current_price, investment_amount, volatility, market_depth):
        if self.portfolio.is_long:
            self.portfolio.close_long(current_price, self.sell_fee, volatility, market_depth)
        elif not self.portfolio.is_short:
            self.portfolio.open_short(current_price, investment_amount, self.sell_fee, volatility, market_depth)

    def _check_done(self, portfolio_value):
        if self.current_step >= len(self.data) - 1 or portfolio_value < 100:
            return True
        return False

    def seed(self, seed: int = 1) -> list:
        """
        Set random seed in environment.

        :param seed: (int) random seed number
        :return: (list) seed number in a list
        """
        self._random_state = np.random.RandomState(seed=seed)
        self._seed = seed
        return [seed]

    def _get_step_observation(self, step_action):
        step_environment_observation = self.normalized_data[self.data_step]
        step_position_features = self._create_position_features()
        step_action_features = self._create_action_features(action=step_action)
        observation = np.concatenate((step_environment_observation,
                                      step_position_features,
                                      step_action_features,
                                      self.step_reward),
                                     axis=None)

        return np.clip(observation, -10., 10.)

    def _create_position_features(self):
        return np.array((self.portfolio.net_inventory_count / self.max_position,
                         self.portfolio.realized_pnl * self.portfolio.pct_scale,
                         self.portfolio.get_unrealized_pnl(self.current_price)
                         * self.portfolio.pct_scale),
                        dtype=np.float32)

    def _create_action_features(self, action: int) -> np.ndarray:
        return self.actions[action]

    def get_reward(self):
        step_reward = 0.0

        if self.reward_type == 'default_with_fills':
            inventory_count = self.portfolio.net_inventory_count
            midpoint_change = self.price_change
            step_pnl = self.portfolio.realized_pnl

            step_reward = self.reward_class.default_with_fills(
                    inventory_count,
                    midpoint_change,
                    step_pnl
                )

        return step_reward

    def calculate_volatility(self):
        # Usar ventana móvil de los últimos 10 días para el cálculo del ATR
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        true_ranges = pd.DataFrame({'HL': high_low, 'HC': high_close, 'LC': low_close})
        atr = true_ranges.rolling(window=10).mean().mean(axis=1)
        return atr


if __name__ == "__main__":
    env = TradingCryptoEnv(env_config)
    done = False
    total_reward = 0
    obs = env.reset()
    while not done:
        if isinstance(obs, tuple):
            obs = obs[0]  # Extract the tensor from the tuple
        # action, _, _ = test_agent.compute_single_action(obs)
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        LOGGER.info(f" Reward: {reward}")
        # test_agent.env.render()

    LOGGER.info(f"Total Reward: {total_reward}")
