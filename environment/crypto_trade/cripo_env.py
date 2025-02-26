from gymnasium import spaces
import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

from config import env_config_test
from environment.crypto_trade.data_pipeline import DataPipeline
from environment.crypto_trade.plot_history import Visualize
from environment.crypto_trade.render_env import TradingGraph
from environment.crypto_trade.statistics import ExperimentStatistics
from portfolio import Portfolio

INVEST_VALUE = 0.05


class TradingCryptoEnv(gym.Env):
    def __init__(self, env_config):
        super(TradingCryptoEnv, self).__init__()

        self.symbol = 'BTCUSD'
        # PARAMS
        self.initial_balance = env_config.get("initial_balance", 200)
        self.buy_fee = env_config.get("buy_fee", 0.001)
        self.sell_fee = env_config.get("sell_fee", 0.0015)
        self.borrow_interest_rate = env_config.get("borrow_interest_rate", 0.01)

        self.time_window = env_config.get("time_window", 5)
        self.look_back_window = env_config.get("look_back_window", 5)
        self.alpha = env_config.get("alpha", 0.6)
        self.df = env_config.get("df")

        self.entry_price = None
        # self.historical_info = []
        self.current_position = 0  # Posición actual del agente: 1 (long), 0 (wait), -1 (short)
        self.timer = 0  # Contador para el tiempo de congelación

        self.data = None
        self.current_step = 0
        self.state = None
        self.action_space = spaces.Discrete(3)  # 0: No action, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.look_back_window * 5,), dtype=np.float32)

        # vis
        self.viz = Visualize(
            columns=['midpoint', 'buys', 'sells', 'inventory', 'realized_pnl'],
            store_historical_observations=True)

        # broker
        self.portfolio = None
        # get Broker class to keep track of PnL and orders
        # self.broker = Broker(max_position=max_position, transaction_fee=transaction_fee)
        self.portfolio = Portfolio(asset=0, fiat=self.initial_balance, buy_fee=self.buy_fee, sell_fee=self.sell_fee)
        self.episode_stats = ExperimentStatistics()

        # get historical data for simulations
        self.data_pipeline = DataPipeline(alpha=0.6)
        self._midpoint_prices, self._raw_data, self._normalized_data = self.data_pipeline.load_environment_data(
            fitting_file='data/1day/binance/',
            include_imbalances=True,
            as_pandas=True, )

        # rendering class
        self._render = TradingGraph(sym=self.symbol)

        # graph midpoint prices
        self._render.reset_render_data(
            y_vec=self._midpoint_prices[:np.shape(self._render.x_vec)[0]])

        self.reset()

    def pick_dataset(self):
        data = self.df
        return data[['open', 'high', 'low', 'close', 'volume']]  # Exclude 'Adj Close' if it's not needed

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.data = self.pick_dataset()
        self.portfolio = Portfolio(asset=0, fiat=self.initial_balance, buy_fee=self.buy_fee, sell_fee=self.sell_fee)
        self.state = self._get_observation()
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['close']  # Close price directly from the dataframe
        investment_amount = self.portfolio.fiat * 0.05  # 5% of the balance
        print(current_price)
        if self.timer == 0:  # Si el timer es 0, el agente puede elegir una nueva acción
            self.timer = self.time_window  # Reiniciar el timer al tiempo de ventana
            self.current_position = action
            self.entry_price = current_price
            if action == 1:
                self._handle_buy_action(current_price, investment_amount)
            elif action == 2:
                self._handle_sell_action(current_price, investment_amount)
            elif action == 0:
                pass  # Do nothing
        else:
            self.current_position = action  # Mantener la acción actual
            self.timer -= 1

        self.current_step += 1

        self.portfolio.update_interest(borrow_interest=self.borrow_interest_rate)
        portfolio_value = self.portfolio.valorisation(current_price)

        done = self._check_done(portfolio_value)

        portfolio_value_reward = portfolio_value - self.initial_balance
        # Calcular la recompensa basada en la diferencia de precios y la acción
        if self.current_position != 0:  # Solo calcular la recompensa si hay una posición abierta
            price_change_reward = ((current_price - self.entry_price) / self.entry_price) * (
                1 if self.current_position == 1 else -1)
        else:
            price_change_reward = 0

        reward = self.combined_reward(price_change_reward, portfolio_value_reward)
        reward = portfolio_value_reward
        print(f"initial_balance  : {self.initial_balance}")
        print(f"portfolio_value  : {portfolio_value}")
        print(f"portfolio_value_reward  : {portfolio_value_reward}")
        if not done:
            self.state = self._get_observation()

        return np.array(self.state, dtype=np.float32), reward, done, False, {}

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

    def _handle_buy_action(self, current_price, investment_amount):
        if not self.portfolio.is_long and not self.portfolio.is_short:
            self.portfolio.open_long(current_price, investment_amount, self.buy_fee)
        elif self.portfolio.is_short:
            self.portfolio.close_short(current_price, self.buy_fee)

    def _handle_sell_action(self, current_price, investment_amount):
        if self.portfolio.is_long:
            self.portfolio.close_long(current_price, self.sell_fee)
        elif not self.portfolio.is_short:
            self.portfolio.open_short(current_price, investment_amount, self.sell_fee)

    def _check_done(self, portfolio_value):
        if self.current_step >= len(self.data) - 1 or portfolio_value < 100:
            return True
        return False

    def _get_observation(self):
        if self.current_step < self.look_back_window:
            padding = np.zeros((self.look_back_window - self.current_step, 5))
            window = self.data.iloc[0:self.current_step].values
            observation = np.vstack((padding, window))
        else:
            observation = self.data.iloc[self.current_step - self.look_back_window:self.current_step].values
        return observation.flatten()


if __name__ == "__main__":
    env = TradingCryptoEnv(env_config_test)
    # If the environment don't follow the interface, an error will be thrown
    # check_env(env, warn=True)
    obs = env.reset()
    obs, reward, done, info, _ = env.step(1)
    print(obs)
