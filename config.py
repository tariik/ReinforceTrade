import logging

# singleton for logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
LOGGER = logging.getLogger('crypto_rl_log')
env_config = {
    "initial_balance": 2000,
    "buy_fee": 0.001,
    "sell_fee": 0.0015,
    "borrow_interest_rate": 0.01,
    "window_size": 10,
    "time_window": 5,
    "alpha": 0.6,
    "max_steps": 1000,
    'train_data': 'D:\\Lab\\quant\\code\\crypto-rl\\data\\train\\',
    'test_data': 'D:\\Lab\\quant\\code\\crypto-rl\\data\\test\\',
    'symbol': 'BTC',
    'timeframe': '1hour',
    'exchanges': ['binance', 'bybit', 'okx', 'kraken', 'coinbase'],
    'reward_type': 'default_with_fills'
}