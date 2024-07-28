from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from config import env_config
from train.env.cripto.crypto_env import TradingCryptoEnv

if __name__ == '__main__':
    og_dir = "./check_freq"

    env = TradingCryptoEnv(env_config,is_test=True)
    env = Monitor(env)
    model = DQN.load("./modelos_pkl/DQN_ALL_INDIC_FILL_PNL_BTC.pkl")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")