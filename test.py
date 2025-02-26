from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from config import env_config_test
from environment.crypto_trade.cripo_env import TradingCryptoEnv

# load modelo from file


if __name__ == "__main__":
    env = TradingCryptoEnv(env_config_test)
    env =Monitor(env)
    model = PPO.load("./modelos/sac_model_binance_1d.pkl")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
