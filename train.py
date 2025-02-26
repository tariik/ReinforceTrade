from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from callback import SaveOnBestTrainingRewardCallback
from config import env_config_train
from environment.crypto_trade.cripo_env import TradingCryptoEnv
from net.modeli import CustomMLP

if __name__ == "__main__":
    # Define the environment
    env = TradingCryptoEnv(env_config_train)
    # Create log dir
    log_dir = "./check_freq"
    # Logs will be saved in log_dir/monitor.csv
    env = Monitor(env, log_dir)

    # Define the model
    policy_kwargs = dict(
        features_extractor_class=CustomMLP,
        features_extractor_kwargs=dict(features_dim=3),
        net_arch=[255, 255]
    )
    model = RecurrentPPO(
        'MlpLstmPolicy',
        env,
        # policy_kwargs=policy_kwargs,
        # learning_rate=0.0007550929113028352,
        # n_steps=2869,
        verbose=1,
        # batch_size=64,
        # n_epochs=10,
        # gamma=0.99,
        # gae_lambda=0.99,
        # ent_coef=0.002222755930887667,
        tensorboard_log="./ppo_tensorboard/")

    callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=log_dir)

    # Entrena el modelo con el callback personalizadoç
    num_datos = 1380  # 70%
    total_timesteps = num_datos * 1000  # Aproximadamente 1,380,400 pasos para 70% de 1972 días

    model.learn(total_timesteps=total_timesteps)
    model.learn(total_timesteps=100000, callback=callback)

    # Save the model
    model.save("./modelos/pporcc_model_binance_1d.pkl")
