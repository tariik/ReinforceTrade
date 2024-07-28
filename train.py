import os

from sb3_contrib import RecurrentPPO, QRDQN, MaskablePPO
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from config import LOGGER, env_config
from train.env.cripto.crypto_env import TradingCryptoEnv

if __name__ == "__main__":
    cwd = os.path.dirname(os.path.realpath(__file__))
    output_directory = os.path.join(cwd, 'modelos_pkl')
    if not os.path.exists(output_directory):
        LOGGER.info('{} does not exist. Creating Directory.'.format(output_directory))
        os.mkdir(output_directory)

    log_dir = "./check_freq"

    tensorboard_log = "./tensorboard_log"
    os.makedirs(log_dir, exist_ok=True)

    env = TradingCryptoEnv(env_config)
    env = Monitor(env, log_dir)

    # Configure the logger for TensorBoard
    # todo: Configuración del agente hiperparámetros
    model = MaskablePPO(
        'MlpPolicy',
        env,
        verbose=1,
        tensorboard_log=tensorboard_log)

    model.learn(total_timesteps=2000000)

    # Save the model
    model.save(os.path.join(output_directory, "MaskablePPO_ALL_INDIC_FILL_PNL_BTC.pkl"))

    #  todo: Configuración de evaluación en tensorboard
    #  todo: Usar la clase de callbacks personalizada
    #  todo: Tunear hiperparámetros (opcional)
    #  todo: Seleccionar el mejor resultado best_checkpoint
    #  todo: Entrenar de neuvo  el agente con la mejor configuración
    #  todo: Guardar el agente entrenado
    #  todo: trainer.save("agents/checkpoint")
    #  todo: Evaluar el agente entrenado
