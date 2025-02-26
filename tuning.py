import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from callback import SaveOnBestTrainingRewardCallback
from config import env_config_train
from environment.crypto_trade.cripo_env import TradingCryptoEnv
import numpy as np
from tensorboardX import SummaryWriter
import os


def objective(trial):
    # Sugerir valores para los hiperparámetros
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    gamma = trial.suggest_uniform('gamma', 0.4, 0.9)
    n_steps = trial.suggest_int('n_steps', 2048, 4096)
    ent_coef = trial.suggest_loguniform('ent_coef', 0.00001, 0.1)
    look_back_window = trial.suggest_int('look_back_window', 1, 20)
    time_window = trial.suggest_int('time_window', 1, 10)

    # Actualizar la configuración del entorno
    env_config_train['look_back_window'] = look_back_window
    env_config_train['time_window'] = time_window

    # Definir el entorno
    env = TradingCryptoEnv(env_config_train)
    log_dir = f"./check_freq_optuna/trial_{trial.number}"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)

    # Crear el modelo PPO con los hiperparámetros sugeridos
    model = PPO('MlpPolicy', env, learning_rate=learning_rate, gamma=gamma, n_steps=n_steps, ent_coef=ent_coef,
                verbose=0)

    # Configurar TensorBoard
    writer = SummaryWriter(log_dir)

    # Entrenar el modelo
    model.learn(total_timesteps=10000, callback=SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir))

    # Evaluar el modelo
    try:
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(y) > 0:
            mean_reward = np.mean(y[-100:])
        else:
            mean_reward = -np.inf
    except Exception as e:
        print(f"Error reading log results: {e}")
        mean_reward = -np.inf

    # Registrar en TensorBoard
    writer.add_scalar('mean_reward', mean_reward, trial.number)
    writer.add_hparams({
        'learning_rate': learning_rate,
        'gamma': gamma,
        'n_steps': n_steps,
        'ent_coef': ent_coef,
        'look_back_window': look_back_window,
        'time_window': time_window
    }, {'mean_reward': mean_reward})
    writer.close()

    return mean_reward


if __name__ == "__main__":
    # Crear un estudio y optimizar la función de objetivo
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, n_jobs=1)

    # Imprimir los mejores hiperparámetros
    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
