import pandas as pd

from train.env.cripto.crypto_env import TradingCryptoEnv


# Definir el entorno de entrenamiento y validación
def env_creator_train(configs):
    return TradingCryptoEnv(configs)  # return an env instance


if __name__ == "__main__":
    df = pd.read_csv('BINANCE_BTCUSDT_15m_data.csv')



    # register_env("validation_env", lambda config: TestTradingCryptoEnv(config))

    # Configuración del agente PPO


    # Set evaluation config
    # Configuración de evaluación


    # Usar la clase de callbacks personalizada
    # config = config.callbacks(MyCallbacks)

    # Tunear hiperparámetros (opcional)


    # results = tuner.fit()

    # Seleccionar el mejor resultado
    # best_result = results.get_best_result()
    # best_checkpoint = None
    # best_config = None

    # if best_result:
    #    best_config = PPOConfig.from_dict(best_result.config)
    #    best_checkpoint = best_result.best_checkpoints[-1][0]

    # Entrenar el agente con la mejor configuración
    # trainer = ppo.PPO(config=best_config, env="train_env")
    # if best_checkpoint:
    #    trainer.restore(best_checkpoint)

    # for _ in range(100):
    # trainer.train()

    # Guardar el agente entrenado
    # trainer.save("agents/checkpoint")

    # Evaluar el agente entrenado


