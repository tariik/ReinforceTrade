import pandas as pd
import json

# Cargar el archivo CSV
df = pd.read_csv('data/1day/Bittrex/btcusd_Bittrex_1day_2012-01-01_2024-07-06.csv')

# Determinar el índice de corte para el 70% de entrenamiento y 30% de prueba
train_size = int(len(df) * 0.7)

# Dividir los datos en conjuntos de entrenamiento y prueba
train_df = df[:train_size]
test_df = df[train_size:]
# Definir variables de configuración
# Cargar las configuraciones desde el archivo JSON
with open('config/env.json', 'r') as config_file:
    config = json.load(config_file)

# Define la configuración del entorno para el conjunto de entrenamiento
env_config_train = {
    "initial_balance": config["initial_balance"],
    "buy_fee": config["buy_fee"],
    "sell_fee": config["sell_fee"],
    "borrow_interest_rate": config["borrow_interest_rate"],
    "time_window": config["time_window"],
    "look_back_window": config["look_back_window"],
    "alpha": config["alpha"],
    "df": train_df
}

# Define la configuración del entorno para el conjunto de prueba
env_config_test = {
    "initial_balance": config["initial_balance"],
    "buy_fee": config["buy_fee"],
    "sell_fee": config["sell_fee"],
    "borrow_interest_rate": config["borrow_interest_rate"],
    "time_window": config["time_window"],
    "look_back_window": config["look_back_window"],
    "alpha": config["alpha"],
    "df": test_df
}