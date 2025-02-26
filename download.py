from datetime import datetime
import requests
import pandas as pd
import os


def get_crypto_data(ticker, start_date, end_date, exchanges, token, resample_freq='1day'):
    """
    Descarga datos históricos de criptomonedas desde Tiingo y los guarda en archivos CSV por exchange.

    Parámetros:
        ticker (str): El símbolo de la criptomoneda, por ejemplo, 'btcusd'.
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        end_date (str): Fecha de fin en formato 'YYYY-MM-DD'.
        exchanges (list): Lista de nombres de exchanges.
        token (str): Token de autenticación de Tiingo.
        resample_freq (str): La frecuencia de muestreo de los datos. Default es '1day'.

    Retorna:
        None
    """
    url = f"https://api.tiingo.com/tiingo/crypto/prices"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Token {token}'
    }

    for exchange in exchanges:
        params = {
            'tickers': ticker,
            'startDate': start_date,
            'endDate': end_date,
            'resampleFreq': resample_freq,
            'exchanges': [exchange],
            'includeRawExchangeData': 'true'
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            if data:
                # Asegurarse de que exchangeData esté presente
                exchange_data = data[0].get('exchangeData', {}).get(exchange.upper(), [])
                if exchange_data:
                    df = pd.DataFrame(exchange_data)
                    output_dir = f"data/{resample_freq}/{exchange}"
                    os.makedirs(output_dir, exist_ok=True)
                    output_file = f"{ticker}_{exchange}_{resample_freq}_{start_date}_{end_date}.csv"
                    df.to_csv(f'{output_dir}/{output_file}', index=False)
                    print(f"Datos guardados en {output_dir}/{output_file}")
                else:
                    print(f"No se encontraron datos para el exchange especificado: {exchange}.")
            else:
                print("No se encontraron datos para los parámetros especificados.")
        else:
            print(f"Error: {response.status_code} para el exchange {exchange}.")


# Ejemplo de uso
if __name__ == "__main__":
    TICKER = "btcusd"
    START_DATE = "2012-01-01"
    END_DATE = datetime.now().strftime("%Y-%m-%d")
    TOKEN = "secret"
    RESAMPLE_FREQ = "1day"  # Puedes cambiar la frecuencia a '5min', '1hour', etc.

    EXCHANGES = [
        "ASCENDEX", "Mainnet", "Polygon", "Bancor", "BHEX", "Bibox", "Bilaxy",
        "Binance", "Bitfinex", "Bitflyer", "Bithumb", "Bitmart", "Bitstamp", "Bittrex", "Bybit",
        "Coinbase", "Cryptopia", "Curve", "DFYN", "Fraxswap",
        "FTX", "Gatecoin", "Gate.io", "Gemini", "HitBTC", "Huobi", "Indodax", "Kraken", "Kucoin",
        "LAToken", "Lbank", "Lydia", "MDEX", "MEXC", "OKex", "Orca", "P2PB2B", "Pancakeswap",
        "Pangolin", "Poloniex", "Quickswap", "Raydium", "Saberswap", "Serum DEX", "Spiritswap",
        "Spookyswap", "Sushiswap (Mainnet)", "Sushiswap", "Terraswap", "Trader Joe",
        "UniswapV2", "UniswapV3", "UniswapV3", "Upbit", "Wualtswap",
        "Yobit"
    ]

    get_crypto_data(TICKER, START_DATE, END_DATE, EXCHANGES, TOKEN, RESAMPLE_FREQ)
