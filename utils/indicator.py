import talib


class Indicators(object):
    def __init__(self):
        pass

    @staticmethod
    def technical(data):
        # Moving Averages
        data['fast_ema'] = talib.EMA(data['close'], timeperiod=10)
        data['slow_ema'] = talib.EMA(data['close'], timeperiod=30)
        data['rsi'] = talib.RSI(data['close'], timeperiod=14)
        data['atr'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
        data['bb_upper'], data['bb_middle'], data['bb_lower'] = talib.BBANDS(data['close'], timeperiod=20,
                                                                             nbdevup=2, nbdevdn=2)
        data['cci'] = talib.CCI(data['high'], data['low'], data['close'], timeperiod=14)
        data['cmo'] = talib.CMO(data['close'], timeperiod=14)
        data['dm_pos'] = talib.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=14)
        data['dm_neg'] = talib.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=14)
        data['dc_upper'] = talib.MAX(data['high'], timeperiod=14)
        data['dc_middle'] = talib.MA(data['close'], timeperiod=14, matype=0)
        data['dc_lower'] = talib.MIN(data['low'], timeperiod=14)


        # Linear Regression
        data['linear_regression_slope'] = talib.LINEARREG_SLOPE(data['close'], timeperiod=14)

        data['linear_regression_value'] = talib.LINEARREG(data['close'], timeperiod=14)

        # MACD
        data['macd'], _, _ = talib.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)

        # On-Balance Volume
        data['obv'] = talib.OBV(data['close'], data['volume'])

        # Rate of Change
        data['rate_of_change'] = talib.ROC(data['close'], timeperiod=10)

        # Stochastics
        data['stochastics_k'], data['stochastics_d'] = talib.STOCH(data['high'], data['low'], data['close'],
                                                                   fastk_period=5, slowk_period=3, slowk_matype=0,
                                                                   slowd_period=3, slowd_matype=0)
        # Volume Weighted Average Price
        data['vwap'] = talib.WMA(data['close'], timeperiod=14)

        # Efficiency Ratio
        # Note: Efficiency Ratio is not directly available in TA-Lib, assuming custom calculation or direct set
        # data['efficiency_ratio'] = None

        # Klinger Volume Oscillator
        # Note: Klinger is not directly available in TA-Lib, assuming custom calculation or direct set
        # cdata['klinger'] = None

        # data['linear_regression_intercept'] = None  # Assuming custom calculation or direct set
        # Pressure (Assuming this is a custom indicator or calculated elsewhere)
        # data['pressure'] = None

        # Psychological Line (Assuming this is a custom indicator or calculated elsewhere)
        #  data['psych_line'] = None
        # Relative Volatility Index (Assuming this is a custom indicator or calculated elsewhere)
        # data['rel_vol_index'] = None
        # Swings Direction (Assuming this is a custom indicator or calculated elsewhere)
        #           data['swings_direction'] = None

        # Vertical Horizontal Filter (Assuming this is a custom indicator or calculated elsewhere)
        # data['vh_filter'] = None

        # Volatility Ratio (Assuming this is a custom indicator or calculated elsewhere)
        # data['volatility_ratio'] = None

        return data
