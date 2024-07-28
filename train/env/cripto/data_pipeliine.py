import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

from config import LOGGER
from utils.indicator import Indicators
from utils.normalizer import Normalizer


class DataPipeline(object):
    def __init__(self, is_test, train_data, test_data, symbol, timeframe, exchanges):
        self._scaler = StandardScaler()
        self.is_test = is_test
        self.train_data = train_data
        self.test_data = test_data
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchanges = exchanges
        self.indicators = Indicators()
        self.normalizer = Normalizer()

    def load_ohcl_data(self):
        directories = [os.path.join(self.train_data, self.symbol, exchange, self.timeframe) for exchange in
                       self.exchanges]
        if self.is_test:
            directories = [os.path.join(self.test_data, self.symbol, exchange, self.timeframe) for exchange in
                           self.exchanges]

        all_csv_files = []
        for directory in directories:
            if os.path.exists(directory):
                csv_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
                all_csv_files.extend(csv_files)

        if not all_csv_files:
            LOGGER.error('No CSV files found in the specified directories.')
            exit()

        LOGGER.info('Pick a random CSV file from the collected list...')
        # Pick a random CSV file from the collected list
        random_csv_file_path = random.choice(all_csv_files)
        # Load the content of the selected CSV file
        data = pd.read_csv(random_csv_file_path)
        LOGGER.info('Preserve the raw data and drop unnecessary columns')

        data = data.drop([col for col in data.columns.tolist()
                          if col in ['date', 'volumeNotional', 'tradesDone']], axis=1)

        prices = data['close']
        data = self.indicators.technical(data=data)
        data = data.interpolate(method='linear').ffill().bfill()

        normalized_data = self.normalizer.normalized_data(data.copy(deep=True))
        # Remove outliers
        normalized_data = np.clip(normalized_data, -10., 10.)
        return prices, data, normalized_data
