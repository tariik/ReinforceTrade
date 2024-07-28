from joblib import dump
from sklearn.preprocessing import StandardScaler


class Normalizer(object):
    def __init__(self):
        self._scaler = StandardScaler()

    def fit(self, data):
        # Scale data with fitting data set
        scaler = self._scaler.fit(data)
        return scaler

    def normalized_data(self, data):
        self._scaler.fit(data)
        dump(self._scaler, 'scaler.joblib')
        return self.scale_data(data)

    def scale_data(self, data):
        return self._scaler.transform(data)

