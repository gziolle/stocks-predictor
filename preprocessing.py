import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
from scipy.signal import spectrogram
from indexes import *

class Preprocessing:
    
    def __init__(self):
        self.sc = None

    def get_frequency_features (self, X, window=256, overlap=255):
        _, _, Sxx = spectrogram(X[:,0], fs=1, nfft=window, noverlap=overlap)
        return np.transpose(Sxx)

    def load_scaler(self):
        self.sc = joblib.load("minmaxscaler.pkl")

    def save_scaler(self):
        joblib.dump(self.sc, "minmaxscaler.pkl")

    def feature_scale(self, data):
        #if self.sc == None:
        sc = MinMaxScaler(feature_range = (-1, 1))
        scaled = sc.fit_transform(data)
        # else:
        #     scaled = self.sc.transform(data)
        return scaled
    
    def create_train_result(self, X, N=1):
        return np.array([X[1:] > X[:-1], X[1:] <= X[:-1]], dtype=np.int).reshape((len(X)-1, 2))

    def create_time_series(self, data, y, N = 50):
        X = []
        for i in range(N, len(data)):
            X.append(data[i-N:i, :])
        X = np.array(X)
        return X, y[N:]

    def split_dataset(self, X, y, N=50, train_div=0.8):
        total_size = len(y)
        div = int((total_size - N) * train_div)
        X_train = X[:div]
        y_train = y[:div]
        X_test = X[div - N:]
        y_test = y[div - N:]
        return X_train, y_train, X_test, y_test
    
    def normalize(self, data, window=10, step=1, forecast=1):
        X, Y = [], []
        #rsi = relative_strength_index(data[:, 4], window)
        #mm = moving_average(data[:, 4], method='exp', N=(9, 21))

        o = np.array([self.feature_scale(data[i:i + window, 1].reshape(-1,1)) for i in range(0, len(data) - window)])
        h = np.array([self.feature_scale(data[i:i + window, 2].reshape(-1,1)) for i in range(0, len(data) - window)])
        l = np.array([self.feature_scale(data[i:i + window, 3].reshape(-1,1)) for i in range(0, len(data) - window)])
        c = np.array([self.feature_scale(data[i:i + window, 4].reshape(-1,1)) for i in range(0, len(data) - window)])
        v = np.array([self.feature_scale(data[i:i + window, 5].reshape(-1,1)) for i in range(0, len(data) - window)])
        Y = self.create_train_result(data[:, 4])
        X = np.column_stack((o.reshape(-1,1), h.reshape(-1,1), l.reshape(-1,1), c.reshape(-1,1), v.reshape(-1,1)))#, rsi.reshape(-1,1)[window:]))
        return np.array(X), np.array(Y)[window:]