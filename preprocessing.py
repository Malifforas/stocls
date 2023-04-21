import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class Preprocessing:
    def __init__(self):
        self.scaler = StandardScaler()

        def preprocess_data(self, df):
            # Get features and target
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

            # Scale features
            X = self.scaler.fit_transform(X)

            # Reshape target to 2D array
            y = y.reshape(-1, 1)

            # Scale target
            y = self.scaler.fit_transform(y)

            # Create sequences of features and target
            lookback = 30
            X_seq, y_seq = [], []
            for i in range(lookback, len(X)):
                X_seq.append(X[i - lookback:i, :])
                y_seq.append(y[i, 0])

            # Convert sequences to numpy arrays
            X_seq, y_seq = np.array(X_seq), np.array(y_seq)

            # Split data into train and test sets
            split_idx = int(0.8 * len(X_seq))
            X_train, y_train = X_seq[:split_idx], y_seq[:split_idx]
            X_test, y_test = X_seq[split_idx:], y_seq[split_idx:]

            return X_train, y_train, X_test, y_test, self.scaler

        def save_preprocessing_info(self, file_path, scaler):
            with open(file_path, 'wb') as f:
                pickle.dump(scaler, f)

        def load_preprocessing_info(self, file_path):
            with open(file_path, 'rb') as f:
                scaler = pickle.load(f)
            return scaler
def get_technical_indicators(data):
    # Create 7 and 21 days Moving Average
    data['ma7'] = data['Close'].rolling(window=7).mean()
    data['ma21'] = data['Close'].rolling(window=21).mean()

    # Create MACD
    data['26ema'] = pd.Series.ewm(data['Close'], span=26).mean()
    data['12ema'] = pd.Series.ewm(data['Close'], span=12).mean()
    data['MACD'] = (data['12ema'] - data['26ema'])

    # Create Bollinger Bands
    data['20sd'] = data['Close'].rolling(window=20).std()
    data['upper_band'] = data['ma21'] + (data['20sd']*2)
    data['lower_band'] = data['ma21'] - (data['20sd']*2)

    # Create Exponential moving average
    data['ema'] = data['Close'].ewm(com=0.5).mean()

    # Create Momentum
    data['momentum'] = data['Close'] - 1

    return data


def get_feature_importance_data(ticker):
    data = pd.read_csv(f"data/{ticker}.csv")
    data = get_technical_indicators(data)
    data = data.dropna()
    X = data.iloc[:, 1:-1]
    y = data.iloc[:, -1]
    return X, y


def get_train_test_data(ticker):
    X, y = get_feature_importance_data(ticker)
    split = int(0.7*len(X))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    return X_train, X_test, y_train, y_test