import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.models
from tensorflow.keras.layers import LSTM, Dense
from sklearn.neural_network import MLPRegressor
import joblib
from sklearn.metrics import mean_squared_error

class Model:
    def __init__(self, model_type):
        self.model_type = model_type
        self.model = None

    def build_and_train_model(self, X_train, y_train, X_test, y_test, batch_size, epochs):
        # Build model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))

        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train model
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs)

        self.model = model

        return history, model

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        return tf.keras.models.load_model(model_path)

# Load the preprocessed data
data = pd.read_csv("data.csv", index_col="Date", parse_dates=True)

# Split the data into X and y
X = data.drop(columns=["Adj Close", "Return"])
y = data["Adj Close"]

# Create the neural network model
model = MLPRegressor(hidden_layer_sizes=(10, 10), activation="relu", solver="adam", max_iter=1000)

# Train the model
model.fit(X, y)

# Save the trained model to file
joblib.dump(model, "model.pkl")

# Function to predict stock prices and decide whether to buy, hold or sell
def predict_and_trade(money, data):
    # Load the trained model from file
    model = joblib.load("model.pkl")

    # Use the model to predict on the data
    y_pred = model.predict(data)

    # Calculate the root mean squared error
    rmse = np.sqrt(mean_squared_error(data["Adj Close"], y_pred))
    print("RMSE:", rmse)

    # Calculate the last close price for each stock in the data
    last_close_prices = data.groupby("Symbol").last()["Adj Close"]

    # Calculate the number of shares we can buy for each stock with the given amount of money
    share_prices = money / last_close_prices

    # Calculate the predicted prices for each stock in the data
    predicted_prices = pd.Series(y_pred, index=data.index)

    # Create a new DataFrame to hold the buy/sell decisions for each stock
    decisions = pd.DataFrame(index=data.index, columns=data["Symbol"])

    # Loop through each date in the data
    for date in data.index:
        # Loop through each stock symbol
        for symbol in data["Symbol"].unique():
            # Calculate the predicted price for the stock on the current date
            predicted_price = predicted_prices.loc[date, symbol]

            # Calculate the last close price for the stock on the current date
            last_close_price = last_close_prices.loc[symbol]

            # Check if the predicted price is higher than the last close price
            if predicted_price > last_close_price:
                # If the predicted price is higher, buy the stock if we have enough money
                if share_prices.loc[symbol] >= 1:
                    decisions.loc[date, symbol] = "Buy"
                else:
                    decisions.loc[date, symbol] = "Hold"
            else:
                # If the predicted price is lower, sell the stock if we have any shares
                if decisions.iloc[-1][symbol] == "Buy":
                    decisions.loc[date, symbol] = "Sell"
                else:
                    decisions.loc[date, symbol] = "Hold"
