import argparse
import numpy as np
from sklearn.metrics import mean_squared_error
from data import Data
from model import train_model, predict


def evaluate(ticker):
    # Load the data for the given ticker
    data = Data(ticker, '2000-01-01', '2023-04-20', 'data')
    df = data.load_data()

    # Train the model on the data
    model = train_model(df)

    # Make predictions using the trained model
    y_true = df['Close'].values
    y_pred = predict(model, df)

    # Compute the RMSE for the predictions
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"RMSE for {ticker}: {rmse}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, required=True, help="Stock ticker symbol")
    args = parser.parse_args()

    evaluate(args.ticker)