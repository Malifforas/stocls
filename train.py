import os
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from data import Data
from preprocessing import Preprocessing
from model import Model
from trader import Trader


def train():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Define arguments
    parser = argparse.ArgumentParser(description='Train the stock prediction model')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Ticker symbol of stock to train on')
    parser.add_argument('--start_date', type=str, default='2010-01-01', help='Start date for training data')
    parser.add_argument('--end_date', type=str, default='2020-12-31', help='End date for training data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--model_type', type=str, default='LSTM', help='Type of model to train')
    parser.add_argument('--data_path', type=str, default='data', help='Path to store data')
    parser.add_argument('--model_path', type=str, default='models', help='Path to store model')
    parser.add_argument('--preprocessing_path', type=str, default='preprocessing', help='Path to store preprocessing info')
    parser.add_argument('--trader_path', type=str, default='trader', help='Path to store trader info')
    args = parser.parse_args()

    # Create directories if they don't exist
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.preprocessing_path):
        os.makedirs(args.preprocessing_path)
    if not os.path.exists(args.trader_path):
        os.makedirs(args.trader_path)

    # Instantiate Data class
    data = Data(args.ticker, args.start_date, args.end_date, args.data_path)

    # Download stock data if not already downloaded
    if not os.path.isfile(data.file_path):
        data.download_data()

    # Load data
    df = pd.read_csv(data.file_path)

    # Instantiate Preprocessing class
    prep = Preprocessing()

    # Preprocess data
    X_train, y_train, X_test, y_test, scaler = prep.preprocess_data(df)

    # Save preprocessing info
    prep.save_preprocessing_info(args.preprocessing_path, scaler)

    # Instantiate Model class
    model = Model(args.model_type)

    # Build and train model
    history = model.build_and_train_model(X_train, y_train, X_test, y_test, args.batch_size, args.epochs)

    # Save model
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_file_path = os.path.join(args.model_path, f'{args.ticker}_{args.model_type}_{now}.h5')
    model.save_model(model_file_path)

    # Instantiate Trader class
    trader = Trader(args.ticker, scaler, model_file_path, args.trader_path)

    # Run trader and print results
    trader.run_trader()


if __name__ == '__main__':
    train()