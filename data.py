import yfinance as yf
import pandas as pd
import numpy as np
import os

class Data:
    def __init__(self, ticker, start_date, end_date, data_path):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data_path = data_path
        self.file_path = os.path.join(data_path, f'{ticker}_{start_date}_{end_date}.csv')

    def download_data(self):
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        df.to_csv(self.file_path, index=False)

    def load_data(self):
        return pd.read_csv(self.file_path)

def download_stock_data(ticker, start_date, end_date):
    # Download stock data from Yahoo Finance API
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # Filter out any rows with missing data
    stock_data.dropna(inplace=True)

    # Add a column for the daily stock return
    stock_data["Return"] = stock_data["Adj Close"].pct_change()

    return stock_data

def preprocess_stock_data(stock_data):
    # Add columns for technical indicators
    stock_data["SMA_5"] = stock_data["Adj Close"].rolling(window=5).mean()
    stock_data["SMA_10"] = stock_data["Adj Close"].rolling(window=10).mean()
    stock_data["SMA_20"] = stock_data["Adj Close"].rolling(window=20).mean()
    stock_data["SMA_50"] = stock_data["Adj Close"].rolling(window=50).mean()
    stock_data["SMA_200"] = stock_data["Adj Close"].rolling(window=200).mean()
    stock_data["EMA_5"] = stock_data["Adj Close"].ewm(span=5, adjust=False).mean()
    stock_data["EMA_10"] = stock_data["Adj Close"].ewm(span=10, adjust=False).mean()
    stock_data["EMA_20"] = stock_data["Adj Close"].ewm(span=20, adjust=False).mean()
    stock_data["EMA_50"] = stock_data["Adj Close"].ewm(span=50, adjust=False).mean()
    stock_data["EMA_200"] = stock_data["Adj Close"].ewm(span=200, adjust=False).mean()

    # Calculate the stock's volatility over the past 5 days
    stock_data["Volatility"] = stock_data["Return"].rolling(window=5).std()

    # Drop any rows with missing data
    stock_data.dropna(inplace=True)

    return stock_data

def load_data(tickers, start_date, end_date):
    # Create an empty DataFrame to hold the preprocessed stock data
    all_data = pd.DataFrame()

    # Loop through each ticker and download/preprocess the data
    for ticker in tickers:
        # Download the stock data
        stock_data = download_stock_data(ticker, start_date, end_date)

        # Preprocess the stock data
        stock_data = preprocess_stock_data(stock_data)

        # Add a column to indicate which stock this data is for
        stock_data["Ticker"] = ticker

        # Append the preprocessed data to the master DataFrame
        all_data = all_data.append(stock_data)

    return all_data

def save_data(data, filename):
    # Save the preprocessed data to a CSV file
    data.to_csv(filename)

def load_preprocessed_data(filename):
    # Load the preprocessed data from a CSV file
    data = pd.read_csv(filename, index_col="Date", parse_dates=True)

    return data