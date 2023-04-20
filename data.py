import pandas as pd
import yfinance as yf
import joblib


class Data:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date

        self.train = None
        self.test = None
        self.columns = None

        self._load_data()
        self._save_data()

    def _load_data(self):
        # Download historical stock data
        try:
            df = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        except Exception as e:
            print("Error downloading data:", e)
            return

        # Fill in missing data using forward fill
        df.fillna(method="ffill", inplace=True)

        # Calculate daily percentage returns
        df["Return"] = df["Adj Close"].pct_change()

        # Create lagged features
        df["Lag1"] = df["Return"].shift(1)
        df["Lag2"] = df["Return"].shift(2)
        df["Lag3"] = df["Return"].shift(3)

        # Drop any rows with missing data
        df.dropna(inplace=True)

        # Split the data into training and testing sets
        self.train = df[df.index < "2021-01-01"]
        self.test = df[df.index >= "2021-01-01"]

        # Save the column names to a file
        self.columns = df.columns.tolist()

    def _save_data(self):
        self.train.to_csv("train.csv")
        self.test.to_csv("test.csv")

        with open("columns.pkl", "wb") as f:
            joblib.dump(self.columns, f)