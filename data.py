import pandas as pd
import yfinance as yf

# Define the stock ticker symbol and date range
symbol = "AAPL"
start_date = "2010-01-01"
end_date = "2023-04-19"

# Download historical stock data
try:
    df = yf.download(symbol, start=start_date, end=end_date)
except Exception as e:
    print("Error downloading data:", e)

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
train = df[df.index < "2021-01-01"]
test = df[df.index >= "2021-01-01"]

# Save the data to a file
train.to_csv("train.csv")
test.to_csv("test.csv")