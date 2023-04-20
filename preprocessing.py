import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    # Fill in missing data using forward fill
    data.fillna(method="ffill", inplace=True)

    # Calculate daily percentage returns
    data["Return"] = data["Adj Close"].pct_change()

    # Create lagged features
    data["Lag1"] = data["Return"].shift(1)
    data["Lag2"] = data["Return"].shift(2)
    data["Lag3"] = data["Return"].shift(3)

    # Drop any rows with missing data
    data.dropna(inplace=True)

    # Scale the data
    scaler = StandardScaler()
    data[["Open", "High", "Low", "Close", "Adj Close", "Volume", "Lag1", "Lag2", "Lag3"]] = scaler.fit_transform(data[["Open", "High", "Low", "Close", "Adj Close", "Volume", "Lag1", "Lag2", "Lag3"]])

    return data

# Load the training data from file
train = pd.read_csv("train.csv", index_col="Date", parse_dates=True)

# Preprocess the training data
train = preprocess_data(train)

# Save the preprocessed data to file
train.to_csv("train_processed.csv")

# Load the test data from file
test = pd.read_csv("test.csv", index_col="Date", parse_dates=True)

# Preprocess the test data using the same steps as for the training data
test = preprocess_data(test)

# Save the preprocessed data to file
test.to_csv("test_processed.csv")