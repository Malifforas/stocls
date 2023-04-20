import pandas as pd
import joblib

# Load the test data
test = pd.read_csv("test_processed.csv", index_col="Date", parse_dates=True)

# Load the trained model from file
model = joblib.load("model.pkl")

# Remove the "Adj Close" column from the test data
X_test = test.drop(columns=["Return", "Adj Close"])

# Use the model to predict on the test data
y_pred = model.predict(X_test)

# Add the predicted returns to the test data
test["Predicted_Return"] = y_pred

# Save the test data with the predicted returns to file
test.to_csv("test_results.csv")