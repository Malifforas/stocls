import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib
from sklearn.metrics import mean_squared_error

# Load the preprocessed training and test data
train = pd.read_csv("train_processed.csv", index_col="Date", parse_dates=True)
test = pd.read_csv("test_processed.csv", index_col="Date", parse_dates=True)

# Split the data into X and y
X_train = train.drop(columns=["Adj Close", "Return"])
y_train = train["Adj Close"]
X_test = test.drop(columns=["Adj Close", "Return"])
y_test = test["Adj Close"]

# Create the neural network model
model = MLPRegressor(hidden_layer_sizes=(10, 10), activation="relu", solver="adam", max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Save the trained model to file
joblib.dump(model, "model.pkl")

# Use the model to predict on the test data
y_pred = model.predict(X_test)

# Calculate the root mean squared error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)