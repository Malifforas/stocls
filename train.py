import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Define the features and target variable
features = ["Lag1", "Lag2", "Lag3"]
target = "Return"

# Load the training data from file
train = pd.read_csv("train.csv", index_col=0)

# Split the training data into features and target
X_train = train[features]
y_train = train[target]

# Define the model and hyperparameters
model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

# Fit the model to the training data
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, "model.joblib")