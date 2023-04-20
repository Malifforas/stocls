import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Define the features and target variable
features = ["Lag1", "Lag2", "Lag3"]
target = "Return"

# Load the test data from file
