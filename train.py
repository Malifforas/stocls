import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define the features and target variable
features = ["Lag1", "Lag2", "Lag3"]
target = "Return"

# Load the training data from file
train = pd.read_csv("train.csv", index_col=0)

# Train the random forest classifier
try:
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(train[features], train[target])
except Exception as e:
    print("Error training model:", e)

# Save the trained model to a file
joblib.dump(clf, "model.joblib")