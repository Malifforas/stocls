import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from data import Data


class Evaluator:
    def __init__(self, features, target):
        self.features = features
        self.target = target

        self.test = None
        self.model = None

        self._load_data()
        self._load_model()

    def _load_data(self):
        data = Data(symbol="AAPL", start_date="2010-01-01", end_date="2023-04-19")
        self.test = pd.read_csv("test.csv")

    def _load_model(self):
        self.model = joblib.load("model.pkl")

    def evaluate(self):
        X_test = self.test[self.features]
        y_test = self.test[self.target]
        y_pred = self.model.predict(X_test)

        # Evaluate the performance of the model
        accuracy = accuracy_score(y_test > 0, y_pred > 0.5)
        print("Accuracy:", accuracy)

        # Tune hyperparameters using grid search
        params = {"C": [0.01, 0.1, 1, 10, 100]}
        clf = GridSearchCV(self.model, params)
        clf.fit(X_test, y_test > 0)
        print("Best hyperparameters:", clf.best_params_)