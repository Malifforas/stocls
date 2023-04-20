import preprocessing
import data
import train
import evaluate
import test

# Set the stock ticker symbol and date range
symbol = "AAPL"
start_date = "2010-01-01"
end_date = "2023-04-20"

# Preprocess the data
preprocessing.preprocess_data(symbol, start_date, end_date)

# Load the preprocessed data
train_data, test_data = data.load_data()

# Train the model
model = train.train_model(train_data)

# Evaluate the model
evaluate.evaluate_model(model, test_data)

# Make predictions on new data
new_data = test.load_new_data()
predictions = test.make_predictions(model, new_data)
print(predictions)