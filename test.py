# Download real-time stock data
df_live = yf.download("AAPL", start="2023-04-19", end="2023-04-19")

# Calculate the lagged features
live_features = [df_live['Return'].shift(1)[0], df_live['Return'].shift(2)[0], df_live['Return'].shift(3)[0]]

# Make a prediction on the live data
live_pred = best_clf.predict([live_features])[0]

# Print the predicted return for the next day
print('Predicted return:', live_pred)