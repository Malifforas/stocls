import pandas as pd
import preprocessing

# Load the data
data = pd.read_csv('train.csv')

# Preprocess the data
processed_data = preprocessing.preprocess_data(data)

# Print the first few rows of the processed data
print(processed_data.head())