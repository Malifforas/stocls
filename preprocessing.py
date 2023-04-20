import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data from the CSV file
df = pd.read_csv("train.csv")

# Split the data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Define the feature columns and target column
feature_cols = train_df.columns[:-1]
target_col = train_df.columns[-1]

# Extract the features and target for the training set
train_features = train_df[feature_cols].values
train_target = train_df[target_col].values

# Extract the features and target for the validation set
val_features = val_df[feature_cols].values
val_target = val_df[target_col].values