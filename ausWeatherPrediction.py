"""Import"""
# Base import
import torch
import numpy as np
import pandas as pd

# Sklearn Preprocessing
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# DNN Model built-in functions
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, TensorDataset, DataLoader

# Visualization
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

'''--Configure--'''
csv_path = "/Users/ryanmin/Documents/dev/python/practice/pytorch_practice/weatherAUS.csv"
columns = ['Location','MinTemp','MaxTemp','Rainfall','WindSpeed9am','WindSpeed3pm','Temp9am','Temp3pm','Humidity9am','Humidity3pm','Pressure3pm','Pressure9am','WindGustDir','WindGustSpeed','RainToday','RainTomorrow']
'''--Configure--'''

# Import CSV file
raw_df = pd.read_csv(csv_path)

"""Preprocessing"""
target_cols = ['RainTomorrow']

# Categorical columns (Bool: No->0, Yes->1) (Encode: Categories to Columns)
bool_cols = ['RainToday','RainTomorrow']
categorical_cols = ['Location', 'WindGustDir']

# Fill NaN columns (Numeric)
fillna_mean_cols = ['MinTemp','MaxTemp','Humidity3pm','Temp9am','Temp3pm','Pressure9am','Pressure3pm']
fillna_median_cols = ['WindGustSpeed','Rainfall','WindSpeed9am','WindSpeed3pm','Humidity9am',]
fillna_mode_cols = []

# Drop NaN rows
dropna_cols = ['RainToday','RainTomorrow','WindGustDir']

# Functions (Preprocessing)
def preprocess_data(df):
    # Drop rows with null values
    df.dropna(subset=dropna_cols, inplace=True)

    # Fill NaN columns (Mean)
    for column in fillna_mean_cols:
        df[column].fillna(df[column].mean(), inplace=True)

    # Fill NaN columns (Median)
    for column in fillna_median_cols:
        df[column].fillna(df[column].median(), inplace=True)

    # Fill NaN columns (Mode)
    for column in fillna_mode_cols:
        df[column].fillna(df[column].mode()[0], inplace=True)

    # Encode boolean columns (Yes->1, No->0)
    df.replace({'Yes': 1, 'No': 0}, inplace=True)

    # Scale Numerical Columns (0 to 1)
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # One-hot encode categorical columns (Categorical values to each columns)
    df = pd.get_dummies(df, columns=categorical_cols, sparse=True)

    return df

# Get input columns
input_cols = [column for column in columns if column not in target_cols]

# Get numerical & categorical columns [No need to change]
raw_df = raw_df[[*input_cols, *target_cols]]
numerical_cols = raw_df.select_dtypes(include=np.number).columns.tolist()

# Preprocess the data
raw_df = preprocess_data(raw_df)

# set input & target columns
input_cols = raw_df.columns.difference([*target_cols])

# Convert to dataframe
input_df = raw_df[input_cols].values
target_df = raw_df[target_cols].values

# Define and split dataset
dataset = TensorDataset(torch.tensor(input_df, dtype=torch.float32), torch.tensor(target_df, dtype=torch.float32))

"""Load data to DataLoader"""
'''--Configure--'''
# set dataset ratio
train_pct = 0.6
val_pct = 0.2
test_pct = 0.2

# DataLoader Batch Size
batch_size = 100
'''--Configure--'''

# Set dataset size
num_records = len(dataset)
num_train = int(num_records * train_pct)
num_val = int(num_records * val_pct)
num_test = num_records - num_train - num_val

# Split dataset
train_ds, val_ds, test_ds = random_split(dataset, [num_train, num_val, num_test])

# Load dataset to the DataLoader
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size)
test_dl = DataLoader(test_ds, batch_size)

"""DNN Model"""
'''--Configure--'''
# Set learning rate
learning_rate = 0.0001

# Set input & output size
output_size = 1

# Set epoch
epochs = 100
'''--Configure--'''

# Neural Network Model
class DNN_Model(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, hidden3, hidden4, hidden5, output_size):
        super(DNN_Model, self).__init__()
        # Input Layer
        self.linear1 = nn.Linear(input_size, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, hidden3)
        self.linear4 = nn.Linear(hidden3, hidden4)
        self.linear5 = nn.Linear(hidden4, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        x = self.linear5(x)
        return x

# Set deep-learning neural size
input_size = len(input_cols) # 73
hidden1_node = int(input_size * (2/3))
hidden2_node = int(hidden1_node * (2/3))
hidden3_node = int(hidden2_node * (2/3))
hidden4_node = int(hidden3_node * (2/3))
hidden5_node = int(hidden4_node * (2/3))

# Create the model instance
model = DNN_Model(input_size, hidden1_node, hidden2_node, hidden3_node, hidden4_node, hidden5_node, output_size)

# Set Loss & Optimizer functions
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training iteration
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_dl:
        optimizer.zero_grad() # Reset gradients
        outputs = model(inputs) # Make prediction
        loss = criterion(outputs.squeeze(), labels.squeeze()) # Compute Loss Function
        loss.backward() # Compute gradient
        optimizer.step() # Optimize

    # Evaluation on the validation set
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_acc = 0.0
        for inputs, labels in val_dl:
            outputs = model(inputs) # Make prediction
            val_loss += criterion(outputs.squeeze(), labels.squeeze()).item() # Compute Loss using Binary Cross Entropy
            val_acc += ((torch.sigmoid(outputs) > 0.5).float() == labels).float().mean().item() # Get Accuracy

        val_loss /= len(val_dl)
        val_acc /= len(val_dl)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {val_loss:.4f}, Accuracy: {val_acc * 100:.2f}%")

"""Visualization"""
# After the training loop
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_dl:
        outputs = model(inputs)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.extend(preds.squeeze())
        all_labels.extend(labels.squeeze())

# Calculate confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds, normalize='pred')  # Use 'true' for percentages

# Display the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt=".2%", cmap='Blues', cbar=False, xticklabels=['Rain', 'Not Rain'], yticklabels=['Rain', 'Not Rain'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Percentage)')
plt.show()