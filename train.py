import os
import pandas as pd

folder_path = 'data'

dataframes = []

for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        df['DATE'] = pd.to_datetime(df['DATE'])
        df.set_index('DATE', inplace=True)
        df.columns = [f"{file_name.replace('.csv', '')}_{col}" for col in df.columns]
        dataframes.append(df)

merged_df = pd.concat(dataframes, axis=1)

merged_df.fillna(method='ffill', inplace=True)

# Remove data older than 1960
merged_df = merged_df[merged_df.index >= '1960-01-01']

merged_df.to_csv('merged_data.csv')

merged_df = pd.read_csv('merged_data.csv')
merged_df['DATE'] = pd.to_datetime(merged_df['DATE'])
merged_df.set_index('DATE', inplace=True)
numeric_df = merged_df.apply(pd.to_numeric, errors='coerce')

growth_df = numeric_df.pct_change()

days_to_predict = 30
numeric_df['GSPC_Close'] = numeric_df['GSPC_Close'].shift(-days_to_predict)

merged_df.fillna(0, inplace=True)

# Train
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Assuming growth_df is a DataFrame
X = growth_df.drop(columns=['GSPC_Close']).values
y = growth_df['GSPC_Close'].values

non_nan_indices = ~pd.isna(y)
X = X[non_nan_indices]
y = y[non_nan_indices]

X[np.isinf(X)] = np.nan
X = pd.DataFrame(X).ffill().bfill().values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the sequence length for the LSTM
sequence_length = 1000  # You can adjust this window size

# Create sequences for LSTM
def create_sequences(data, target, seq_length):
    Xs, ys = [], []
    for i in range(len(data) - seq_length):
        Xs.append(data[i:i + seq_length])
        ys.append(target[i + seq_length])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)

# Convert to tensors
X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32).view(-1, 1)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # LSTM returns output and hidden state, we only need output
        lstm_out, _ = self.lstm(x)
        
        # Take the output of the last time step
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

# Hyperparameters
input_size = X_train.shape[1]  # Number of features per time step
hidden_size = 128  # LSTM hidden layer size
num_layers = 2  # Number of LSTM layers
output_size = 1  # Predicting a single value

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate the model
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

X_train_tensor = X_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set batch size
batch_size = 64  # You can adjust this based on available memory

# Create DataLoaders for training and testing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Training
best_test_loss = float('inf')
best_model_state = None
best_epoch = 0

total_epochs = 100_000
increment = 500

train_losses = []
test_losses = []

for epoch_range in range(increment, total_epochs + 1, increment):
    model.train()
    for epoch in range(increment):
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            y_pred = model(X_batch)
            
            # Compute loss
            loss = criterion(y_pred, y_batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        # Print average training loss per epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f'Epoch [{epoch + 1}/{increment}], Training Loss: {avg_train_loss:.6f}')

    model.eval()
    with torch.no_grad():
        test_loss_total = 0
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_test_pred = model(X_batch)
            test_loss = criterion(y_test_pred, y_batch)
            test_loss_total += test_loss.item()
        
        avg_test_loss = test_loss_total / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f'Epoch Range [{epoch_range}], Test Loss: {avg_test_loss:.6f}')
        
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model_state = model.state_dict()
            best_epoch = epoch_range
    
    model.train()

# Load the best model
model.load_state_dict(best_model_state)

# save the best model to file
torch.save(model.state_dict(), 'best_model.pth')

print(f'Best model found at epoch {best_epoch} with Test Loss: {best_test_loss:.6f}')

import matplotlib.pyplot as plt

# plot the training and test losses
plt.plot(train_losses[1000:], label='Training Loss')    
plt.plot(test_losses[1000:], label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# move the data back to the CPU
X_test_tensor = X_test_tensor.cpu()
y_test_tensor = y_test_tensor.cpu()

# we also neet to move the model back to the CPU
model = model.cpu()

import matplotlib.pyplot as plt

# Set the model to evaluation mode and generate predictions for the test set
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_tensor)

# Convert predictions and actual values to NumPy arrays for plotting
y_test_pred_np = y_test_pred.numpy().flatten()
y_test_actual_np = y_test_tensor.numpy().flatten()

# Plotting the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test_actual_np, label='Actual GSPC_Close', color='b')
plt.plot(y_test_pred_np, label='Predicted GSPC_Close', color='r', linestyle='--')
plt.title('Actual vs Predicted GSPC_Close')
plt.xlabel('Test Sample')
plt.ylabel('GSPC_Close Value')
plt.legend()
plt.show()

# Set the model in evaluation mode
model.eval()

# Ensure gradients are enabled for the input data
X_test_tensor.requires_grad_(True)

# Forward pass
y_test_pred = model(X_test_tensor)

# Compute gradients (backpropagate with respect to the inputs)
y_test_pred.mean().backward()

# Get the absolute value of gradients for each input feature
input_gradients = X_test_tensor.grad.mean(dim=0).numpy()

# Example: Use feature names in the plot
feature_names = growth_df.columns.drop('GSPC_Close')
plt.figure(figsize=(10, 6))
plt.bar(feature_names, input_gradients)
plt.xticks(rotation=90)
plt.title('Feature Importance based on Input Gradients')
plt.xlabel('Feature Name')
plt.ylabel('Average Absolute Gradient')
plt.show()