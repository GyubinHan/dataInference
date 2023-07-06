import numpy as np
import pandas as pd
import random
import string
import psycopg2.extras
import pandas as pd
from sqlalchemy import create_engine
import time
from dateutil import parser
import os
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.metrics import classification_report
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from faker import Faker
from datetime import datetime, timedelta
import uuid
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    # def forward(self, x):
    #     batch_size = x.size(0)
    #     seq_len = x.size(1)
    #     h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
    #     c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
    #     out, _ = self.lstm(x, (h0, c0))
    #     out = self.fc(out[:, -1, :])
    #     return out
    
    def forward(self, x, h=None, c=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        # print(type(h))
        num_directions =2
        # if h is None and c is None:
        hx = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(device)  # Initialize the hidden state
        cx = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(device)  # Initialize the cell state
            # print(type(h))
            
        # else:
        #     # print(type(h))
        #     h = h.squeeze(0)  # Add an extra dimension for the num_directions
        #     c = c.squeeze(0)  # Add an extra dimension for the num_directions
        
        hx = hx.view(hx.size(0), -1)
        cx = cx.view(cx.size(0), -1)     
        out, _ = self.lstm(x, (hx, cx))
        out = self.fc(out[:, -1, :])
        return out

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, df):
        self.data = df.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx, :-1], dtype=torch.float32)
        y = torch.tensor(self.data[idx, -1], dtype=torch.float32)
        return x, y

def merged_dict(service_name,api_name,zipkin_timestamp,duration,traceId,timestamp_5seconds,container_name,cpu_usage,metricset_timestamp):
    new_dict = {
        "service_name":service_name,
        "api_name":api_name,
        "zipkin_timestamp":zipkin_timestamp,
        "duration":duration,
        "traceId":traceId,
        "timestamp_5seconds":timestamp_5seconds,
        "container_name":container_name,
        "cpu_usage":cpu_usage,
        "metricset_timestamp":metricset_timestamp
    }
    
    return new_dict


merged = pd.read_csv("merged-230704.csv")
merged['zipkin_timestamp'] = pd.to_datetime(merged['zipkin_timestamp'])
merged['timestamp_5seconds'] = pd.to_datetime(merged['timestamp_5seconds'])
merged['metricset_timestamp'] = pd.to_datetime(merged['metricset_timestamp'])
merged['zipkin_timestamp'] = merged[['zipkin_timestamp']].apply(lambda x: x[0].timestamp(), axis=1).astype(int)
merged['timestamp_5seconds'] = merged[['timestamp_5seconds']].apply(lambda x: x[0].timestamp(), axis=1).astype(int)
merged['metricset_timestamp'] = merged[['metricset_timestamp']].apply(lambda x: x[0].timestamp(), axis=1).astype(int)

merged['duration'] = merged['duration'].astype(float)
merged['cpu_usage'] = merged['cpu_usage'].astype(float)

# print(merged)

# Define the string columns to be converted
string_columns = ['service_name', 'api_name', 'traceId', 'container_name']

label_encoder = LabelEncoder()
for column in string_columns:
    merged[column] = label_encoder.fit_transform(merged[column])

# Convert unsupported data types to supported types
merged['duration'] = merged['duration'].astype(float)
merged['cpu_usage'] = merged['cpu_usage'].astype(float)
# merged['service_name'] = merged['service_name'].astype(np.string_)
# merged['api_name'] = merged['api_name'].astype(np.string_)
# merged['traceId'] = merged['traceId'].astype(np.string_)
# merged['container_name'] = merged['container_name'].astype(np.string_)



# print(merged)
# print(merged.info())


# Define hyperparameters
input_size = 8  # Number of features in the input data
hidden_size = 64  # Number of LSTM units
num_layers = 2  # Number of LSTM layers
output_size = 1  # Number of output units
num_epochs = 1000
batch_size = 32


# device select
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create the LSTM model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Create the DataLoader
dataset = CustomDataset(merged)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# print(dataset.shape())
# Training loop
best_loss = float('inf')
best_epoch = 0
best_batch = 0


print(len(dataloader))
print(len(dataloader[:]))
print(len(dataloader[1]))


    # print(outputs)
print("training model start")
total_iterations = 0
num_directions = 1
for epoch in range(num_epochs):
    # hx = torch.zeros(num_layers * num_directions, batch_size, hidden_size).to(device)  # Initialize the hidden state
    # cx = torch.zeros(num_layers * num_directions, batch_size, hidden_size).to(device)  # Initialize the cell state
    # print(hx)
    for batch, (inputs, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Adjust the dimensions of hx and cx
        # hx = hx.unsqueeze(0)
        # cx = cx.unsqueeze(0)
        
        # Adjust the dimensions of hx and cx
 
        # hx = hx.view(hx.size(0), -1)
        # cx = cx.view(cx.size(0), -1)        
        # print(hx)
        # print(hx.size())
        # print(hx.size())
        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        # loss = criterion(outputs, targets)
        loss = criterion(outputs[:, -1, :], targets.unsqueeze(1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            best_batch = batch

        total_iterations += 1

        if total_iterations % 100 == 0:
            # Save model checkpoint every 100 iterations
            checkpoint_path = f"model_iteration_{total_iterations}.pt"
            torch.save(model.state_dict(), checkpoint_path)

        
print(f"Best epoch: {best_epoch + 1}")
print(f"Best batch: {best_batch + 1}")

# Save the final trained model
torch.save(model.state_dict(), "final_model.pt")
