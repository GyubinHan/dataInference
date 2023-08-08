import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from sklearn.model_selection import train_test_split

# Load your data into a pandas DataFrame
# Replace 'your_data.csv' with the path to your data file


metric1 = pd.read_csv("metricbeat-230802-ai-broker-1.csv")
metric2 = pd.read_csv("metricbeat-230802-ai-broker-2.csv")
metric3 = pd.read_csv("metricbeat-230802-ai-broker-3.csv")
metric4 = pd.read_csv("metricbeat-230802-ai-broker-4.csv")
metric5 = pd.read_csv("metricbeat-230802-ai-broker-5.csv")

zipkin = pd.read_csv("zipkin-230801-all-broker.csv")



metric_lst = [metric1,metric2,metric3,metric4,metric5]
metric_merge = pd.merge(metric1,metric2, on='container_name')
metric_merge = pd.concat([metric1,metric2,metric3,metric4,metric5])
metric_merge.drop(['Unnamed: 0'], axis = 1, inplace = True)
metric_merge.drop(['new_index'], axis = 1, inplace = True)
zipkin.drop(['Unnamed: 0'], axis = 1, inplace = True)
zipkin.drop(['new_index'], axis = 1, inplace = True)
# print(metric_merge)

merged = pd.merge(metric_merge,zipkin,on=['timestamp_5seconds'])
# Select the relevant columns for anomaly detection



# selected_columns = ['cpu_usage', 'timestamp_5seconds', 'duration']
# data = merged[selected_columns]

# # Convert timestamp to datetime and set it as the index
# data['timestamp_5seconds'] = pd.to_datetime(data['timestamp_5seconds'])
# data.set_index('timestamp_5seconds', inplace=True)

# # Normalize the data using Min-Max scaling
# scaler = MinMaxScaler()
# data_normalized = scaler.fit_transform(data)

# # Define the sequence length for LSTM input
# sequence_length = 10  # You can adjust this based on your data size and characteristics

# # Prepare the input sequences and labels
# X = []
# y = []
# for i in range(len(data_normalized) - sequence_length):
#     X.append(data_normalized[i:i + sequence_length])
#     y.append(data_normalized[i + sequence_length][0])  # We'll use cpu_usage as the target for anomaly detection

# X = np.array(X)
# y = np.array(y)

# # Split the data into training and testing sets
# split_ratio = 0.8
# split_index = int(len(X) * split_ratio)

# X_train, X_test = X[:split_index], X[split_index:]
# y_train, y_test = y[:split_index], y[split_index:]

# # Build the LSTM model
# model = Sequential()
# model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Set up early stopping to prevent overfitting 
# early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

# # Define the filepath to save the model
# checkpoint_filepath = '/Users/e8l-20210032/Documents/GyubinHanAI/dataInference/model_checkpoint/weights.{epoch:02d}-{val_loss:.2f}.h5'
# os.makedirs('model_checkpoint', exist_ok=True)

# # Set up ModelCheckpoint to save the model per 10 epochs
# model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, save_freq='epoch', period=10)

# # Train the model
# model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.1, callbacks=[early_stopping])

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Denormalize the predictions and the actual values
# y_pred_denormalized = scaler.inverse_transform(y_pred)
# y_test_denormalized = scaler.inverse_transform(y_test.reshape(-1, 1))

# # Calculate the mean squared error (MSE) for anomaly detection
# mse = np.mean((y_test_denormalized - y_pred_denormalized.reshape(-1, 1)) ** 2)
# print("Mean Squared Error (MSE):", mse)

# # Define a threshold for anomaly detection (e.g., 2 times the standard deviation of the MSE)
# threshold = 2 * np.std((y_test_denormalized - y_pred_denormalized.reshape(-1, 1)) ** 2)

# # Detect anomalies based on the threshold
# anomalies = np.where((y_test_denormalized - y_pred_denormalized.reshape(-1, 1)) ** 2 > threshold)[0]

# # Print the indices of anomalies in the original DataFrame
# print("Anomaly indices:")
# print(anomalies)


# Load your data into a pandas DataFrame
# Replace 'your_data.csv' with the path to your data file



######################### new code ##############################
# Select the relevant columns for anomaly detection
selected_columns = ['cpu_usage', 'timestamp_5seconds', 'duration']
data = merged[selected_columns]

# Convert timestamp to datetime and set it as the index
data['timestamp_5seconds'] = pd.to_datetime(data['timestamp_5seconds'])
data.set_index('timestamp_5seconds', inplace=True)

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)



###############3 X,Y list 
# Define the sequence length for LSTM input
sequence_length = 10  # You can adjust this based on your data size and characteristics

# Prepare the input sequences and labels
X = []
y = []
for i in range(len(data_normalized) - sequence_length):
    X.append(data_normalized[i:i + sequence_length])
    y.append(data_normalized[i + sequence_length][0])  # We'll use cpu_usage as the target for anomaly detection

X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

# x_df_scaled = pd.DataFrame(data_normalized, None, X_train.keys())
# x_df_scaled_expanded = np.expand_dims(x_df_scaled, axis=0)


# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
# X_train,x_test, y_train,y_test = train_test_split(data['timestamp_5seconds'],data['cpu_usage', 'duration'])
# print(X_train)
# print("*************")
# print(y_train)

# print(X_train.shape[1])
# print(X_train.shape[2])
# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Set up early stopping to prevent overfitting
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

# Define the filepath to save the model
checkpoint_filepath = 'model_checkpoint/weights.{epoch:02d}-{val_loss:.2f}.h5'
os.makedirs('model_checkpoint', exist_ok=True)

# Set up ModelCheckpoint to save the model per 10 epochs
checkpoint_filepath = '/Users/e8l-20210032/Documents/GyubinHanAI/dataInference/model_checkpoint/weights2.{epoch:02d}-{val_loss:.2f}.h5'
model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, save_freq='epoch', period=10)

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.1, callbacks=[early_stopping, model_checkpoint])

# Make predictions on the test set
y_pred = model.predict(X_test)
print(y_pred)
print(y_pred.shape)
# Denormalize the predictions and the actual values
y_pred_denormalized = scaler.inverse_transform(y_pred).flatten()
y_test_denormalized = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Calculate the mean squared error (MSE) for anomaly detection
mse = np.mean((y_test_denormalized - y_pred_denormalized) ** 2)
print("Mean Squared Error (MSE):", mse)

# Define a threshold for anomaly detection (e.g., 2 times the standard deviation of the MSE)
threshold = 2 * np.std((y_test_denormalized - y_pred_denormalized) ** 2)

# Detect anomalies based on the threshold
anomalies = np.where((y_test_denormalized - y_pred_denormalized) ** 2 > threshold)[0]

# Print the indices of anomalies in the original DataFrame
print("Anomaly indices:")
print(anomalies)

