import numpy as np
import pandas as pd
import random
import string
import psycopg2.extras
from sqlalchemy import create_engine
from dateutil import parser
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from faker import Faker
from datetime import datetime, timedelta
import uuid
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense,Embedding
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import MSE
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

merged = pd.read_csv("merged-230704.csv")

################## chatgpt  ############################################################################################################
# selected_features = ['api_name', 'zipkin_timestamp', 'traceId', 'timestamp_5seconds', 'container_name',
#                      'cpu_usage', 'metricset_timestamp']
# df_selected = merged[selected_features]

# # Convert timestamp columns to datetime
# timestamp_columns = ['zipkin_timestamp', 'timestamp_5seconds', 'metricset_timestamp']
# for column in timestamp_columns:
#     df_selected[column] = pd.to_datetime(df_selected[column])

# # Convert categorical columns to one-hot encoding
# categorical_columns = ['api_name', 'container_name']
# df_encoded = pd.get_dummies(df_selected, columns=categorical_columns)

# # Normalize numeric columns (e.g., CPU usage)
# numeric_columns = ['cpu_usage']
# scaler = MinMaxScaler()
# df_encoded[numeric_columns] = scaler.fit_transform(df_encoded[numeric_columns])

# # Define the LSTM model

# model = Sequential() 
# model.add(Embedding(78550, 10)) 
# model.add(LSTM(10, dropout = 0.2, recurrent_dropout = 0.2)) 
# model.add(Dense(1, activation = 'sigmoid'))

# # Compile the model
# model.compile(loss='mean_squared_error', optimizer='adam')

# # Split the data into training and testing sets
# X = df_encoded.drop('cpu_usage', axis=1).values
# y = df_encoded['cpu_usage'].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshape the input data for LSTM (assuming a time window size of 10)
# window_size = 10
# X_train = np.reshape(X_train, (X_train.shape[0] - window_size + 1, window_size, X_train.shape[1]))
# y_train = y_train[window_size - 1:]

# # Train the LSTM model
# model.fit(X_train, y_train, epochs=10, batch_size=32)

# # Make predictions
# # X_test = np.reshape(X_test, (X_test.shape[0] - window_size + 1, window_size, X_test.shape[1]))
# y_pred = model.predict(X_test)

# # Inverse transform the predicted values
# y_pred = scaler.inverse_transform(y_pred)

# # Evaluate the model (optional)
# # Calculate evaluation metrics, e.g., root mean squared error (RMSE)
# # and compare them with the actual values y_test

# # Print the predicted CPU usage
# print(y_pred)





################## Ghan3  ############################################################################################################

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
# Convert timestamp columns to datetime
# timestamp_columns = ['zipkin_timestamp', 'timestamp_5seconds', 'metricset_timestamp']
# for column in timestamp_columns:
#     merged[column] = pd.to_datetime(merged[column])
# Normalize numeric columns (e.g., CPU usage)
numeric_columns = ['cpu_usage']
scaler = MinMaxScaler()
merged[numeric_columns] = scaler.fit_transform(merged[numeric_columns])

 
print(merged.shape) # (78550, 10)


model = Sequential() 
model.add(Embedding(78550, 10)) 
model.add(LSTM(10, dropout = 0.2, recurrent_dropout = 0.2)) 
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', 
   optimizer = 'adam', metrics = ['accuracy'])

X = merged[['service_name', 'api_name', 'zipkin_timestamp', 'duration', 'traceId', 'timestamp_5seconds', 'container_name', 'metricset_timestamp']]
y = merged['cpu_usage']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

model.fit(
   X_train, y_train, 
   batch_size = 32, 
   epochs = 20, 
   validation_data = (X_test, y_test)
)



score, acc = model.evaluate(X_test, y_test, batch_size = 32) 
   
print('Test score:', score) 
print('Test accuracy:', acc) 