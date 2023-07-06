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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.metrics import classification_report
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from faker import Faker
from datetime import datetime, timedelta
import uuid

def generate_random_id(length):
    characters = string.digits + string.ascii_lowercase
    random_id = ''.join(random.choice(characters) for _ in range(length))
    return random_id


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



es_df = pd.read_csv("metricset-230704.csv")
zipkin_df = pd.read_csv("zipkin-230704.csv")
es_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
zipkin_df.drop(['Unnamed: 0'], axis = 1, inplace = True)


merged_df = pd.merge(zipkin_df,es_df, on='timestamp_5seconds')
es_df = es_df.drop(['new_index'],axis=1)
# print(es_df)
# print(len(es_df))

zipkin_df = zipkin_df.drop(['new_index'],axis=1)

# print(zipkin_df)
# print(len(zipkin_df))

merged = pd.merge(zipkin_df, es_df, on='timestamp_5seconds')
 
# combined_df = combined_df.dropna()
# print(combined_df)

print(merged.info())


cpu_threshold = 0.030

# X = merged[['service_name', 'api_name', 'zipkin_timestamp', 'duration', 'traceId', 'timestamp_5seconds', 'container_name', 'cpu_usage', 'metricset_timestamp']]
# y = merged['cache_recommendation']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(merged)
print(len(merged))
# print(merged['cache_recommendation'].value_counts())


# Initialize Faker object
fake = Faker()

# Set the random seed for reproducibility
random.seed(28)

# Define the number of fake data points to generate
num_samples = 1000

# Generate synthetic data
data = []
print("Fake data start creating")
for _ in range(num_samples):
    service_name = 'iotdatabroker-service'
    api_name = 'get /ndxpro/v2/iotbroker/temporal/entities'
    zipkin_timestamp = fake.date_time_this_year()
    duration = random.uniform(1.0, 100.0)
    trace_id = uuid.uuid4().hex[:16]
    # trace_id = generate_random_id(16)
    timestamp_5seconds = zipkin_timestamp - timedelta(seconds=zipkin_timestamp.second % 5)
    container_name = 'iot-data-broker-1'
    cpu_usage = random.uniform(0.03, 0.10)
    metricset_timestamp = zipkin_timestamp
    merge_dict = merged_dict(service_name,api_name,zipkin_timestamp,duration,trace_id,timestamp_5seconds,container_name,cpu_usage,metricset_timestamp)
    # print(merge_dict)
    # merged = merged.append(merge_dict,ignore_index=True)
    # dictionary append in row
    merged.loc[len(merged)] = list(merge_dict.values())
    

merged['cache_recommendation'] = ['cache recommended' if cpu >= cpu_threshold else 'cache need but work fine' if cpu>= 0.2 else 'no cache recommendation'for cpu in merged['cpu_usage']]

print("Fake data creating done")


merged = merged.sort_values('cpu_usage', ascending=False)
print(merged)
print(len(merged))
print(merged['cache_recommendation'].value_counts())

# print("saving to csv")

# zipkin_df.to_csv("/Users/e8l-20210032/Documents/GyubinHanAI/dataInference/merged_fakedata30000_230705.csv",sep=',',na_rep='NaN')

# print("CSV SAVING DONE")

print("Done")

# # Separate the categorical and numerical columns
# categorical_columns = ['service_name','api_name', 'duration', 'traceId','container_name' ]
# numerical_columns = ['zipkin_timestamp','timestamp_5seconds', 'cpu_usage', 'metricset_timestamp']

# # Apply one-hot encoding to the categorical columns
# encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
# encoded_categorical = encoder.fit_transform(merged[categorical_columns])


# Separate the features and target variable
X = merged[['service_name', 'api_name', 'zipkin_timestamp', 'duration', 'traceId', 'timestamp_5seconds', 'container_name', 'cpu_usage', 'metricset_timestamp']]
y = merged['cache_recommendation']

# Convert datetime columns to numerical representation
X['zipkin_timestamp'] = pd.to_datetime(X['zipkin_timestamp']).astype(int) / 10**9
X['timestamp_5seconds'] = pd.to_datetime(X['timestamp_5seconds']).astype(int) / 10**9
X['metricset_timestamp'] = pd.to_datetime(X['metricset_timestamp']).astype(int) / 10**9

# Convert categorical variables to numerical representation
label_encoders = {}
for column in ['service_name', 'api_name', 'traceId', 'container_name']:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)



rf_classifier = RandomForestClassifier()
# Define the number of epochs and the interval for saving the model
num_epochs = 100
save_interval = 100


print("Training start")
# Train the model in multiple epochs
for epoch in range(num_epochs):
    
    # Train the model on the training data
    rf_classifier.fit(X_train, y_train)
    
    # Evaluate the model on the testing data
    y_pred = rf_classifier.predict(X_test)
    
    # Print classification report for the current epoch
    print(f"Epoch {epoch + 1}:")
    print(classification_report(y_test, y_pred))
    
    # Save the model file every save_interval epochs
    if (epoch + 1) % save_interval == 0:
        model_file_name = f"rf_model_epoch_{epoch + 1}.pt"
        joblib.dump(rf_classifier, model_file_name)
        print(f"Saved model file: {model_file_name}")
        # You can also store the model file in a specific directory if needed
        # os.rename(model_file_name, os.path.join('models', model_file_name))
        