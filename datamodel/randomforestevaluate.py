import numpy as np
import pandas as pd
import random
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



es_df = pd.read_csv("metricset-230704.csv")
zipkin_df = pd.read_csv("zipkin-230704.csv")
es_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
zipkin_df.drop(['Unnamed: 0'], axis = 1, inplace = True)


merged_df = pd.merge(zipkin_df,es_df, on='timestamp_5seconds')
es_df = es_df.drop(['new_index'],axis=1)
print(es_df)
print(len(es_df))

zipkin_df = zipkin_df.drop(['new_index'],axis=1)

print(zipkin_df)
print(len(zipkin_df))

merged = pd.merge(zipkin_df, es_df, on='timestamp_5seconds')
 
# combined_df = combined_df.dropna()
# print(combined_df)

merged = merged.sort_values('cpu_usage', ascending=False)



cpu_threshold = 0.01
merged['cache_recommendation'] = ['cache recommended' if cpu >= cpu_threshold else 'no cache recommendation' for cpu in merged['cpu_usage']]

print(merged['cache_recommendation'].value_counts())
print(merged.info())




# # Separate the features and target variable
# X = merged[['service_name', 'api_name', 'zipkin_timestamp', 'duration', 'traceId', 'timestamp_5seconds', 'container_name', 'cpu_usage', 'metricset_timestamp']]
# y = merged['cache_recommendation']

# # Convert datetime columns to numerical representation
# X['zipkin_timestamp'] = pd.to_datetime(X['zipkin_timestamp']).astype(int) / 10**9
# X['timestamp_5seconds'] = pd.to_datetime(X['timestamp_5seconds']).astype(int) / 10**9
# X['metricset_timestamp'] = pd.to_datetime(X['metricset_timestamp']).astype(int) / 10**9

# # Convert categorical variables to numerical representation
# label_encoders = {}
# for column in ['service_name', 'api_name', 'traceId', 'container_name']:
#     label_encoders[column] = LabelEncoder()
#     X[column] = label_encoders[column].fit_transform(X[column])


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)




# model = joblib.load('rf_model_epoch_100.pt')
# # Use the trained model to make predictions on the testing data
# y_pred = model.predict(X_test)

# # Calculate the evaluation metrics
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred,average="binary", pos_label="cache recommended" )
# recall = recall_score(y_test, y_pred,average="binary", pos_label="cache recommended")
# f1 = f1_score(y_test, y_pred,average="binary", pos_label="cache recommended")

# # Print the evaluation metrics
# print(f"Accuracy: {accuracy}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1-Score: {f1}")