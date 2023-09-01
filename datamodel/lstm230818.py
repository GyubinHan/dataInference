import pandas as pd
from prophet import Prophet
import time
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing import LabelEncoder


## 데이터 로드
metric1 = pd.read_csv("metricbeat-230802-ai-broker-1.csv")
metric2 = pd.read_csv("metricbeat-230802-ai-broker-2.csv")
metric3 = pd.read_csv("metricbeat-230802-ai-broker-3.csv")
metric4 = pd.read_csv("metricbeat-230802-ai-broker-4.csv")
metric5 = pd.read_csv("metricbeat-230802-ai-broker-5.csv")
metric6 = pd.read_csv("metricbeat-230811-ai-broker-1.csv")
metric7 = pd.read_csv("metricbeat-230811-ai-broker-2.csv")
metric8 = pd.read_csv("metricbeat-230811-ai-broker-3.csv")
metric9 = pd.read_csv("metricbeat-230811-ai-broker-4.csv")
metric10 = pd.read_csv("metricbeat-230811-ai-broker-5.csv")
metric11 = pd.read_csv("metricbeat-230818-ai-broker-1.csv")
metric12 = pd.read_csv("metricbeat-230818-ai-broker-2.csv")
metric13 = pd.read_csv("metricbeat-230818-ai-broker-3.csv")
metric14 = pd.read_csv("metricbeat-230818-ai-broker-4.csv")
metric15 = pd.read_csv("metricbeat-230818-ai-broker-5.csv")



zipkin1 = pd.read_csv("zipkin-230801-all-broker.csv")
zipkin2 = pd.read_csv("zipkin-230811-all-broker.csv")
zipkin3 = pd.read_csv("zipkin-230818-all-broker.csv")



# print(zipkin.shape)

# print(metric1.head())
# print(zipkin.head())

metric_lst = [metric1,metric2,metric3,metric4,metric5,metric6,metric7,metric8,metric9,metric10,metric11,metric12,metric13,metric14,metric15]
# metric_merge = pd.merge(metric1,metric2, on='container_name')
metric_merge = pd.concat([metric1,metric2,metric3,metric4,metric5,metric6,metric7,metric8,metric9,metric10,metric11,metric12,metric13,metric14,metric15])
zipkin_merge = pd.concat([zipkin1,zipkin2,zipkin2,zipkin3])

metric_merge.drop(['Unnamed: 0'], axis = 1, inplace = True)
metric_merge.drop(['new_index'], axis = 1, inplace = True)
zipkin_merge.drop(['Unnamed: 0'], axis = 1, inplace = True)
zipkin_merge.drop(['new_index'], axis = 1, inplace = True)
zipkin_merge.drop_duplicates(['traceId'],keep='first')

##### data merge
merged = pd.merge(metric_merge, zipkin_merge,left_on='timestamp_5seconds', right_on='timestamp_5seconds',how='right')
# print(merged)
# print(merged_inner)

# print(merged_inner['cpu_usage'].mean())







merged = merged.dropna()
# print(merged)

###### 
merged_grouped = merged.groupby(merged['traceId'])
merged_grouped = merged_grouped['cpu_usage'].mean()


merged_grouped = merged_grouped.reset_index()

merged_grouped['cpu_mean'] = merged_grouped['cpu_usage']

merged_final = pd.merge(merged,merged_grouped,on='traceId')
# print(merged_final)
print(merged_final)
print(merged_final.drop_duplicates(['traceId'],keep='first'))
merged_final_pivot = pd.pivot_table(merged_final, values='cpu_usage_x', index=['zipkin_timestamp'], columns=['api_name'])

# print(merged_final_pivot)
merged_final = merged_final.drop_duplicates(['traceId'],keep='first')
merged_final = merged_final.sort_values(by=['traceId'],axis=0,ascending=True)
# merged_final.drop(['cpu_usage_x'],inplace=True, axis=1)
merged_final.drop(['cpu_usage_y'],inplace=True, axis=1)

merged_final = merged_final.dropna()

# print(merged_final)
# print(merged_final.sort_values(by=['zipkin_timestamp']))

# merged_final_pivot = pd.pivot_table(merged_final, values='cpu_usage_x', index=['zipkin_timestamp'], columns=['api_name'])

# print(merged_final_pivot)


# for i in range(0, len(merged_final_pivot.columns)):
#     merged_final_pivot.iloc[:,i].interpolate(inplace = True)
 
 
 
 
 
 
 
 
# import plotly
# import plotly.express as px
# import plotly.graph_objects as go
# merged_final_pivot.reset_index(inplace=True)
# merged_final_pivot['zipkin_timestamp'] = pd.to_datetime(merged_final_pivot['zipkin_timestamp'])

# trace1 = go.Scatter(
#  x = merged_final_pivot['zipkin_timestamp'],
#  y = merged_final_pivot['get /ndxpro/v1/aibroker/entities'],
#  mode = 'lines',
#  name = 'cpu_1'
# )
# trace2 = go.Scatter(
#  x = merged_final_pivot['zipkin_timestamp'],
#  y = merged_final_pivot['get /ndxpro/v1/aibroker/entities/iot'],
#  mode = 'lines',
#  name = 'cpu_2'
# )
# trace3 = go.Scatter(
#  x = merged_final_pivot['zipkin_timestamp'],
#  y = merged_final_pivot['get /ndxpro/v1/aibroker/entities/{entityid}'],
#  mode = 'lines',
#  name = 'cpu_3'
# )
# trace4 = go.Scatter(
#  x = merged_final_pivot['zipkin_timestamp'],
#  y = merged_final_pivot['get /ndxpro/v1/aibroker/entities/iot/history'],
#  mode = 'lines',
#  name = 'cpu_4'
# )
# trace5 = go.Scatter(
#  x = merged_final_pivot['zipkin_timestamp'],
#  y = merged_final_pivot['get /ndxpro/v1/aibroker/entities/iot/temporal'],
#  mode = 'lines',
#  name = 'cpu_5'
# )
# trace6 = go.Scatter(
#  x = merged_final_pivot['zipkin_timestamp'],
#  y = merged_final_pivot['get /ndxpro/v1/aibroker/entities/iot/{entityid}'],
#  mode = 'lines',
#  name = 'cpu_6'
# )

# layout = go.Layout(
#  title = "CPU usage",
#  xaxis = {'title' : "Date"},
#  yaxis = {'title' : "Cpu Usage"}
# )

# fig = go.Figure(data=[trace1, trace2, trace3,trace4,trace5,trace6], layout=layout)
# fig.show()