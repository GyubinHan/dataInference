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
# from keras.preprocessing import LabelEncoder


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

zipkin1 = pd.read_csv("zipkin-230801-all-broker.csv")
zipkin2 = pd.read_csv("zipkin-230811-all-broker.csv")


# print(zipkin.shape)

# print(metric1.head())
# print(zipkin.head())

metric_lst = [metric1,metric2,metric3,metric4,metric5,metric6,metric7,metric8,metric9,metric10]
# metric_merge = pd.merge(metric1,metric2, on='container_name')
metric_merge = pd.concat([metric1,metric2,metric3,metric4,metric5,metric6,metric7,metric8,metric9,metric10])
zipkin_merge = pd.concat([zipkin1,zipkin2])

metric_merge.drop(['Unnamed: 0'], axis = 1, inplace = True)
metric_merge.drop(['new_index'], axis = 1, inplace = True)
zipkin_merge.drop(['Unnamed: 0'], axis = 1, inplace = True)
zipkin_merge.drop(['new_index'], axis = 1, inplace = True)
zipkin_merge.drop_duplicates(['traceId'],keep='first')
# print(metric_merge)

# print(zipkin)
# print(metric_merge)

##### data merge
merged = pd.merge(metric_merge, zipkin_merge,left_on='timestamp_5seconds', right_on='timestamp_5seconds',how='right')
# print(merged_inner)

# print(merged_inner['cpu_usage'].mean())




###### 
merged_grouped = merged.groupby(merged['traceId'])
merged_grouped = merged_grouped['cpu_usage'].mean()


merged_grouped = merged_grouped.reset_index()

merged_grouped['cpu_mean'] = merged_grouped['cpu_usage']

merged_final = pd.merge(merged,merged_grouped,on='traceId')

merged_final = merged_final.drop_duplicates(['traceId'],keep='first')
merged_final = merged_final.sort_values(by=['traceId'],axis=0,ascending=True)
# merged_final.drop(['cpu_usage_x'],inplace=True, axis=1)
merged_final.drop(['cpu_usage_y'],inplace=True, axis=1) 

merged_final.to_csv("/Users/e8l-20210032/Documents/GyubinHanAI/dataInference/merged-230901-all-broker.csv",sep=',',na_rep='NaN')
print("DONE")
# print(merged_final)
# merged_final['zipkin_timestamp'] = pd.to_datetime(merged_final['zipkin_timestamp']).dt.tz_localize(None)

# print(merged_final['zipkin_timestamp'][0])
# print(merged_final['zipkin_timestamp'][1])


# print("fail")
# merged = merged_final['zipkin_timestamp'].astype('float32')
# print("well done")
# merged_final = merged_final.sort_values(by=['zipkin_timestamp'],axis=0,ascending=True)
# print(merged_final)


merged_final = merged_final.dropna()




print(merged_final)
# print(merged_final)
# print(merged_final.shape)



print(merged_final['api_name'].unique())



####### LSTM modelling
# for i in range(len(merged_final)):
  # merged_final['zipkin_timestamp'][i]=  merged_final['zipkin_timestamp'][i].timestamp()
  
# for idx, row in merged_final.iterrows():
#   print(merged_final['zipkin_timestamp'][idx],row[idx])

# merged_final
# X = merged_final.iloc[:,:-1]
# y = merged_final.iloc[:,-1]

# print(X)
# print(y)
# X = merged_final.drop(['cpu_mean'],axis=1).values()
# y = merged['cpu_mean'].values()
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=28)
print(type(X_train))

# print(X_train.shape[1])
# X_train = X_train.values
# X_test = X_test.values
# y_train = y_train.values
# y_test = y_test.values

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# X_train = X_train.iloc[:,:].to_numpy()
# X_test = X_test.iloc[:,:].to_numpy()
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# print(len(X_train))
# print(int(X_train)/46)
# print(X_train)
# X_train = X_train.reshape(len(X_train)/46,46,X_train.shape[1])
# X_test = X_test.reshape(len(X_test)/46,46,X_test.shape[1])


X_train = X_train.reshape(236647,2,X_train.shape[1])
X_test = X_test.reshape(59162,2,X_test.shape[1])


print("Model learning start")
learning_rate = 0.01
model = Sequential()

model.add(LSTM(50, input_shape=(2,X_train.shape[1])))
model.add(Dense(100))
model.add(Dense(4))
model.compile(optimizer = 'adam', loss='mae')

history = model.fit(X_train,y_train, epochs=50, batch_size=16, validation_data=(X_test,y_test), verbose = 2, shuffle=False)

plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
plt.legend()
plt.show()

# prophet_data = pd.DataFrame()
# # prophet_data['ds'] = merged['timestamp_5seconds']
# prophet_data['ds'] = pd.to_datetime(merged['zipkin_timestamp'])

# prophet_data['ticker'] = merged['api_name'] 
# # prophet_data['y'] = 
# prophet_data['y'] = merged['cpu_usage']*merged['duration']
# prophet_data['duration'] = merged['duration']
# # print(prophet_data.info())
# groups_by_api_name = prophet_data.groupby('ticker')
# # print(groups_by_api_name.groups.keys())



# ticker_list = []
# for i in groups_by_api_name.groups.keys():
#     # print(i)
#     ticker_list.append(i)

# # prophet_data y
# prophet_data['utility'] = merged['cpu_usage'] * merged['duration']

# # 날짜별
# filtered_df = prophet_data.loc[(prophet_data['ds'] >= '2023-7-27 00:00') & (prophet_data['ds'] < '2023-07-27 20:00')]
# print(filtered_df)

# ds = []
# y = []

# anomarly_y = []
# anomarly_ds = []
# for idx, row in filtered_df.iterrows():
#     if row['ticker'] == 'get /ndxpro/v1/aibroker/entities/iot/history':
#         if row['utility'] <= 0.04 or row['utility'] >= 0.14:
#             anomarly_ds.append(row['ds'])
#             anomarly_y.append(row['utility'])        
#         else:
#             ds.append(row['ds'])
#             y.append(row['utility'])
#     else:
#         continue        



##### anomalies
# count = 0
# for idx, row in filtered_df.iterrows():
#     if row.loc[(row['ds'] == '2023-07-27 18:28:51')]:
#         count += 1
#     else:
#         continue
# print(count)
# for i in range(len(ds)):
#     if y[i] >= 0.04 and y[i] <= 0.17:
#         anomarly_y.append(y[i])
#     else:
#         continue


# filtered_df['duration'].plot()

# plt.plot(ds,y,'-')
# plt.plot(anomarly_ds,anomarly_y,'ro')
# # # plt.set_xlim((np.datetime64('2023-07-27 00:00'), np.datetime64('2023-07-028 23:59')))
# plt.show()



########### Prophet#######################################################
# # model = Prophet(interval_width=0.9)
# # model.add_regressor('cpu_usage', standardize=False)
# # model.add_regressor('duration', standardize=False)
# def train_and_forecast(group):
#   # Initiate the model
#   m = Prophet()
# #   m.add_regressor('cpu_usage',standardize=False)
# #   m.add_regressor('duration',standardize=False)
  
  
  
#   # Fit the model
#   m.fit(group)
#   # Make predictions
#   future = m.make_future_dataframe(periods=24,freq ='H')
#   forecast = m.predict(future)[['ds','yhat', 'yhat_lower', 'yhat_upper']]
#   forecast['ticker'] = group['ticker'].iloc[0]
  
#   # Return the forecasted results
#   return forecast[['ds', 'ticker', 'yhat', 'yhat_upper', 'yhat_lower']]


# # Start time
# start_time = time.time()
# # Create an empty dataframe
# for_loop_forecast = pd.DataFrame()
# # Loop through each ticker
# for ticker in ticker_list:
#   # Get the data for the ticker
#   group = groups_by_api_name.get_group(ticker)  
#   # Make forecast
#   forecast = train_and_forecast(group)
#   print("Done :", group)
#   # Add the forecast results to the dataframe
#   for_loop_forecast = pd.concat((for_loop_forecast, forecast))






# # print('The time used for the for-loop forecast is ', time.time()-start_time)
# # # Take a look at the data
# # print("Saving start")

# # for_loop_forecast.to_csv("prophetmodel-230808-2.csv")
# # print("Saving Done")
