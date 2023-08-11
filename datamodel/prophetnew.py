import pandas as pd
from prophet import Prophet
import time
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
import time
import numpy as np
metric1 = pd.read_csv("metricbeat-230802-ai-broker-1.csv")
metric2 = pd.read_csv("metricbeat-230802-ai-broker-2.csv")
metric3 = pd.read_csv("metricbeat-230802-ai-broker-3.csv")
metric4 = pd.read_csv("metricbeat-230802-ai-broker-4.csv")
metric5 = pd.read_csv("metricbeat-230802-ai-broker-5.csv")

zipkin = pd.read_csv("zipkin-230801-all-broker.csv")

# print(zipkin.shape)

# print(metric1.head())
# print(zipkin.head())

metric_lst = [metric1,metric2,metric3,metric4,metric5]
# metric_merge = pd.merge(metric1,metric2, on='container_name')
metric_merge = pd.concat([metric1,metric2,metric3,metric4,metric5])
metric_merge.drop(['Unnamed: 0'], axis = 1, inplace = True)
metric_merge.drop(['new_index'], axis = 1, inplace = True)
zipkin.drop(['Unnamed: 0'], axis = 1, inplace = True)
zipkin.drop(['new_index'], axis = 1, inplace = True)
# print(metric_merge)


# print(zipkin)
# print(metric_merge)


merged = pd.merge(metric_merge, zipkin,left_on='timestamp_5seconds', right_on='timestamp_5seconds',how='right')
# print(merged_inner)
# print(merged_inner.shape)

# print(merged_inner['cpu_usage'].mean())


merged_grouped = merged.groupby(merged['traceId'])
merged_grouped = merged_grouped['cpu_usage'].mean()
# print(merged_grouped)
merged_grouped = merged_grouped.reset_index()

merged_grouped['cpu_mean'] = merged_grouped['cpu_usage']

merged_final = pd.merge(merged,merged_grouped,on='traceId')
print(type(merged_grouped))

merged_final = merged_final.drop_duplicates(['traceId'],keep='first')
merged_final = merged_final.sort_values(by=['traceId'],axis=0,ascending=True)
merged_final.drop(['cpu_usage_x'],inplace=True, axis=1)
merged_final.drop(['cpu_usage_y'],inplace=True, axis=1)

print(merged_final)
# print(merged_inner)
# merged_grpuped = merged_grouped['cpu_usage'].mean().to_frame()


# merged_mean = merged.merge(merged,merged_grouped,on='cpu_usage')
# print(merged_mean)
# print(merged_mean.shape)


# print(len(merged_grouped.groups.keys())) ## 384937
# merged_keys = (list(merged_grouped.groups.keys()))
# print(merged_grouped['cpu_usage'].mean()['ff0e7d539fff2955'])


count = 0
# for idx, row in merged.iterrows()
# for idx, row in merged.iterrows():
#     for i in range(len(merged_keys)):
#         if merged_keys[i] == row['traceId']:
#             merged[idx] == merged_grouped['cpu_usage'].mean()[merged_keys[i]]
#         else:
#             continue
        
# print(merged)

# merged_trace_id = merged_grouped['cpu_usage'].mean().keys()

# cpu_mean = merged_grouped['cpu_usage'].mean()
# cpu_mean = cpu_mean.reset_index()
# print(cpu_mean)


# merged['cpu_mean'] = cpu_mean['cpu_usage']

# print(merged)



# merged_inner = merged_inner.drop_duplicates('traceId',keep='first')

# print(merged_inner)
# merged_inner = merged_inner.sort_values(by=['traceId'],axis=0,ascending=True)

# print(merged_inner)

# print(merged_inner)

# for idx, row in merged_inner.iterrows():
#     if row['traceId'] == merged_grouped['cpu_usage'].mean()[idx]:
#         merged_inner[idx]['cpu_mean'] = merged_grouped['cpu_usage'].mean()[idx]
#     else:
#         continue













# print(merged_grouped['cpu_usage'].mean()['00003604669f8986'])
    



# merged = pd.merge(zipkin,metric_merge,on=['timestamp_5seconds'])
# print(merged)
# print(merged.shape)
# print("merged saving ")
# merged.to_csv("230808merged.csv")
# Select the relevant columns for anomaly detection

# merged_new = merged.resample(rule='1S').mean()

# merged_new = pd.DataFrame()
# merged_new = merged
# merged_new['zipkin_timestamp'] = pd.to_datetime(merged['zipkin_timestamp'])
# merged_new.set_index('timestamp_5seconds',drop=False)
# print(merged_new)
# print(merged_new.groupby(['cpu_usage']).mean())









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
# plt.show()

# plt.plot(ds,y,'-')
# plt.plot(anomarly_ds,anomarly_y,'ro')
# # # plt.set_xlim((np.datetime64('2023-07-27 00:00'), np.datetime64('2023-07-028 23:59')))
# plt.show()

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
# print('The time used for the for-loop forecast is ', time.time()-start_time)
# # Take a look at the data
# print("Saving start")

# for_loop_forecast.to_csv("prophetmodel-230808-2.csv")
# print("Saving Done")
