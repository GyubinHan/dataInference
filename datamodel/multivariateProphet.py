import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.offline as py
import cmdstanpy

from sklearn.preprocessing import LabelEncoder
import time
# cmdstanpy.install_cmdstan()
# cmdstanpy.install_cmdstan(compiler=True) 
encode = LabelEncoder()

# py.init_notebook_mode()


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
# merged.drop(['Unnamed: 0'], axis = 1, inplace = True)
# merged.drop(['new_index'], axis = 1, inplace = True)
# print(merged)
print(merged['api_name'].unique())

api_name_lst = []
for i in merged["api_name"].unique():
    api_name_lst.append(i)


train_dataset = pd.DataFrame()
train_dataset['ds'] = pd.to_datetime(merged['timestamp_5seconds'])
train_dataset['y'] = merged['cpu_usage']
train_dataset['duration'] = merged['duration']
train_dataset['ticker'] = merged['api_name']
print(train_dataset)
print(type(train_dataset))


groups_by_ticker = train_dataset.groupby('ticker')
print(groups_by_ticker.groups.keys())
print(train_dataset.info())
model = Prophet()
# m = Prophet()
# m.add_regressor('duration')
# m.fit(train_dataset)
###### Prophet model
def train_and_forecast(group):
  # Initiate the model
  m = Prophet()
  
  # Fit the model
  m.fit(group)
  # Make predictions
  future = m.make_future_dataframe(periods=30)
  forecast = m.predict(future)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
  forecast['ticker'] = group['ticker'].iloc[0]
  
  # Return the forecasted results
  return forecast[['ds', 'ticker', 'yhat', 'yhat_upper', 'yhat_lower']]

print(merged)
print(merged.info())
# # Start time
# start_time = time.time()
# # Create an empty dataframe
# for_loop_forecast = pd.DataFrame()
# # Loop through each ticker
# for ticker in api_name_lst:
#   # Get the data for the ticker
#   group = groups_by_ticker.get_group(ticker)  
#   # Make forecast
#   forecast = train_and_forecast(group)
#   # Add the forecast results to the dataframe
#   for_loop_forecast = pd.concat((for_loop_forecast, forecast))
# print('The time used for the for-loop forecast is ', time.time()-start_time)
# # Take a look at the data
# # for_loop_forecast.head()
# print(for_loop_forecast)


# print("saving csv for forecast")
# for_loop_forecast.to_csv("/Users/e8l-20210032/Documents/GyubinHanAI/dataInference/23-08-03")
# print("Done")
# plt.plot(for_loop_forecast['ds'],for_loop_forecast['yhat'])
# # plt.plot(for_loop_forecast['ds'],for_loop_forecast['yhat'])
# plt.show()
# plot
# train_dataset.set_index('ds').plot()


# # plt.plot(train_dataset['ds'],train_dataset[''])
# # plt.show()
# # prophet modeling
# prophet_basic = Prophet()
# prophet_basic.fit(train_dataset)


# future = prophet_basic.make_future_dataframe(periods=1)
# print(future.tail())


# forecast = prophet_basic.predict(future)
# print(forecast)

# # # cpu plotting
# count = 0
# X = []
# Y = []
# for idx, row in merged.iterrows():
#     if row['api_name'] == api_name_lst[0]: # row['api_name'] == api_name_lst[1] and 
#         X.append(row['timestamp_5seconds'])
#         Y.append(row['cpu_usage'])
#         count += 1
#         print(count)
#     else:
#         count += 1
#         print(count)
#         continue
# # print(count)

# plt.plot(X,Y,'-')
# plt.show()