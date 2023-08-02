import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.offline as py
import cmdstanpy
from sklearn.preprocessing import LabelEncoder

cmdstanpy.install_cmdstan()
cmdstanpy.install_cmdstan(compiler=True) 
encode = LabelEncoder()

# py.init_notebook_mode()


metric1 = pd.read_csv("metricbeat-230731-ai-broker-1.csv")
metric2 = pd.read_csv("metricbeat-230731-ai-broker-2.csv")
metric3 = pd.read_csv("metricbeat-230731-ai-broker-3.csv")
metric4 = pd.read_csv("metricbeat-230731-ai-broker-4.csv")
metric5 = pd.read_csv("metricbeat-230731-ai-broker-5.csv")

zipkin = pd.read_csv("zipkin-230731-ai-brokerservice.csv")



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

print(train_dataset)
print(type(train_dataset))


## plot
# train_dataset.set_index('ds').plot()



# prophet modeling
prophet_basic = Prophet()
prophet_basic.fit(train_dataset)


future = prophet_basic.make_future_dataframe(periods=1)
print(future.tail())


forecast = prophet_basic.predict(future)
print(forecast)
print("hello world")

# # cpu plotting
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