import os, random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from tqdm.notebook import trange
# from TaPR_pkg import etapr
from pathlib import Path
import time

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

# Set seeds to make the experiment more reproducible.
def seed_everything(seed=28):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 28
seed_everything()

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
merged = merged.dropna()


merged_grouped = merged.groupby(merged['traceId'])
merged_grouped = merged_grouped['cpu_usage'].mean()
merged_grouped = merged_grouped.reset_index()
merged_grouped['cpu_mean'] = merged_grouped['cpu_usage']
merged_final = pd.merge(merged,merged_grouped,on='traceId')
merged_final.drop(['cpu_usage_y'],inplace=True, axis=1)
print(merged_final)
merged_final = merged_final.drop_duplicates(['traceId'],keep='first')


merged_final['metricset_timestamp'] = pd.to_datetime(merged_final['metricset_timestamp'])
merged_final['metricset_timestamp'].min(), merged_final['metricset_timestamp'].max()

train, test = merged_final.loc[merged_final['metricset_timestamp'] <= '2023-08-05'], merged_final.loc[merged_final['metricset_timestamp'] > '2023-08-07']
print(train.shape, test.shape)


TIME_STEPS=30

def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        Xs.append(X.iloc[i:(i+time_steps)].values)
        ys.append(y.iloc[i+time_steps])
    
    return np.array(Xs), np.array(ys)

X_train, y_train = create_sequences(train[['cpu_mean']], train['cpu_mean'])
X_test, y_test = create_sequences(test[['cpu_mean']], test['cpu_mean'])

print(f'Training shape: {X_train.shape}')
print(f'Testing shape: {X_test.shape}')


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow import keras
tf.random.set_seed(28)

model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(rate=0.2))
model.add(RepeatVector(X_train.shape[1]))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(TimeDistributed(Dense(X_train.shape[2])))
model.compile(optimizer='adam', loss='mae')
model.summary()

# model save callbacks
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    filepath="/Users/e8l-20210032/Documents/GyubinHanAI/dataInference/lstmautoencodermodel/model_checkpoint_{epoch:02d}.h5",
    save_best_only=False,
    save_weights_only=False,
    period=3  # Save every 10 epochs
)


history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[checkpoint_callback], shuffle=False)
# plot cpu_usage
# import plotly.graph_objects as go


# fig = go.Figure()
# fig.add_trace(go.Scatter(x=merged_final['metricset_timestamp'], y=merged_final['cpu_mean'], name='cpu_usage'))
# fig.update_layout(showlegend=True, title='cpu_usage of docker container')
# fig.show()