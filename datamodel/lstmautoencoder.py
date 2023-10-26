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
from keras import optimizers

import matplotlib.pyplot as plt
import plotly.graph_objects as go


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
# print(merged_final)
merged_final = merged_final.drop_duplicates(['traceId'],keep='first')


merged_final['metricset_timestamp'] = pd.to_datetime(merged_final['metricset_timestamp'])
merged_final['metricset_timestamp'].min(), merged_final['metricset_timestamp'].max()

train, test = merged_final.loc[merged_final['metricset_timestamp'] <= '2023-08-05'], merged_final.loc[merged_final['metricset_timestamp'] > '2023-08-07']
# print(train.shape, test.shape)


TIME_STEPS=100

def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        Xs.append(X.iloc[i:(i+time_steps)].values)
        ys.append(y.iloc[i+time_steps])
    
    return np.array(Xs), np.array(ys)

X_train, y_train = create_sequences(train[['cpu_mean']], train['cpu_mean'])
X_test, y_test = create_sequences(test[['cpu_mean']], test['cpu_mean'])




valid_1 = pd.read_csv("metricbeat-230911-ai-broker-1.csv")
valid_2 = pd.read_csv("metricbeat-230911-ai-broker-2.csv")
valid_3 = pd.read_csv("metricbeat-230911-ai-broker-3.csv")
valid_4 = pd.read_csv("metricbeat-230911-ai-broker-4.csv")
valid_5 = pd.read_csv("metricbeat-230911-ai-broker-5.csv")
valid_zip = pd.read_csv("zipkin-230911-all-broker.csv")


metric_lst = [valid_1,valid_2,valid_3,valid_4,valid_5]
# metric_merge = pd.merge(metric1,metric2, on='container_name')
metric_merge_valid = pd.concat([valid_1,valid_2,valid_3,valid_4,valid_5])
zipkin_merge_valid = pd.concat([valid_zip])

metric_merge_valid.drop(['Unnamed: 0'], axis = 1, inplace = True)
metric_merge_valid.drop(['new_index'], axis = 1, inplace = True)
zipkin_merge_valid.drop(['Unnamed: 0'], axis = 1, inplace = True)
zipkin_merge_valid.drop(['new_index'], axis = 1, inplace = True)
zipkin_merge_valid.drop_duplicates(['traceId'],keep='first')

##### data merge
merged = pd.merge(metric_merge_valid, zipkin_merge_valid,left_on='timestamp_5seconds', right_on='timestamp_5seconds',how='right')
merged = merged.dropna()


merged_grouped = merged.groupby(merged['traceId'])
merged_grouped = merged_grouped['cpu_usage'].mean()
merged_grouped = merged_grouped.reset_index()
merged_grouped['cpu_mean'] = merged_grouped['cpu_usage']
merged_final = pd.merge(merged,merged_grouped,on='traceId')
merged_final.drop(['cpu_usage_y'],inplace=True, axis=1)
# print(merged_final)
merged_final = merged_final.drop_duplicates(['traceId'],keep='first')


merged_final['metricset_timestamp'] = pd.to_datetime(merged_final['metricset_timestamp'])
merged_final['metricset_timestamp'].min(), merged_final['metricset_timestamp'].max()

train, test = merged_final.loc[merged_final['metricset_timestamp'] <= '2023-08-20'], merged_final.loc[merged_final['metricset_timestamp'] > '2023-09-11']
# print(train.shape, test.shape)


TIME_STEPS=100

def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        Xs.append(X.iloc[i:(i+time_steps)].values)
        ys.append(y.iloc[i+time_steps])
    
    return np.array(Xs), np.array(ys)

x_valid, y_valid = create_sequences(train[['cpu_mean']], train['cpu_mean'])
X_test_valid, y_test_valid = create_sequences(test[['cpu_mean']], test['cpu_mean'])



# print(f'Training shape: {X_train.shape}')
# print(f'Training shape: {y_train.shape}')

# print(f'Testing shape: {X_test.shape}')
# print(f'Testing shape: {y_test.shape}')

# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow import keras
tf.random.set_seed(28)

model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(rate=0.4))
model.add(RepeatVector(X_train.shape[1]))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(rate=0.4))
model.add(TimeDistributed(Dense(X_train.shape[2])))
adam = optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=adam, loss='mae', metrics=['mse'])
model.summary()

# model save callbacks
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    filepath="/Users/e8l-20210032/Documents/GyubinHanAI/dataInference/lstmautoencodermodel/230926model_checkpoint_{epoch:02d}.h5",
    save_best_only=False,
    save_weights_only=False,
    period=5  # Save every 10 epochs
)

print("new model loading")
# history = model.fit(X_train, y_train, validation_data= (x_valid,y_valid), epochs=35, batch_size=32, validation_split=0.1, callbacks=[checkpoint_callback], shuffle=False)
new_model = tf.keras.models.load_model('/Users/e8l-20210032/Documents/GyubinHanAI/dataInference/lstmautoencodermodel/230914model_checkpoint_35.h5')
# new_model2 = model.load_weights('/Users/e8l-20210032/Documents/GyubinHanAI/dataInference/lstmautoencodermodel/230914model_checkpoint_35.h5')
print("new model loading done")

# evaluate the model
# _, train_mse = history.evaluate(X_train, y_train, verbose=0)
# _, test_mse = history.evaluate(X_test, y_test, verbose=0)# print(loss, accuracy)
# print("train_mse 1",train_mse)
# print("test_mse 1",test_mse)



# evaluate the model


# new_model.evaluate(X_train,y_train, verbose =0)

# plt.plot(new_model['loss'], label='Training loss')
# plt.plot(new_model['val_loss'], label='Validation loss')
# plt.legend()
# plt.show()



print("new model training")

history = new_model.fit(X_train, y_train, validation_data= (x_valid,y_valid), epochs=185, batch_size=32, validation_split=0.1, callbacks=[checkpoint_callback], shuffle=False)

new_model.save("/Users/e8l-20210032/Documents/GyubinHanAI/dataInference/lstmautoencodermodel/230926model_checkpoint_188.keras")

print("new model evaluating")
# new_model.evaluate(X_train, y_train, verbose=0)
_, train_mse = new_model.evaluate(X_train, y_train)
# _, train_mse = new_model.evaluate(X_train, y_train, verbose=0)
_, test_mse = new_model.evaluate(X_test, y_test)# print(loss, accuracy)

# _, test_mse = history.evaluate(X_test, y_test, verbose=0)# print(loss, accuracy)



# plot cpu_usage
# import plotly.graph_objects as go


# fig = go.Figure()
# fig.add_trace(go.Scatter(x=merged_final['metricset_timestamp'], y=merged_final['cpu_mean'], name='cpu_usage'))
# fig.update_layout(showlegend=True, title='cpu_usage of docker container')
# fig.show()



X_test_pred = new_model.predict(X_test)
print(X_test_pred)
print(len(X_test_pred))
test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
test_threshold = np.max(test_mae_loss)
print(f'Reconstruction error test threshold: {test_threshold}')

# mse = np.mean(np.power(flatten(X_test) - flatten(X_test_pred), 2), axis=1)

# error_df = pd.DataFrame({'Reconstruction_error':mse, 
#                          'True_class':list(y_valid)})
# precision_rt, recall_rt, threshold_rt = metrics.precision_recall_curve(error_df['True_class'], error_df['Reconstruction_error'])


# test_score_df = pd.DataFrame(test[1000:])
# test_score_df['loss'] = test_mae_loss
# test_score_df['threshold'] = test_threshold
# test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
# test_score_df['cpu_mean'] = test[TIME_STEPS:]['cpu_mean']
# print(test_score_df)
# print(test_score_df['threshold'])



# fig = go.Figure()
# fig.add_trace(go.Scatter(x=test_score_df['metricset_timestamp'], y=test_score_df['loss'], name='Test loss'))
# fig.add_trace(go.Scatter(x=test_score_df['metricset_timestamp'], y=test_score_df['threshold'], name='Threshold'))
# fig.update_layout(showlegend=True, title='Test loss vs. Threshold')
# fig.show()