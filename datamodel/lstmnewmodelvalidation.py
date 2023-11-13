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
metric1 = pd.read_csv("metricbeat-230911-ai-broker-1.csv")
metric2 = pd.read_csv("metricbeat-230911-ai-broker-2.csv")
metric3 = pd.read_csv("metricbeat-230911-ai-broker-3.csv")
metric4 = pd.read_csv("metricbeat-230911-ai-broker-4.csv")
metric5 = pd.read_csv("metricbeat-230911-ai-broker-5.csv")




zipkin1 = pd.read_csv("zipkin-230911-all-broker.csv")



# print(zipkin.shape)

# print(metric1.head())
# print(zipkin.head())

metric_lst = [metric1,metric2,metric3,metric4,metric5]
# metric_merge = pd.merge(metric1,metric2, on='container_name')
metric_merge = pd.concat([metric1,metric2,metric3,metric4,metric5])
zipkin_merge = pd.concat([zipkin1])

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
merged_final['metricset_timestamp'] = pd.to_datetime(merged_final['metricset_timestamp'])


test =  merged_final.loc[merged_final['metricset_timestamp'] >= '2023-09-09']
scaler = StandardScaler()
scaler = scaler.fit(test[['cpu_mean']])

# train['cpu_mean'] = scaler.transform(train[['cpu_mean']])
test['cpu_mean'] = scaler.transform(test[['cpu_mean']])
new_model = tf.keras.models.load_model('/Users/e8l-20210032/Documents/GyubinHanAI/dataInference/lstmautoencodermodel/230911model_checkpoint_30.h5')
print(new_model)



TIME_STEPS=100

def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        Xs.append(X.iloc[i:(i+time_steps)].values)
        ys.append(y.iloc[i+time_steps])
    
    return np.array(Xs), np.array(ys)

X_test, y_test = create_sequences(test[['cpu_mean']], test['cpu_mean'])

# plt.plot(history.history['loss'], label='Training loss')
# plt.plot(history.history['val_loss'], label='Validation loss')
# plt.legend()

evaluation = new_model.evaluate(X_test, y_test)
# print(new_model.evaluate(X_test, y_test))


X_test_pred = new_model.predict(X_test, verbose =0)
test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
test_threshold = np.max(test_mae_loss) - 1.5
print(f'Reconstruction error test threshold: {test_threshold}')

# mse = np.mean(np.power(flatten(X_test) - flatten(X_test_pred), 2), axis=1)

# error_df = pd.DataFrame({'Reconstruction_error':mse, 
#                          'True_class':list(y_valid)})
# precision_rt, recall_rt, threshold_rt = metrics.precision_recall_curve(error_df['True_class'], error_df['Reconstruction_error'])


test_score_df = pd.DataFrame(test[TIME_STEPS:])
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = test_threshold
test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
test_score_df['cpu_mean'] = test[TIME_STEPS:]['cpu_mean']
print(test_score_df)
print(test_score_df['threshold'])



fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df['metricset_timestamp'], y=test_score_df['loss'], name='Test loss'))
fig.add_trace(go.Scatter(x=test_score_df['metricset_timestamp'], y=test_score_df['threshold'], name='Threshold'))
fig.update_layout(showlegend=True, title='Test loss vs. Threshold')
fig.show()