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


new_model = tf.keras.models.load_model('/Users/e8l-20210032/Documents/GyubinHanAI/dataInference/lstmautoencodermodel/model_checkpoint_99.h5')
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend();


X_train_pred = model.predict(X_train, verbose=0)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
threshold = np.max(train_mae_loss)

print(f'Reconstruction error threshold: {threshold}')

test_score_df = pd.DataFrame(test[TIME_STEPS:])
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = threshold
test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
test_score_df['cpu_mean'] = test[TIME_STEPS:]['cpu_mean']



# fig = go.Figure()
# fig.add_trace(go.Scatter(x=test_score_df['metricset_timestamp'], y=test_score_df['loss'], name='Test loss'))
# fig.add_trace(go.Scatter(x=test_score_df['metricset_timestamp'], y=test_score_df['threshold'], name='Threshold'))
# fig.update_layout(showlegend=True, title='Test loss vs. Threshold')
# fig.show()


anomalies = test_score_df.loc[test_score_df['anomaly'] == True]
anomalies.shape