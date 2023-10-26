import os, random
# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


# Set seeds to make the experiment more reproducible.
def seed_everything(seed=28):
    random.seed(seed)
    np.random.seed(seed)
#     tf.random.set_seed(seed)
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
# print(merged_final.info())
merged_final.to_csv("/Users/e8l-20210032/Documents/GyubinHanAI/dataInference/anomalies_merged.csv")
# merged_final = merged_final.drop_duplicates(['traceId'],keep='first')

for idx, row in merged_final.iterrows():
    # if merged_final[idx]['traceId'] == merged_final['traceId'][3573344]:
    #     print(merged_final[idx])
    if row['traceId'] == merged_final['traceId'][3336554]:
        print(row)
    elif row['traceId'] == merged_final['traceId'][3336536]:
        print(row)
    elif row['traceId'] == merged_final['traceId'][3336518]:
        print(row)
    elif row['traceId'] == merged_final['traceId'][3336500]:
        print(row)
    elif row['traceId'] == merged_final['traceId'][3336482]:
        print(row)
    else:
        continue