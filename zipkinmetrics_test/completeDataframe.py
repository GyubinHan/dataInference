import pandas as pd
import numpy as np
from datetime import datetime,timedelta


es_df = pd.read_csv("metricbeat-230711-all-broker-1.csv")
zipkin_df = pd.read_csv("zipkin-230711-all-broker.csv")
es_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
zipkin_df.drop(['Unnamed: 0'], axis = 1, inplace = True)


print(zipkin_df.info())
print(es_df.info())




merged = pd.merge(es_df,zipkin_df,on=['timestamp_5seconds'])
print(merged)
print(merged.info())