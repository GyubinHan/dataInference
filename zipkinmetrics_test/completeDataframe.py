import pandas as pd
import numpy as np
from datetime import datetime,timedelta


es_df = pd.read_csv("elastic-data-broker-1-5seconds.csv")
zipkin_df = pd.read_csv("zipkin-data-broker-1-5seconds.csv")
es_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
zipkin_df.drop(['Unnamed: 0'], axis = 1, inplace = True)


print(zipkin_df.info())
print(es_df.info())




