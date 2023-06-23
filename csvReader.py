import pandas as pd
import numpy as np

es_df = pd.read_csv("elastic-data-broker-1-5seconds.csv")
# drop past number id 
es_df.drop(['Unnamed: 0'], axis = 1, inplace = True)

print(type(es_df[:100]['timestamp'][:0]))

