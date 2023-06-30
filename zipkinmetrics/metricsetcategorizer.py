import pandas as pd
import numpy as np
from datetime import datetime,timedelta

es_df = pd.read_csv("elastic-data-broker-1-5seconds.csv")
# drop past number id 
es_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
es_df['new_timestamp'] = 0
for idx, row in es_df.iterrows():
    timestamp = datetime.strptime(row['timestamp'],"%Y-%m-%dT%H:%M:%S")
#     seconds = datetime.strptime(row['timestamp'],"%S")
    sec = timestamp.strftime("%S")
    new_sec = int(sec)
    
#     print((new_sec%100//10)*10 + (new_sec%10)
    es_df.loc[idx]['new_timestamp']
    
    
    
    # new_timestamp in catetory every 5 seconds
    if new_sec%10 < 5:
            # es_df.loc[es_df['timestamp'],"new_timestamp"] = timestamp - timedelta(seconds = new_sec%10)
            es_df.loc[idx:idx+1,'new_timestamp'] = timestamp - timedelta(seconds = new_sec%10)
            # print((new_sec%100//10)*10) + (new_sec%10)
            # print(            timestamp - timedelta(seconds = new_sec%10))
        
    else:
        # es_df.loc[idx]['new_timestamp'] = timestamp - timedelta(seconds = 3)
        # es_df.loc[es_df['timestamp'],"new_timestamp"] = timestamp - timedelta(seconds = 3)
        # print(new_sec)
        es_df.loc[idx:idx+1,'new_timestamp'] = timestamp - timedelta(seconds = new_sec%10 -5)
            # print(            timestamp - timedelta(seconds = 3))
        
        
        
        
        
#     print(new_sec, (new_sec%100//10)*10 , new_sec%10)
#     print(timestamp,datetime.strptime(sec,"%S"))
#     es_df['categorical_column'] = pd.cut(es_df['time_diff_seconds'], bins=[0, 5, 10,15,20,25,30,35,40,45,50,55],
#                                          labels=['0~5','5~10','10~15','15~20','20~25','25~30',
#                                                 '30~35','35~40','40~45','45~50','50~55'])
        

print(es_df)
        
