import pandas as pd
import numpy as np
from datetime import datetime,timedelta

es_str = "metricbeat-230719-ai-broker-"
es_df1= pd.read_csv(es_str + str(1) + ".csv")
es_df2 = pd.read_csv(es_str + str(2) + ".csv")
es_df3 = pd.read_csv(es_str + str(3) + ".csv")
es_df4 = pd.read_csv(es_str + str(4) + ".csv")
es_df5 = pd.read_csv(es_str + str(5) + ".csv")

es_df1.drop(['Unnamed: 0'], axis = 1, inplace = True)
es_df1.drop(['new_index'], axis = 1, inplace = True)

# print(es_df1)

es_df2.drop(['Unnamed: 0'], axis = 1, inplace = True)
es_df2.drop(['new_index'], axis = 1, inplace = True)
# print(es_df2)
es_df3.drop(['Unnamed: 0'], axis = 1, inplace = True)
es_df3.drop(['new_index'], axis = 1, inplace = True)
# print(es_df3)
es_df4.drop(['Unnamed: 0'], axis = 1, inplace = True)
es_df4.drop(['new_index'], axis = 1, inplace = True)
# print(es_df4)
es_df5.drop(['Unnamed: 0'], axis = 1, inplace = True)
es_df5.drop(['new_index'], axis = 1, inplace = True)
# print(es_df5)


es_df = pd.concat([es_df1,es_df2,es_df3,es_df4,es_df5])


zipkin_df = pd.read_csv("zipkin-230720-ai-brokerservice.csv")
zipkin_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
zipkin_df.drop(['new_index'], axis = 1, inplace = True)


merged = pd.merge(es_df,zipkin_df,on=['timestamp_5seconds'])

merged['cache'] = ""

merged2 = merged.copy()[:]

print(merged2)
print(merged2.info())
count = 0



for idx, row in merged2.iterrows():
    if row['cpu_usage'] > 0.03 or row['duration'] > 15:
        # print(row)
        count += 1
        merged2.loc[idx,'cache'] = 'yes'
        
    else:
        merged2.loc[idx,'cache'] = 'no'
        continue
    # print(merged.loc[idx])
cache_count = 0
for idx, row in merged2.iterrows():
    if row['cache'] == 'yes':
        cache_count += 1
    else:
        continue
    
print(cache_count)

# print("saving to csv")

# zipkin_df.to_csv("/Users/e8l-20210032/Documents/GyubinHanAI/dataInference/zipkin-230720-ai-broker-cache.csv",sep=',',na_rep='NaN')

# print("CSV SAVING DONE")




# for i in range(2,6):
#     es_str_new = es_str + str(i) + ".csv"
#     es_df_new = pd.read_csv(es_str_new)
#     es_df_new.drop(['Unnamed: 0'], axis = 1, inplace = True)
#     print(es_df_new)
    
#     if i == 2:
#         merged = pd.merge(es_df,es_df_new,on=['container_name'])
        
#     else:
#         merged = pd.merge(merged,es_df_new,on=['container_name'])

# print(merged)
# print(len(merged))
# print(merged.shape)
# es_df = pd.read_csv("metricbeat-230719-ai-broker-1.csv")
# zipkin_df = pd.read_csv("zipkin-230720-ai-brokerservice.csv")
# es_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
# zipkin_df.drop(['Unnamed: 0'], axis = 1, inplace = True)


# print(zipkin_df.info())
# print(es_df.info())




# merged = pd.merge(es_df,zipkin_df,on=['timestamp_5seconds'])
# print(merged)
# print(merged.info())