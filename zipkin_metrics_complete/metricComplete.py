from elasticsearch import Elasticsearch,exceptions
import elasticsearch
import time
import pandas as pd
from datetime import datetime, timedelta
import re
import psycopg2
import psycopg2.extras
import pandas as pd
from sqlalchemy import create_engine
import time
from dateutil import parser
import os
import logging


def round_seconds(obj: datetime) -> datetime:
    if obj.microsecond >= 500_000:
        obj += timedelta(seconds=1)
    return obj.replace(microsecond=0)

def es_dict(container_name,cpu_usage,timestamp):
    new_dict = {
        # "service_name":service_name,
        # "api_name":api_name,
        "container_name":container_name,
        "cpu_usage":cpu_usage,
        "metricset_timestamp":timestamp,
        "timestamp_5seconds":0
    }
    
    return new_dict



    
    
_KEEP_ALIVE_LIMIT='20s'


logging.basicConfig(filename='data_insert.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# os environment

es = Elasticsearch('http://elastic:ndxpro123!@172.16.28.222:59200')






# index = "metricbeat-7.17.0-2023.06.19-000001"
index = "metric*"
# now = datetime.now()
# now = now - timedelta(days = 1)
# now = now.strftime("%Y.%m.%d")

# index = index + now + '*'


query = {
            "query":{
                "bool":{
                    "must":[
                        {
                        "match":{
                            "metricset.name":"cpu"
                        }
                        },
                        {
                        "match":{
                            "container.name":"data-broker-ai-4"
                            # "container.name": CONTAINER_NAME
                            
                            # "localEndpoint.serviceName":SERVICE_NAME
                            
                        }
                        }
                    ]
                }
            # },
            }
            # "_source":[
            #     "duration",
            #     "localEndpoint.serviceName",
            #     "timestamp_millis",
            #     "name"
            # ]
            
}

response = es.search(index = index,
                   scroll = _KEEP_ALIVE_LIMIT,
                   size = 100,
                   body = query)

sid = response['_scroll_id']
fetched = len(response['hits']['hits'])

es_df = pd.DataFrame(columns=['container_name','cpu_usage','metricset_timestamp','timestamp_5seconds'])
es_lst = []
print("data insert start ")
start = time.time()
count = 0

try:
    for i in range(fetched):
        container_name = response['hits']['hits'][i]['_source']['container']['name']
        cpu_usage = response['hits']['hits'][i]['_source']['docker']["cpu"]["total"]["pct"]
        time_stamp = response['hits']['hits'][i]['_source']['@timestamp']
        
        date = parser.parse(time_stamp)
#     # print(date)
        res_time = date + timedelta(hours=9)
        res_timestamp = round_seconds(res_time).strftime("%Y-%m-%dT%H:%M:%S")
        
        # print(time_stamp, res_timestamp)
        # new = es_dict(container_name,cpu_usage,res_timestamp)
        es_df.loc[len(es_df)] = list(es_dict(container_name,cpu_usage,res_timestamp).values())
        
        count += 1
        print(count)
        
        # es_df = es_df.append(es_dict(response['hits']['hits'][i]['_source']["localEndpoint"]["serviceName"],response['hits']['hits'][i]['_source'],response['hits']['hits'][i]['_source'],response['hits']['hits'][i]['_source']))
    while(fetched>0):
        response = es.scroll(scroll_id=sid, scroll=_KEEP_ALIVE_LIMIT)
        fetched = len(response['hits']['hits'])
        for i in range(fetched):
            # es_df.append(response['hits']['hits'][i]['_source']['@timestamp'])
            # es_df.append([response['hits']['hits'][i]['_source']['docker']["cpu"]["total"]["pct"], response['hits']['hits'][i]['_source']['@timestamp']])
            container_name = response['hits']['hits'][i]['_source']['container']['name']
            cpu_usage = response['hits']['hits'][i]['_source']['docker']["cpu"]["total"]["pct"]
            time_stamp = response['hits']['hits'][i]['_source']['@timestamp']
            # print(type(time_stamp),time_stamp)                
            date = parser.parse(time_stamp)
            
            res_time = date + timedelta(hours=9)
            res_timestamp = round_seconds(res_time).strftime("%Y-%m-%dT%H:%M:%S")
            
            new = es_dict(container_name,cpu_usage,res_timestamp)
            # es_df = pd.concat([es_df, pd.DataFrame(es_dict)], ignore_index=True)
            # es_df.loc[len(es_df)] = list(es_dict.values())
            es_df.loc[len(es_df)] = list(es_dict(container_name,cpu_usage,res_timestamp).values())
            count += 1
            print(count)
            # es_df.append(response['hits']['hits'][i]['_source'])

except exceptions.ConnectionError as e:
    print("Connection error:", e)
    # Handle connection error here
    # For example, you can retry the operation or log the error.

except exceptions.RequestError as e:
    print("Request error:", e)
    # Handle request error here
    # For example, you can log the error or notify the user about the bad request.

except exceptions.NotFoundError as e:
    print("Index not found error:", e)
    # Handle index not found error here
    # For example, you can create the index or notify the user about the missing index.

except Exception as e:
    print("Unexpected error:", e)
    # Handle any other unexpected errors here
    # For example, you can log the error or take appropriate action.

es_df = es_df.sort_values('metricset_timestamp', ascending=True)

# print(type(es_df[0][1]))
# print(es_df[0][1])3z
print("finished in ", time.time() - start)



es_df = es_df.reset_index().rename(columns={'index': 'new_index'})

# # zipkin_df['timestamp_5seconds'] = 0


for idx, row in es_df.iterrows():
    timestamp = datetime.strptime(row['metricset_timestamp'],"%Y-%m-%dT%H:%M:%S")
#     seconds = datetime.strptime(row['timestamp'],"%S")
    sec = timestamp.strftime("%S")
    new_sec = int(sec)
    
#     print((new_sec%100//10)*10 + (new_sec%10)
    es_df.loc[idx]['timestamp_5seconds']
    
    
    
    # new_timestamp in catetory every 5 seconds
    if new_sec%10 < 5:
            # es_df.loc[es_df['timestamp'],"new_timestamp"] = timestamp - timedelta(seconds = new_sec%10)
            es_df.loc[idx:idx+1,'timestamp_5seconds'] = timestamp - timedelta(seconds = new_sec%10)
            # print((new_sec%100//10)*10) + (new_sec%10)
            # print(            timestamp - timedelta(seconds = new_sec%10))
        
    else:
        # es_df.loc[idx]['new_timestamp'] = timestamp - timedelta(seconds = 3)
        # es_df.loc[es_df['timestamp'],"new_timestamp"] = timestamp - timedelta(seconds = 3)
        # print(new_sec)
        es_df.loc[idx:idx+1,'timestamp_5seconds'] = timestamp - timedelta(seconds = new_sec%10 -5)
            # print(            timestamp - timedelta(seconds = 3))
        
        
print(es_df)

print("saving to csv")

es_df.to_csv("/Users/e8l-20210032/Documents/GyubinHanAI/dataInference/metricbeat-230901-ai-broker-4.csv",sep=',',na_rep='NaN')

print("CSV SAVING DONE")























############################## dbeaver connect
# conn = psycopg2.connect(
#     host="172.16.28.223", 
#     database="postgres",
#     user="postgres",
#     password="ndxpro123!"
# )

# Create a SQLAlchemy engine
# local
# engine = create_engine('postgresql+psycopg2://postgres:123123@localhost:5432/postgres')


# try:
#     conn = psycopg2.connect(
#         host="172.16.28.223",
#         database="postgres",
#         user="postgres",
#         password="ndxpro123!"
#     )
#     logging.info("Database connection established")
    
#     #local
#     # engine = create_engine('postgresql+psycopg2://postgres:123123@localhost:5432/postgres')
#     engine = create_engine('postgresql+psycopg2://postgres:ndxpro123!@172.16.28.223:55433/postgres')

#     schema_name = "datainferencemetricset"
#     # local
#     table_name = "databrokerservice"
#     #table_name = CONTAINER_NAME

#     # Convert the DataFrame to a PostgreSQL-compatible format using the 'to_sql' method
#     es_df.to_sql(schema=schema_name,name=table_name,con=engine, if_exists='replace', index=False) # table명 환경변수화 해야함

    
#     # Rest of your code ...
    
# except psycopg2.Error as e:
#     logging.error(f"Error connecting to the database: {e}")

