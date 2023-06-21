from elasticsearch import Elasticsearch, exceptions
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
        "timestamp":timestamp
    }
    
    return new_dict


_KEEP_ALIVE_LIMIT='20s'



# os environment

es = Elasticsearch('http://elastic:ndxpro123!@172.16.28.220:59200')
index = "metricbeat-7.17.0-2023.06.19-000001"

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
                            "container.name":"data-broker-2"
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

es_df = pd.DataFrame(columns=['container_name','cpu_usage','timestamp'])
es_lst = []
print("data insert start ")
start = time.time()

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
        
        
        
        # es_df.append(response['hits']['hits'][i]['_source'])
        
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
                            
            date = parser.parse(time_stamp)
            res_time = date + timedelta(hours=9)
            res_timestamp = round_seconds(res_time).strftime("%Y-%m-%dT%H:%M:%S")
            
            new = es_dict(container_name,cpu_usage,res_timestamp)
            # es_df = pd.concat([es_df, pd.DataFrame(es_dict)], ignore_index=True)
            # es_df.loc[len(es_df)] = list(es_dict.values())
            es_df.loc[len(es_df)] = list(es_dict(container_name,cpu_usage,res_timestamp).values())

            # es_df.append(response['hits']['hits'][i]['_source'])
            
except exceptions.ElasticsearchException as e:
    # Handle the exception
    print(f"An Elasticsearch error occurred: {e}")

print (es_df)
print(es_df.sort_values('timestamp', ascending=True))
print(len(es_df))

# print(type(es_df[0][1]))
# print(es_df[0][1])
print("finished in ", time.time() - start)