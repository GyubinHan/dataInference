from elasticsearch import Elasticsearch, exceptions
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


def connect_db(host, dbname, user, password, port):
    return psycopg2.connect(host = host,dbname = dbname,user = user,password = password,port = port)
    
def select(conn, dbname):
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute(f"select * from {dbname}".format(dbname))
    rows = cur.fetchall()
    
    return rows

def docker_dict(header, row):
    result_dict = dict(zip(header,row))
    return result_dict

def elasticsearch_dict(service_name,api_name, timestamp, duration):
    
    new_dict = {
        "service_name": service_name,
        "api_name":api_name,
        "timestamp": datetime,
        "duration": duration
    }
    
    return new_dict


def insert(conn,schema,tablename,container,cpu,mem,datetime):
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    # cur.execute(f"insert into {schema}.{tablename}(container_name,cpu,mem,datetime) values ({container},{cpu},{mem},{datetime})"
    #             .format(schema,tablename,container,cpu,mem,datetime))
    insert_query = """ INSERT INTO datainferencedocker.tablename (container_name, cpu, mem, datetime) VALUES (%s,%s,%s,%s,%s,%s)"""
    record_to_insert = (container,cpu,mem,datetime)

    cur.execute(insert_query, record_to_insert)


def round_seconds(obj: datetime) -> datetime:
    if obj.microsecond >= 500_000:
        obj += timedelta(seconds=1)
    return obj.replace(microsecond=0)


# os.environment
# SERVICE_NAME = os.environ['SERVICE_NAME']
# from pandas.io.json import json_normalize
es = Elasticsearch('http://elastic:ndxpro123!@172.16.28.220:59200')
# es = Elasticsearch('http://elastic:ndxpro123!@123.37.5.152:59200')

print(es)

# index = "zipkin-span*"
index = "zipkin-span*"
_KEEP_ALIVE_LIMIT='20s'

query = {
            "query":{
                "bool":{
                    "must":[
                        {
                        "wildcard":{
                            "name":"*ndxpro/*"
                        }
                        },
                        {
                        "match":{
                            "localEndpoint.serviceName":"databroker-service"
                            # "localEndpoint.serviceName":SERVICE_NAME
                            
                        }
                        }
                    ]
                }
            },
            "_source":[
                "duration",
                "localEndpoint.serviceName",
                "timestamp_millis",
                "name"
            ]
            
}


response = es.search(index = index,
                   scroll = _KEEP_ALIVE_LIMIT,
                   size = 100,
                   body = query)

elastic_lst = []
# elastic search data ingesting

sid = response['_scroll_id']
fetched = len(response['hits']['hits'])
es_df = pd.DataFrame(columns=['service_name','api_name', 'timestamp', 'duration'])

print("data insert start ")


# for i in range(len(res['hits']['hits'])):
#     # response time
#     res_timestamp = res['hits']['hits'][i]["_source"]["timestamp_millis"]
#     res_datetime = datetime.fromtimestamp(res_timestamp/1000)
#     res_datetime = round_seconds(res_datetime).strftime("%Y-%m-%dT%H:%M:%S")
    
#     #servicename
#     res_service_name = res['hits']['hits'][i]["_source"]["localEndpoint"]["serviceName"]
#     # if res_service_name.find("manager"):
#         # res_service_name = re.sub("data","data",res_service_name)
#     res_service_name = re.sub("-service","service",res_service_name)
#     # print(res_service_name)
    
    
#     # duration
#     duration_time = res['hits']['hits'][i]["_source"]["duration"]
#     # print(type(duration_time)) # int
    
#     # print(duration_time/1000.0)
    

#     # api name 
#     api_name = res['hits']['hits'][i]["_source"]["name"]

#     # elasticsearch data ingesting
#     elastic_dict = elasticsearch_dict(res_service_name,api_name, res_datetime, duration_time/1000.0)   
#     elastic_lst.append(elastic_dict)


#### scrolling data
try:
    for i in range(fetched):
        # container_name = response['hits']['hits'][i]['_source']['container']['name']
        # cpu_usage = response['hits']['hits'][i]['_source']['docker']["cpu"]["total"]["pct"]
        # time_stamp = response['hits']['hits'][i]['_source']['@timestamp']
        
        
        
        res_timestamp = response['hits']['hits'][i]["_source"]["timestamp_millis"]
        res_datetime = datetime.fromtimestamp(res_timestamp/1000)
        res_datetime = round_seconds(res_datetime).strftime("%Y-%m-%dT%H:%M:%S")
        
        #servicename
        res_service_name = response['hits']['hits'][i]["_source"]["localEndpoint"]["serviceName"]
        # if res_service_name.find("manager"):
            # res_service_name = re.sub("data","data",res_service_name)
        res_service_name = re.sub("-service","service",res_service_name)
        # print(res_service_name)
        
        
        # duration
        duration_time = response['hits']['hits'][i]["_source"]["duration"]
        # print(type(duration_time)) # int
        
        # print(duration_time/1000.0)
        

        # api name 
        api_name = response['hits']['hits'][i]["_source"]["name"]
        
        # # data time adding 9 hours to make uct
        # date = parser.parse(time_stamp)
        # res_time = date + timedelta(hours=9)
        # res_timestamp = round_seconds(res_time).strftime("%Y-%m-%dT%H:%M:%S")
        
        # # print(time_stamp, res_timestamp)
        # # new = es_dict(container_name,cpu_usage,res_timestamp)
        es_df.loc[len(es_df)] = list(elasticsearch_dict(res_service_name,api_name, res_datetime, duration_time).values())
        
        
        
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
            
            # es_df = pd.concat([es_df, pd.DataFrame(es_dict)], ignore_index=True)
            # es_df.loc[len(es_df)] = list(es_dict.values())
            es_df.loc[len(es_df)] = list(elasticsearch_dict(res_service_name,api_name, res_datetime, duration_time).values())

            # es_df.append(response['hits']['hits'][i]['_source'])
            
except exceptions.ElasticsearchException as e:
    # Handle the exception
    print(f"An Elasticsearch error occurred: {e}")

print (es_df)
print(es_df.sort_values('timestamp', ascending=True))
print(len(es_df))