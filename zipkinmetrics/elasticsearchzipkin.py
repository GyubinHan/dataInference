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

def elasticsearch_dict(service_name,api_name, timestamp_millisecond, duration):
    
    new_dict = {
        "service_name": service_name,
        "api_name":api_name,
        "zipkin_timestamp": timestamp_millisecond,
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
zipkin_df = pd.DataFrame(columns=['service_name','api_name', 'timestamp', 'duration'])

print("data insert start ")

# print(response['hits']['hits'])

try:
    for i in range(fetched):
        service_name = response['hits']['hits'][i]['_source']['localEndpoint']['serviceName']
        api_name = response['hits']['hits'][i]['_source']['name']
        timestamp_millis = response['hits']['hits'][i]['_source']['timestamp_millis']
        duration = response['hits']['hits'][i]['_source']['duration']
        timestamp_millis = str(datetime.fromtimestamp(timestamp_millis/1000))

        # datetime to UTC
        date_timemilli = parser.parse(timestamp_millis)
        res_time = date_timemilli + timedelta(hours=9)
        res_timestamp = round_seconds(res_time).strftime("%Y-%m-%dT%H:%M:%S")
       
        # duration seconds 추출
        duration = datetime.fromtimestamp(duration/1000)
        duration = "{}.{}".format(duration.second, duration.microsecond)
            
        zipkin_df.loc[len(zipkin_df)] = list(elasticsearch_dict(service_name,api_name,res_timestamp,duration).values())
    
        
        
        # es_df = es_df.append(es_dict(response['hits']['hits'][i]['_source']["localEndpoint"]["serviceName"],response['hits']['hits'][i]['_source'],response['hits']['hits'][i]['_source'],response['hits']['hits'][i]['_source']))
    while(fetched>0):
        response = es.scroll(scroll_id=sid, scroll=_KEEP_ALIVE_LIMIT)
        fetched = len(response['hits']['hits'])
        for i in range(fetched):
            service_name = response['hits']['hits'][i]['_source']['localEndpoint']['serviceName']
            api_name = response['hits']['hits'][i]['_source']['name']
            timestamp_millis = response['hits']['hits'][i]['_source']['timestamp_millis']
            duration = response['hits']['hits'][i]['_source']['duration']
            timestamp_millis = str(datetime.fromtimestamp(timestamp_millis/1000))
            
            # duration seconds 추출
            duration = datetime.fromtimestamp(duration/1000)
            duration = "{}.{}".format(duration.second, duration.microsecond)
            # print(duration)
            
            
            # datetime to UTC
            date_timemilli = parser.parse(timestamp_millis)
            res_time = date_timemilli + timedelta(hours=9)
            res_timestamp = round_seconds(res_time).strftime("%Y-%m-%dT%H:%M:%S")
            
            # duration to datetime
            # date_duration = parser.parse(duration)
            # res_duration = date_duration.strftime("%ss")
            

            zipkin_df.loc[len(zipkin_df)] = list(elasticsearch_dict(service_name,api_name,res_timestamp,duration).values())

            # es_df.append(response['hits']['hits'][i]['_sour

except exceptions.ElasticsearchException as e:
    # Handle the exception
    print(f"An Elasticsearch error occurred: {e}")

print(zipkin_df)

print(zipkin_df.sort_values('timestamp', ascending=True))
print(len(zipkin_df))


print("saving to csv")

zipkin_df.to_csv("/Users/e8l-20210032/Documents/GyubinHanAI/dataInference/zipkin-data-broker-1-5seconds.csv",sep=',',na_rep='NaN')

print("CSV SAVING DONE")