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

def elasticsearch_dict(service_name,api_name, timestamp_millisecond, duration,traceid):
    
    new_dict = {
        "service_name": service_name,
        "api_name":api_name,
        "zipkin_timestamp": timestamp_millisecond,
        "duration": duration,
        "traceId": traceid,
        "timestamp_5seconds":0
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
es = Elasticsearch('http://elastic:ndxpro123!@172.16.28.223:59200')
# es = Elasticsearch('http://elastic:ndxpro123!@123.37.5.152:59200')

print(es)

# index = "zipkin-span*

now = datetime.now()
now = now - timedelta(days = 1)
now = now.strftime("%Y-%m-%d")
index = "zipkin-span-"
index = index + now + '*'
print(index)
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
                "name",
                "traceId"
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
zipkin_df = pd.DataFrame(columns=['service_name','api_name', 'zipkin_timestamp', 'duration','traceId','timestamp_5seconds'])

print("data insert start ")

# print(response['hits']['hits'])

try:
    for i in range(fetched):
        service_name = response['hits']['hits'][i]['_source']['localEndpoint']['serviceName']
        api_name = response['hits']['hits'][i]['_source']['name']
        timestamp_millis = response['hits']['hits'][i]['_source']['timestamp_millis']
        duration = response['hits']['hits'][i]['_source']['duration']
        traceid = response['hits']['hits'][i]['_source']['traceId']
        
        timestamp_millis = str(datetime.fromtimestamp(timestamp_millis/1000))

        # datetime to UTC
        date_timemilli = parser.parse(timestamp_millis)
        res_time = date_timemilli + timedelta(hours=9)
        res_timestamp = round_seconds(res_time).strftime("%Y-%m-%dT%H:%M:%S")
       
        # duration seconds 추출
        duration = datetime.fromtimestamp(duration/1000)
        duration = "{}.{}".format(duration.second, duration.microsecond)
            
        zipkin_df.loc[len(zipkin_df)] = list(elasticsearch_dict(service_name,api_name,res_timestamp,duration,traceid).values())
    
        
        
        # es_df = es_df.append(es_dict(response['hits']['hits'][i]['_source']["localEndpoint"]["serviceName"],response['hits']['hits'][i]['_source'],response['hits']['hits'][i]['_source'],response['hits']['hits'][i]['_source']))
    while(fetched>0):
        response = es.scroll(scroll_id=sid, scroll=_KEEP_ALIVE_LIMIT)
        fetched = len(response['hits']['hits'])
        for i in range(fetched):
            service_name = response['hits']['hits'][i]['_source']['localEndpoint']['serviceName']
            api_name = response['hits']['hits'][i]['_source']['name']
            timestamp_millis = response['hits']['hits'][i]['_source']['timestamp_millis']
            duration = response['hits']['hits'][i]['_source']['duration']
            traceid = response['hits']['hits'][i]['_source']['traceId']
            
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
            

            zipkin_df.loc[len(zipkin_df)] = list(elasticsearch_dict(service_name,api_name,res_timestamp,duration,traceid).values())

            # es_df.append(response['hits']['hits'][i]['_sour

except exceptions.ElasticsearchException as e:
    # Handle the exception
    print(f"An Elasticsearch error occurred: {e}")

zipkin_df = zipkin_df.sort_values('zipkin_timestamp', ascending=True)
print(zipkin_df)
print(len(zipkin_df))

# print(zipkin_df.info())

zipkin_df = zipkin_df.reset_index().rename(columns={'index': 'new_index'})
print(zipkin_df)

# # zipkin_df['timestamp_5seconds'] = 0


for idx, row in zipkin_df.iterrows():
    timestamp = datetime.strptime(row['zipkin_timestamp'],"%Y-%m-%dT%H:%M:%S")
#     seconds = datetime.strptime(row['timestamp'],"%S")
    sec = timestamp.strftime("%S")
    new_sec = int(sec)
    
#     print((new_sec%100//10)*10 + (new_sec%10)
    zipkin_df.loc[idx]['timestamp_5seconds']
    
    
    
    # new_timestamp in catetory every 5 seconds
    if new_sec%10 < 5:
            # es_df.loc[es_df['timestamp'],"new_timestamp"] = timestamp - timedelta(seconds = new_sec%10)
            zipkin_df.loc[idx:idx+1,'timestamp_5seconds'] = timestamp - timedelta(seconds = new_sec%10)
            # print((new_sec%100//10)*10) + (new_sec%10)
            # print(            timestamp - timedelta(seconds = new_sec%10))
        
    else:
        # es_df.loc[idx]['new_timestamp'] = timestamp - timedelta(seconds = 3)
        # es_df.loc[es_df['timestamp'],"new_timestamp"] = timestamp - timedelta(seconds = 3)
        # print(new_sec)
        zipkin_df.loc[idx:idx+1,'timestamp_5seconds'] = timestamp - timedelta(seconds = new_sec%10 -5)
            # print(            timestamp - timedelta(seconds = 3))
        
        
        

conn = psycopg2.connect(
    host="172.16.28.223",
    database="postgres",
    user="postgres",
    password="ndxpro123!"
)

# Create a SQLAlchemy engine
# local
# engine = create_engine('postgresql+psycopg2://postgres:123123@localhost:5432/postgres')

# docker
engine = create_engine('postgresql+psycopg2://postgres:ndxpro123!@172.16.28.223:55433/postgres')

schema_name = "datainferencezipkin"
# local
table_name = "databrokerservice"
#table_name = CONTAINER_NAME

# Convert the DataFrame to a PostgreSQL-compatible format using the 'to_sql' method
zipkin_df.to_sql(schema=schema_name,name=table_name,con=engine, if_exists='replace', index=False) # table명 환경변수화 해야함




# print("saving to csv")

# zipkin_df.to_csv("/Users/e8l-20210032/Documents/GyubinHanAI/dataInference/zipkin223-data-broker-1-2023-06-28.csv",sep=',',na_rep='NaN')

# print("CSV SAVING DONE")