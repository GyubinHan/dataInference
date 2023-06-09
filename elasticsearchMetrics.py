from elasticsearch import Elasticsearch
import pandas as pd 
from datetime import datetime, timedelta
import re
import psycopg2
import psycopg2.extras
import pandas as pd
from sqlalchemy import create_engine
from dateutil import parser
import time
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

def elasticsearch_dict(service_name,api_name, datetime, duration):
    
    new_dict = {
        "service_name": service_name,
        "api_name":api_name,
        "response_time": datetime,
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
# CONTAINER_NAME = os.environ['CONTAINER_NAME']

# from pandas.io.json import json_normalize
es = Elasticsearch('http://elastic:ndxpro123!@172.16.28.220:59200')
# es = Elasticsearch('http://elastic:ndxpro123!@123.37.5.152:59200')

print(es)

# index = "zipkin-span*"
index = "metricbeat-7.17.0*"

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

res = es.search(index=index,body = query, scroll= '2m', size=10000)

elastic_lst = []
# elastic search data ingesting
count = 0 
# Start scrolling through the results
scroll_id = res['_scroll_id']
total_docs = res['hits']['total']['value']
scroll_size = len(res['hits']['hits'])

# while scroll_size > 0:
#     # Get the next page of results
#     response = es.scroll(
#         scroll_id=scroll_id,
#         scroll="2m"
#     )
#     new_data = pd.DataFrame()
#     # Process the current page of results
for hit in res['hits']['hits']:
    # Do something with the document (hit['_source'])
    print(hit['_source'])
    # new_data.append(pd.DataFrame(hit["_source"]))
    count += 1

# Update the scroll ID and scroll size for the next iteration
# scroll_id = res['_scroll_id']
scroll_size = len(res['hits']['hits'])

# Clear the scroll
es.clear_scroll(
    scroll_id=scroll_id
)

print("Count",count)
# print(new_data.head())




# for i in range(len(res['hits']['hits'])):
#     # print(res['hits']['hits'][i])
    
#     # container_name
#     res_container_name = res['hits']['hits'][i]['_source']["container"]["name"]
    
#     # response_time
#     res_time = res['hits']['hits'][i]['_source']["@timestamp"]
#     date = parser.parse(res_time)
#     # print(date)
#     res_time = date + timedelta(hours=9)
#     print(res_time.strftime('%Y-%m-%d %H:%M:%S' ))
#     # print(datetime.strptime(res_time, '%Y-%m-%d %H:%M:%S'))
#     # print(datetime.strptime(res_time))
# print(len(res['hits']['hits']))
    # print(res_container_name, res_time)
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



elastic_df = pd.DataFrame(elastic_lst)
print(elastic_df)

# for index, row in elastic_df.iterrows():
#     print(row['response_time'])


try:
        elastic_conn = connect_db("localhost",'postgres','postgres','123123',5432)
        
        cur = elastic_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # gateway_header = ['service_id','api_id','request_time','response_time']
        # logger.info(f"{container_name}: DB connected successfully,  Time: {datetime.now()}")
        
except psycopg2.DatabaseError as db_err:
    print(db_err)
    
    
# for index, row in elastic_df.iterrows():
#         if row["service_name"] == "databrokerservice":
#             insert_query = """ INSERT INTO datainferenceelastic.databrokerservice (service_name, api_name, response_time, duration) VALUES (%s,%s,%s,%s)"""
#             record_to_insert = (row["service_name"],row["api_name"],row["response_time"],row["duration"])
#             cur.execute(insert_query, record_to_insert)
#             elastic_conn.commit()





# count = 1
# start = time.time()
# for index, row in elastic_df.iterrows(): 
    
#     # table_name = row['service_name']
#     # print(table_name)
#     # print(row)
    
    
    
#     schema_name = 'datainferenceelastic'
#     table_name = row['service_name']
#     full_table_name = f'{schema_name}.{table_name}'

#     # Connect to the PostgreSQL database
    
#     print(elastic_df)
#     print(type(row))
#     # Create a SQLAlchemy engine
#     # engine = create_engine('postgresql+psycopg2://postgres:123123@localhost:5432/postgres')

#     # cur.execute(f"INSERT INTO {full_table_name} (service_name,api_name,response_time,duration) VALUES ({row['service_name']},{row['api_name']},{row['response_time']},{row['duration']})")
#     # conn.commit()
#     # row.to_sql(full_table_name,engine, if_exists='append', index = False)
#     # print(row['service_name'], count)
#     # count += 1


# print("time cost: ",time.time()-start)
# cur.close()
# conn.close()
    
# postgres_data = []
# postgres_conn = connect_db("localhost",'postgres','postgres','123123',5432)

# postgres_rows = insert(postgres_conn,"datainferencedocker",res_service_name,res_service_name,res_datetime,duration_time/1000.0)


