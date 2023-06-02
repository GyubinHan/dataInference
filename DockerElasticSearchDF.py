from elasticsearch import Elasticsearch
import pandas as pd 
from datetime import datetime
import re
import psycopg2
import psycopg2.extras
import pandas as pd

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



# from pandas.io.json import json_normalize
es = Elasticsearch('http://elastic:ndxpro123!@172.16.28.220:59200')
print(es)

# index = "zipkin-span*"
index = "zipkin-span-2023-05*"

query = {"query": {

    "wildcard": {

        "name": "*ndxpro/*"
    }
  },

   "_source": ["duration", "localEndpoint.serviceName","timestamp_millis","name"]
}

res = es.search(index=index,body = query, size=1000)

elastic_lst = []
# elastic search data ingesting

for i in range(len(res['hits']['hits'])):
    # response time
    res_timestamp = res['hits']['hits'][i]["_source"]["timestamp_millis"]
    res_datetime = datetime.fromtimestamp(res_timestamp/1000)
    
    #servicename
    res_service_name = res['hits']['hits'][i]["_source"]["localEndpoint"]["serviceName"]
    res_service_name = re.sub("data","data-",res_service_name)
    res_service_name = re.sub("-service","",res_service_name)
    # print(res_service_name)
    
    
    # duration
    duration_time = res['hits']['hits'][i]["_source"]["duration"]
    # print(type(duration_time)) # int
    
    # print(duration_time/1000.0)
    

    # api name 
    api_name = res['hits']['hits'][i]["_source"]["name"]

    # elasticsearch data ingesting
    elastic_dict = elasticsearch_dict(res_service_name,api_name, res_datetime, duration_time/1000.0)   
    elastic_lst.append(elastic_dict)
    
elastic_df = pd.DataFrame(elastic_lst)
print(elastic_df['response_time'].sort_values())
    # df_dict = pd.DataFrame(new_dict)
    # print(df_dict)
    # print(new_dict['service_name'], new_dict["response_time"])





# ## docker data ingesting

# try:
#     postgres_data = []
#     postgress_conn = connect_db("localhost",'postgres','postgres','123123',5432)
    
#     postgres_rows = select(postgress_conn,"datainference.dockerstatus")
#     # postgres_rows = select(postgress_conn,"datainference.dockerstatus")
    
    
#     postgres_header = ['container_name','CPU','Mem','datetime']
    
#     for row in postgres_rows:
#         postgres_data.append(docker_dict(postgres_header,row))
    
#     final_data = pd.DataFrame(postgres_data)
    
    
#     ## 1초의 시간 차이 계산
#     print(postgres_data[0]['datetime']-elastic_dict['response_time'])


#     for i in range(len(postgres_data)):
#         for k, v in postgres_data[i].items():
#             if k == 'service_id':
#                 pass

#             else:
#                 final_data[k] = v
                
#         for index, row in final_data.iterrows():
#         # if final_data[i]['api_id'] == final_data[i]['container_id']:
#         #     print(final_data[i])
        
#             if row['containerID'] == row['api_id']:
#                 print(row)
    
#     # gateway_data.append(dict_maker(header,gateway_rows))
#     lst = [i[0].split(',') for i in postgres_header]
#     df = []
#     resource_dict = {}
    
#     for row in gateway_row:
#         result_dict = dict(zip(postgres_header,row))
#         df.append(result_dict)
        
#         df.append(dict)
#     print([dict(zip(lst[0], v)) for v in lst[1:]])
    
#     print(postgres_data)
# except psycopg2.DatabaseError as db_err:
#     print(db_err)