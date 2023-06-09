import requests
import os
import paramiko
import getpass
import time
import json
from json import dumps, loads
from datetime import datetime, timedelta
import psycopg2
import psycopg2.extras
import logging



def round_seconds(dt):
    seconds = (dt - dt.min).seconds
    rounding = (seconds + 5) // 10 * 10
    rounded_dt = dt + timedelta(0, rounding - seconds)
    return rounded_dt

def round_seconds2(obj: datetime) -> datetime:
    if obj.microsecond >= 500_000:
        obj += timedelta(seconds=1)
    return obj.replace(microsecond=0)

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

def insert(conn,schema,tablename,container,cpu,mem,datetime):
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    # cur.execute(f"insert into {schema}.{tablename}(container_name,cpu,mem,datetime) values ({container},{cpu},{mem},{datetime})"
    #             .format(schema,tablename,container,cpu,mem,datetime))
    insert_query = """ INSERT INTO schema.tablename (container_name, cpu, mem, datetime) VALUES (%s,%s,%s,%s,%s)"""
    record_to_insert = (container,cpu,mem,datetime)

    cur.execute(insert_query, record_to_insert)




# os environment
# container_name = os.environ['CONTAINER_NAME']
# docker_log = os.environ['DOCKER_LOG']
container_name = "data-broker-3"
# container_name = "data-broker-2"



# logging 

logging.basicConfig(
            format='%(asctime)s %(levelname)s %(message)s', 
            level=logging.INFO, 
            datefmt='%m/%d/%Y %I:%M:%S %p'
            )
logger = logging.getLogger('ndxpro-datainference-docker-datainsert')


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# f1 = logging.FileHandler(filename = docker_log)
# f1.setLevel(logging.INFO)
# f1.setFormatter(formatter)
# logger.addHandler(f1)




# print(table)

# table = table.replace(container_name[3:],"broker")
# print(table)



 # ssh  접속 
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)

# 220 서버
server = "172.16.28.220"
user = "root"
# pwd = getpass.getpass("Password: ")
cli.connect(server, username=user, password="!!ndxpro123!!220")


stdin, stdout, stderr = cli.exec_command("ls -la")
lines = stdout.readlines()


# 새로운 interactive shell session 생성
channel = cli.invoke_shell()

# container_name = "dat"

logger.setLevel(level=logging.INFO)
# scheme = "datainferencedocker"
# container_name = "data-broker-1"

# table명 정리
table = container_name.replace("data-", "data")
table = table.replace(table[4:],"brokerservice")
# postgres_data = []

# postgres_rows = select(postgress_conn,"postgres")
# postgres_conn = connect_db("localhost",'postgres','postgres','123123',5432)


postgres_header = ['containerID','CPU','Mem','datetime']


while True:
    try:
        postgres_conn = connect_db("host.docker.internal",'postgres','postgres','123123',5432)
        
        cur = postgres_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # gateway_header = ['service_id','api_id','request_time','response_time']
        logger.info(f"{container_name}: DB connected successfully,  Time: {datetime.now()}")
                    
    except psycopg2.DatabaseError as db_err:
        print(db_err)

    logger.info(f"{container_name}: docker status ingest starting,  Time: {datetime.now()}")

    send_message = f"curl --unix-socket /var/run/docker.sock http://localhost/v1.41/containers/{container_name}/stats?stream=true\n"
    # print(send_message)
    channel.send(send_message)
    time.sleep(0.9)
    # 결과 수신
    output = channel.recv(65535).decode("UTF-8").replace(";",'"')

    output_lst = output.splitlines(0)
    # now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # new_now = f'{now}'
    now = datetime.now()
    rounded_now = str(round_seconds2(now).strftime("%Y-%m-%dT%H:%M:%S"))
    try:
        resource = json.loads(output_lst[len(output_lst)-1].replace("'", "\""),strict=False)

            # Process the JSON data
    except FileNotFoundError:
        print("File not found.")
    except json.JSONDecodeError as e:
        print("JSON decoding error:", str(e))
    # print(resource)
    # print(type(resource))
    # print(string_to_dict['cpu_stats']['cpu_usage']['total_usage'])
    
    # memory usage percentage
    used_memory =  resource['memory_stats']['usage'] - resource['memory_stats']['stats']['cache']
    available_memory = resource['memory_stats']['limit']
    memory_usage = (used_memory/available_memory) * 100.0
    
    # cpu usage percentage
    cpu_delta = resource['cpu_stats']['cpu_usage']['total_usage'] - resource['precpu_stats']['cpu_usage']['total_usage']
    num_cpus = resource['cpu_stats']['online_cpus']
    if len(resource['precpu_stats']) < 4:
        system_cpu_delta = resource['cpu_stats']['system_cpu_usage'] - 0
    else: 
        system_cpu_delta = resource['cpu_stats']['system_cpu_usage'] - resource['precpu_stats']['system_cpu_usage']

    cpu_usage = (cpu_delta/system_cpu_delta) * num_cpus *100.0
    
    postgres_conn = connect_db("localhost",'postgres','postgres','123123',5432)

    docker_dict ={"container_name":container_name,
                  "cpu": cpu_usage,
                  "mem": memory_usage,
                  "datetime": now}
    insert_query = """ INSERT INTO datainferencedocker.databrokerservice (container_name, cpu, memory, datetime) VALUES (%s,%s,%s,%s)"""
    record_to_insert = (container_name,cpu_usage,memory_usage,rounded_now)
    cur.execute(insert_query, record_to_insert)
    
    # insert_query = """ INSERT INTO %s.%s (container_name, cpu, mem, datetime) VALUES (%s,%s,%s,%s)"""
    # insert_q2 = f'INSERT INTO {scheme}.{table} (container_name, cpu, mem, datetime) VALUES ({container_name},{cpu_usage},{memory_usage},{new_now} )'
    
    # # record_to_insert = (scheme,table,container_name,cpu_usage,memory_usage,now)

    # cur.execute(insert_q2)
    # cur.execute(f"insert into {schema}.{tablename}(container_name,cpu,mem,datetime) values ({container},{cpu},{mem},{datetime})"
    #             .format(schema,tablename,container,cpu,mem,datetime))
    
    postgres_conn.commit()
    
    logger.info(f"{container_name} Time: {now} CPU: {cpu_usage} Memory: {memory_usage}")
    # postgres_conn.close()
    # channel.close()

    # cpu usage in percentages
    # print((string_to_dict['cpu_stats']['cpu_usage']['total_usage'] / string_to_dict['cpu_stats']['system_cpu_usage']) * 100) 












import psycopg2
import psycopg2.extras
import pandas as pd


# conn = psycopg2.connect(host="localhost",dbname='postgres',user='postgres',password='123123',port=5432)
# cursor = conn.cursor()
# row = []



def connect_db(host, dbname, user, password, port):
    return psycopg2.connect(host = host,dbname = dbname,user = user,password = password,port = port)
    
def select(conn, dbname):
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute(f"select * from {dbname}".format(dbname))
    rows = cur.fetchall()
    
    return rows

def dict_maker(header, row):
    result_dict = dict(zip(header,row))
    return result_dict

    # gatway_conn = connect_db(host="localhost",dbname='gateway',user='postgres',password='123123',port=5432)
    
    # cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    # cur2 = conn2.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # cur.execute("select * from postgres")
    # cur2.execute("select * from gateway")
    
    # rows = cur.fetchall()

try:

    postgres_data = []
    gateway_data = []
    postgress_conn = connect_db("localhost",'postgres','postgres','123123',5432)
    gateway_conn = connect_db("localhost",'postgres','postgres','123123',5432)
    
    postgres_rows = select(postgress_conn,"postgres")
    
    gateway_rows = select(gateway_conn,"gateway")
    
    postgres_header = ['serviceID','containerID','CPU','Mem','datetime']
    gateway_header = ['service_id','api_id','request_time','response_time']
    
    for row in postgres_rows:
        postgres_data.append(dict_maker(postgres_header,row))
    
    for row in gateway_rows:
        gateway_data.append(dict_maker(gateway_header,row))
    
    final_data = pd.DataFrame(postgres_data)
    
    for i in range(len(gateway_data)):
        for k, v in gateway_data[i].items():
            if k == 'service_id':
                pass

            else:
                final_data[k] = v
                
        for index, row in final_data.iterrows():
        # if final_data[i]['api_id'] == final_data[i]['container_id']:
        #     print(final_data[i])
        
            if row['containerID'] == row['api_id']:
                print(row)
    
    # gateway_data.append(dict_maker(header,gateway_rows))
    # lst = [i[0].split(',') for i in header]
    # df = []
    # resource_dict = {}
    
    # for row in rows:
    #     result_dict = dict(zip(header,row))
    #     df.append(result_dict)
        
        # df.append(dict)
    # print([dict(zip(lst[0], v)) for v in lst[1:]])
    
    # print(postgres_data)
    # print(gateway_data)
except psycopg2.DatabaseError as db_err:
    print(db_err)
    
channel.close()
