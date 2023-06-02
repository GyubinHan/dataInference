import requests
import os
import paramiko
import getpass
import time
import json
from json import dumps, loads
from datetime import datetime
import psycopg2
import psycopg2.extras
import pandas as pd
import logging


def connect_db(host, dbname, user, password, port):
    return psycopg2.connect(host = host,dbname = dbname,user = user,password = password,port = port)

def insert(conn,schema,tablename,container,cpu,mem,datetime):
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    # cur.execute(f"insert into {schema}.{tablename}(container_name,cpu,mem,datetime) values ({container},{cpu},{mem},{datetime})"
    #             .format(schema,tablename,container,cpu,mem,datetime))
    insert_query = """ INSERT INTO datainference.dockerstatus (container_name, cpu, mem, datetime) VALUES (%s,%s,%s,%s)"""
    record_to_insert = (container,cpu,mem,datetime)

    cur.execute(insert_query, record_to_insert)
    
#(container,cpu,mem,datetime) 
def select(conn, dbname):
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute(f"select * from {dbname}".format(dbname))
    rows = cur.fetchall()
    
    return rows

def dict_maker(header, row):
    result_dict = dict(zip(header,row))
    return result_dict

# logging
logger = logging.getLogger('docker')
logger.setLevel(logging.DEBUG)





# 220 서버 ssh 연결
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)
server = "172.16.28.220"
user = "root"
# pwd = getpass.getpass("Password: ")
cli.connect(server, username=user, password="!!ndxpro123!!220")


stdin, stdout, stderr = cli.exec_command("ls -la")
lines = stdout.readlines()

 
# 새로운 interactive shell session 생성
channel = cli.invoke_shell()
 
count = 0 
container_number = 1
container_name = 'data-manager'

while True:
    start_time = time.time()
    container_name_save = container_name + str(container_number)
    channel.send(f"curl --unix-socket /var/run/docker.sock http://localhost/v1.41/containers/{container_name}/stats?stream=true\n".format(container_name))
    time.sleep(0.)
    # 결과 수신
    output = channel.recv(65535).decode("UTF-8").replace(";",'"')
    output_lst = output.splitlines(0)
    # print(output_lst)
    try:
        resource = json.loads(output_lst[len(output_lst)-1].replace("'", "\""),strict=False)

            # Process the JSON data
    except FileNotFoundError:
        print("File not found.")
    except json.JSONDecodeError as e:
        print("JSON decoding error:", str(e))
    # resource = json.loads(output_lst[len(output_lst)-1].replace("'", "\""),strict=False)

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

    # print("memory :", memory_usage, datetime.now())
    # print("cpu usage :", cpu_usage)
    
    count = str(datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
    if container_number < 5:
        container_number += 1
    else:
        container_number = 1
    
    # docker stat dictionary
    docker_stat = {"container_name":container_name_save,
                   'cpu': cpu_usage,
                   'mem':memory_usage,
                   'datetime':count}
    
    
    
    # try:

    #     postgres_data = []
    #     postgres_conn = connect_db("localhost",'postgres','postgres','123123',5432)
        
    #     postgres_rows = insert(postgres_conn,"datainference",'dockerstatus',container_name_save,cpu_usage,memory_usage,count)
    #     postgres_conn.commit()
    #     logger.info('Finished logging, postgre insert successfully')
    #     print(docker_stat)
    #     print(time.time()-start_time)
        
    # except psycopg2.DatabaseError as db_err:
        
    #     print(db_err)
                

channel.close()

