import requests
import os
import paramiko
import getpass
import time
import json
from json import dumps, loads
from datetime import datetime


cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)

# ## 222 서버
# server = "172.16.28.222"
# user = "e8ight"
# cli.connect(server, username=user, password="ndxpro123!!")
 
# # 새로운 interactive shell session 생성
# channel = cli.invoke_shell()

# # 명령 송신
# channel.send("su\n")
# channel.send("ndxpro123!!")

# channel.send("curl --unix-socket /var/run/docker.sock http://localhost/v1.43/containers/logstash/stats?stream=true\n")
# time.sleep(1.0)
# # 결과 수신
# output = channel.recv(65535).decode("UTF-8").replace(";",'"')


# 220 서버
server = "172.16.28.220"
user = "root"
# pwd = getpass.getpass("Password: ")
cli.connect(server, username=user, password="!!ndxpro123!!220")


stdin, stdout, stderr = cli.exec_command("ls -la")
lines = stdout.readlines()

 
# 새로운 interactive shell session 생성
channel = cli.invoke_shell()
 
count = 0 
while True:
    channel.send("curl --unix-socket /var/run/docker.sock http://localhost/v1.41/containers/data-manager/stats?stream=true\n")
    time.sleep(1.0)
    # 결과 수신
    output = channel.recv(65535).decode("UTF-8").replace(";",'"')



    output_lst = output.splitlines(0)
    # print(output_lst[5]) # output의 split lines 6번째가 stats 값이다.

    # print(output_lst)
    # print(len(output_lst))
    # json.load(output_lst[5],)
    resource = json.loads(output_lst[len(output_lst)-1].replace("'", "\""))

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

    print("memory :", memory_usage, datetime.now())
    print("cpu usage :", cpu_usage)
    # cpu usage in percentages
    # print((string_to_dict['cpu_stats']['cpu_usage']['total_usage'] / string_to_dict['cpu_stats']['system_cpu_usage']) * 100) 
    count += 1

channel.close()
