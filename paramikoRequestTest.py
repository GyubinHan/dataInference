import requests
from requests.auth import HTTPBasicAuth
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

# def get_container_memory_stats(container_id):
#     url = f"http://172.16.28.220/containers/{container_id}/stats"
#     user = 'root'
#     password = '!!ndxpro123!!220'
#     # session = requests.Session()
#     # session.auth = (user, password)
#     response = requests.get(url, stream=True,auth = HTTPBasicAuth(user, password), 
#                            headers={ "User-Agent" : "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"})

    
    
#     if response.status_code == 200:
#         for data in response.iter_lines():
#             # Process the streamed data
#             memory_stats = data.decode('utf-8')
#             # You can parse and extract the memory statistics from the 'memory_stats' variable
#             print(memory_stats)
#     else:
#         print(f"Error: {response.status_code} - {response.text}")

# # Replace 'container_id' with the actual ID of the container you want to get memory stats for
# container_id = "data-manager"
# get_container_memory_stats(container_id)




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
    channel.send("docker ps")
    time.sleep(1.0)
    # 결과 수신
    output = channel.recv(65535).decode("UTF-8").replace(";",'"')

    print(output)

    output_lst = output.splitlines(0)
    
    # print(output_lst[5]) # output의 split lines 6번째가 stats 값이다.

    # print(output_lst)
    # print(len(output_lst))
    # json.load(output_lst[5],)
    # resource = json.loads(output_lst[len(output_lst)-1].replace("'", "\""))
    # print(resource)