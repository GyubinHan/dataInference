import requests
import json
import docker
from docker.errors import APIError, TLSParameterError

# def get_docker_stats(server_url):
#     client = docker.DockerClient(base_url=server_url)
#     containers = client.containers.list()
    
#     stats = {}
#     for container in containers:
#         container_stats = container.stats(stream=False)
#         stats[container.name] = container_stats
    
#     return stats

# # Specify the server URL where Docker daemon is running
# server_url = 'http://172.16.28.220:54008'

# # Get Docker container stats from the specified server
# container_stats = get_docker_stats('localhost:8000')

# # Print the container stats
# for container_name, stats in container_stats.items():
#     print(f"Container: {container_name}")
#     print(f"Stats: {stats}\n")

import docker

def get_container_status(container_name, server_ip):
    # Create a Docker client object
    client = docker.DockerClient(base_url='tcp://' + server_ip + ':2375')
    auth_config = {
        'username': 'root',
        'password': '!!ndxpro123!!220',
    }

    # Set the authentication configuration on the Docker client
    docker_client.login(**auth_config)

    # Get a list of all containers
    containers = client.containers.list()

    # Iterate over the containers to find the desired container
    for container in containers:
        if container_name in container.name:
            # Get the container's status
            status = container.status
            return status

    # Return None if the container is not found
    return None

# Provide the container name and server IP
container_name = 'data-manger'
server_ip = '172.16.28.220'

# Call the function to get the container status
status = get_container_status(container_name, server_ip)

# Print the container status
print(f"Container status: {status}")


# import requests

# params = {
#     'stream': 'true',
# }

# response = requests.get('http://localhost/v1.41/containers/data-manager/stats', params=params)

# while response:
#     print(response)

# import os

# def get_docker_server_info():
#     docker_host = os.environ.get('DOCKER_HOST')
#     if docker_host:
#         server_url = docker_host.split('//')[1]
#         server_ip = server_url.split(':')[0]
#         server_port = server_url.split(':')[1]
#         return server_ip, server_port
#     else:
#         return None, None

# # Get Docker engine server IP and port
# server_ip, server_port = get_docker_server_info()

# # Print the IP and port
# print(f"Docker server IP: {server_ip}")
# print(f"Docker server port: {server_port}")


####################################
# 되는 코드
# client = docker.from_env()
# print(client.containers.list())
# # client = docker.DockerClient(base_url='unix:///var/run/docker.sock')
# client = docker.DockerClient(base_url='tcp://172.19.0.19:2375')


# for i in client.containers.list():
#     for stat in i.stats():
#         # print(str(stat))
#         # new_stat = dict(str(stat))
#         # new_stat = stat    
#         # serialized = json.dumps(stat)

#         new_stat = stat.decode('utf-8').replace(";",'"')
#         print(type(new_stat))
#         string_to_dict = json.loads(new_stat.replace("'", "\""))
#         print(string_to_dict['cpu_stats']['system_cpu_usage'])
#         # deserialized = json.loads(stat)        
#         # print(deserialized['memory_usage'])
#     # new_stat = list(stat)
#         # print(new_stat)

####################################
# 안되는 코드 
# import requests

# # API 엔드포인트와 컨테이너 ID 설정
# api_endpoint = 'http://172.19.0.19:54008/containers/baefd8dbb30d/stats'
# container_id = 'baefd8dbb30d'

# # API 호출
# response = requests.get(api_endpoint.format(container_id=container_id))

# # 응답 확인
# if response.status_code == 200:
#     stats_data = response.json()
#     # stats 데이터 처리
#     # ...
#     print(stats_data)
# else:
#     print("API 호출 실패:", response.text)
    
    
    
    
    
    
    
    
####################################
# 안되는 코드 
# def get_container_memory_usage(docker_engine_ip, container_id):
#     url = f"http://{docker_engine_ip}/containers/{container_id}/stats"
    
#     response = requests.get(url)
#     if response.status_code != 200:
#         print(f"Error: Failed to retrieve container stats. Status code: {response.status_code}")
#         return None
    
#     stats = json.loads(response.content.decode('utf-8'))
#     memory_usage = stats['memory_stats']['usage']
    
#     return memory_usage

# def get_mem_perc(stats):
#     mem_used = stats["memory_stats"]["usage"] - stats["memory_stats"]["stats"]["cache"] + stats["memory_stats"]["stats"]["active_file"]
#     limit = stats['memory_stats']['limit']
#     return round(mem_used / limit * 100, 2)


# docker_engine_ip = '172.19.0.19'
# container_id = 'data-manager'

# memory_usage = get_container_memory_usage(docker_engine_ip, container_id)
# print(f"Memory usage: {memory_usage}")
