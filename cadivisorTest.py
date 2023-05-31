import requests

# URL for cAdvisor API endpoint
cadvisor_url = 'http://172.16.28.220:54008/api/v1.3/containers/docker/data-manager'

# Replace {container_id} with the actual container ID you want to monitor

# Make a request to cAdvisor API
response = requests.get(cadvisor_url)

# Parse the response
if response.status_code == 200:
    stats = response.json()
    cpu_usage = stats['stats'][-1]['cpu']['usage']['total']
    print('CPU Usage:', cpu_usage)
else:
    print('Error retrieving container stats:', response.status_code)