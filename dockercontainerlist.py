import docker

# Create a Docker client
client = docker.from_env()

# Get all containers
containers = client.containers.list(all=True)

# Iterate over the containers and print their status
for container in containers:
    container_status = container.status
    print(f"Container ID: {container.id}")
    print(f"Status: {container_status}")
    print("---")