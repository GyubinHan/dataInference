FROM python:3.10.11-slim

WORKDIR /app

COPY requirements.txt requirements.txt
COPY /dockerdatainsert.py dockedatainsert.py
COPY /docker-volume/  /app/logFile

# ENV CONTAINER_NAME "data-broker-3"
# ENV DOCKER_LOG /app/logFile/docker-databroker.log

RUN pip install -r requirements.txt


CMD ["dockedatainsert.py" ]
ENTRYPOINT [ "python" ]