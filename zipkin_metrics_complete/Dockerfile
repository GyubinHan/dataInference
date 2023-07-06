FROM python:3.10.11-slim

WORKDIR /app

COPY requirements2.txt requirements.txt
COPY elasticsearchmetrics.py elasticsearchmetrics.py
COPY elasticsearchzipkin.py elasticsearchzipkin.py



# ENV CONTAINER_NAME "data-broker-3"
# ENV DOCKER_LOG /app/logFile/docker-databroker.log
RUN apt-get update && apt-get install -y procps && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get -y install cron
RUN pip install -r requirements.txt
RUN cron


# CMD ["crontest.py" ]
# ENTRYPOINT [ "python" ]
# Copy cron file to the cron.d directory on container
COPY cron_zipkin /etc/cron.d/cron_zipkin
# Give execution access
RUN chmod 0755 /etc/cron.d/cron_zipkin
# Run cron job on cron file
RUN crontab /etc/cron.d/cron_zipkin
# Create the log file
RUN touch /var/log/cron_zipkin.log

COPY cron_metricset /etc/cron.d/cron_metricset
# Give execution access
RUN chmod 0755 /etc/cron.d/cron_metricset
# Run cron job on cron file
RUN crontab /etc/cron.d/cron_metricset
# Create the log file
RUN touch /var/log/cron_metricset.log

CMD ["cron", "-f"]