FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY main.py .
COPY physics.py .
COPY database.py .

RUN mkdir -p /app/data
ENV ACM_DB_PATH=/app/data/acm_sim.db

VOLUME ["/app/data"]
EXPOSE 8000
CMD ["python3", "main.py"]
