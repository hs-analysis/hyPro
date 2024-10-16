# dockerfile
# Dockerfile for hypro server

# Copyringht HS Analysis GmbH, 2024
# Author: Valentin Haas

FROM python:3.12-slim AS BASE
WORKDIR /app

# For Healthchecks
RUN apt-get update -yq &&\ 
    apt-get upgrade -yq &&\ 
    apt-get install -yq curl && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade --no-cache-dir -r requirements.txt

# Copy the source code
COPY . /app

HEALTHCHECK --interval=10s --timeout=2s --start-period=5s --retries=3 CMD [ "curl", "--fail", "http://localhost:80/v1/is_alive/" ]

# Run the server
CMD ["fastapi", "run", "/app/server.py", "--proxy-headers", "--port", "80"]