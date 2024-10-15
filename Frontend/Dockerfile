# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libmariadb-dev \
    libmariadb-dev-compat \
    libmariadb3 \
    && rm -rf /var/lib/apt/lists/*
    
# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Run app.py when the container launches
CMD ["python", "app.py"]
