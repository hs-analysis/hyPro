# hyPro prediction Server 🖌️

This project is the model API for the Hypro application.
It is build using a fastAPI Server, uses pyTorch for model inference, and is containerized using docker.
The corresponding frontend can use the api and provide suiable visualisations.

## Prerequisites 📋

Before you begin, ensure you have met the following requirements:

- Docker Compose: [Install Docker Compose](https://docs.docker.com/compose/install/)

### Installation 🔧

1. **Setup the environment variables:**
   Change the values of the `.env` file where necessary.


## Usage 🚀

### Building with Docker Compose 🏗️

Build the application using Docker Compose:

```bash
docker-compose build
```

This will load the necessary dependencies and build the application. Once the build is complete, you can run the application using Docker Compose.

### Running the Application 🏃

Start the application using Docker Compose:

```bash
docker-compose up
```

This will start all the services defined in your docker-compose.yml file. 
By default, api will run on http://localhost:80.

By default, automatically generated fastAPI API documentation can be accessible in a webbrowser at http://localhost:80/docs