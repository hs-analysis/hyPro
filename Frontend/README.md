# hyPro Frontend 🖌️

This project is the frontend for the Hypro application. It is built using the Gradio library and is used to provide a user interface for the Hypro ML models. The frontend is built using Python and is containerized using Docker. The frontend is designed to be used in conjunction with the model API.

## Prerequisites 📋

Before you begin, ensure you have met the following requirements:

- Docker Compose: [Install Docker Compose](https://docs.docker.com/compose/install/)

### Installation 🔧

1. **Setup the environment variables:**
   Create an `.env` file in the root directory and add the following variables:

```bash
DB_USERNAME=user
DB_PASSWORD="password" # Must be urlencoded
DB_HOST=X.X.X.X
DB_PORT=3306
DB_NAME=XXXXXXX
```

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

This will start all the services defined in your docker-compose.yml file. By default, the Gradio frontend will be accessible at http://localhost:7860.

## Testing 🧪

For testing purposes, the `documents` directory contains a `manual_input.csv` file that can be used as a template for the input data. The `time_series.csv` file contains an example of time series data that can also be used as input data.
