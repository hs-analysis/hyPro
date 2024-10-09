# hyPro/Prediction_Server/server.py
# Main Entrypoint for the hyPro Server

# HS Analysis GmbH, 2024
# Author: Valentin Haas

# Python Imports
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import logging
import random

# External Imports
from fastapi import FastAPI, HTTPException, UploadFile, Form, File
import pandas as pd

# Local Imports
from ai_models import list_models
from data_models import AIModel, MachinePreset, PredictionResponse, PredictionSegment
from dataset.input_validation import check_for_model, validate_columns
from dataset.clean import prepare_timeseries

# Constants
VERSION = "0.2.0"
CWD = Path(__file__).parent.resolve()


LOG_FOLDER = CWD / "server_logs"

# region Setup logging
logger = logging.getLogger("fastapi")

logfile = str(LOG_FOLDER / "server.log")
logger.setLevel(logging.DEBUG)
logger.basicConfig = {
    "level": logging.DEBUG,
    "format": "[%(asctime)s.%(msecs)03d] %(levelname)s [%(thread)d] - %(message)s",
}

# Create a file handler for the log file
file_logging_handler = TimedRotatingFileHandler(
    logfile, when="midnight", backupCount=30
)

file_logging_handler.setFormatter(
    logging.Formatter(
        "[%(asctime)s.%(msecs)03d] %(levelname)s [%(thread)d] - %(message)s",
    )
)

file_logging_handler.setLevel(logging.DEBUG)
file_logging_handler.suffix = "%Y-%m-%d"
logger.addHandler(file_logging_handler)
# endregion Setup logging

# region Helper Functions


async def get_model(model_id: str, model_version: str) -> AIModel:
    """
    Load the model information for the given model ID and version.

    Args:
        model_id (str): The ID of the model to load.
        model_version (str): The version of the model to load.

    Raises:
        HTTPException: Model not found.
        HTTPException: Model version not found.
        HTTPException: Multiple models with the same ID and version found.

    Returns:
        AIModel: The model information for the given model ID and version.
    """
    try:
        model = check_for_model(model_id, model_version)
        logger.debug(
            f"Retrieved model information for model {model_id}:{model_version}"
        )
        return model

    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        msg = f"Error while getting model information: {e}"
        logger.error(msg)
        raise HTTPException(status_code=500, detail=msg)


async def extract_timeseries_from_file(file: UploadFile) -> pd.DataFrame:
    """
    Extract the timeseries data from the given file.

    Args:
        file (UploadFile): The file to extract the timeseries data from, as uploaded by the user.

    Raises:
        HTTPException: Error while reading the input file.
        HTTPException: Empty file content.
        HTTPException: Invalid file format.
        HTTPException: Error while saving the input file.

    Returns:
        pd.DataFrame: The extracted timeseries data as a pandas DataFrame.
    """
    file_ext = file.filename.split(".")[-1].lower()
    try:
        file_content = await file.read()
        # Read the file content
        if len(file_content) == 0:
            msg = "Empty file content."
            logger.error(msg)
            raise HTTPException(status_code=400, detail=msg)

        # Save the file to the log folder
        log_file_path = (
            Path(LOG_FOLDER)
            / f'{datetime.now().isoformat().replace(":", "-")}_input.csv'
        )

        try:
            with open(str(log_file_path), "wb") as f:
                f.write(file_content)

        except Exception as e:
            msg = f"Error while saving the input file: {e}"
            logger.error(msg)
            raise HTTPException(status_code=500, detail=msg)

        if file_ext == "csv":
            model_input = pd.read_csv(log_file_path, sep=";", decimal=",")
        elif file_ext == "xlsx" or file_ext == "xls":
            model_input = pd.read_excel(log_file_path)
        else:
            msg = "Invalid file format. Only CSV and XLSX files are supported."
            logger.error(msg)
            raise HTTPException(status_code=400, detail=msg)

        # Remove the input file
        log_file_path.unlink()

        return model_input

    except Exception as e:
        msg = f"Error while reading the input file: {e}"
        logger.error(msg)
        raise HTTPException(status_code=400, detail=msg)


# endregion Helper Functions

# region API Endpoints
# Create a new FastAPI instance
app = FastAPI()


@app.get("/v1/is_alive")
async def is_alive() -> bool:
    """
    Check if the server is alive and responding to requests.

    Returns:
        bool: True if the server is alive, False otherwise
    """
    return True


@app.get("/v1/version")
async def get_version() -> str:
    """
    Get the version of the server.

    Returns:
        str: The version of the server
    """
    return VERSION


@app.get("/v1/models")
async def get_models() -> dict:
    """
    Get a list of all available models.

    Returns:
        dict: A dictionary with the list of available models.
    """

    try:
        models = list_models()
        logger.debug(f"Listed {len(models)} models")
        return {"models": models}

    except Exception as e:
        # Return an error response if an exception occurs
        msg = f"Error while listing models: {e}"
        logger.error(msg)
        raise HTTPException(status_code=500, detail=msg)


@app.get("/v1/models/{model_id}/{model_version}")
async def get_model(model_id: str, model_version: str) -> AIModel:
    """
    Get detailed information about a specific model.

    Args:
        model_id (str): The ID of the model to retrieve information for.

    Returns:
        dict: A dictionary with the detailed information about the model.
    """

    try:
        model = check_for_model(model_id, model_version)
        logger.debug(
            f"Retrieved model information for model {model_id}:{model_version}"
        )
        return model

    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        msg = f"Error while getting model information: {e}"
        logger.error(msg)
        raise HTTPException(status_code=500, detail=msg)


@app.post("/v1/pred/from_timeseries/{model_id}/{model_version}/")
async def predict_from_timeseries(
    model_id: str, model_version: str, file: UploadFile, machine_id: str = "Unknown"
) -> PredictionResponse:
    """
    Endpoint for predicting the P2V and RMS values for a machine using input sensor timeseries data.

    Args:
        model_id (str): The ID of the model that should be used for the prediction.
        model_version (str): The version of the model that should be used for the prediction.
        file (UploadFile): The file containing the input data for the prediction. Must be a CSV file.
        machine_id (str): Optional. The ID of the machine for which the prediction is requested.

    Returns:
        PredictionResponse: The predicted P2V and RMS values for the machine based on the input sensor timeseries data.
    """

    model = await get_model(model_id, model_version)
    model_input = await extract_timeseries_from_file(file)

    # Check model - content compatibility
    try:
        validate_columns(set(sorted(model.input_cols)), model_input)
    except ValueError as ve:
        msg = f"Error while validating input data: {ve}"
        logger.error(msg)
        raise HTTPException(status_code=400, detail=msg)

    # Clean the input data
    prepared_input = prepare_timeseries(model_input, model)

    # Prefill the response with the request information
    response = PredictionResponse(
        machine_id=machine_id,
        ai_model_id=model_id,
        ai_model_version=model_version,
        request_timestamp=datetime.now().isoformat(),
        response_time_s=0.0,
        segments=[],
    )

    try:
        # Placeholder for actual prediction logic
        grid = (5, 5)
        for i in range(grid[0] * grid[1]):
            x = i % grid[0]
            y = i // grid[1]
            response.segments.append(
                PredictionSegment(
                    id=i,
                    pred_p2v=random.uniform(0.0, 1.0),
                    pred_rms=random.uniform(0.0, 1.0),
                    x_range_mm=(x * 10, (x + 1) * 10),
                    y_range_mm=(y * 10, (y + 1) * 10),
                    xy_location=(x, y),
                )
            )

    except Exception as e:
        # Return an error response if an exception occurs
        msg = f"Error while predicting: {e}"
        logger.error(msg)
        raise HTTPException(status_code=500, detail=msg)

    return response


@app.post("/v1/pred/from_presets/{model_id}/{model_version}/")
async def predict_from_presets(
    model_id: str,
    model_version: str,
    presets: MachinePreset,
    machine_id: str = "Unknown",
) -> PredictionResponse:
    """
    Endpoint for predicting the P2V and RMS values for a machine using presets.

    Args:
        model_id (str): The ID of the model that should be used for the prediction.
        model_version (str): The version of the model that should be used for the prediction.
        presets (MachinePreset): The machine presets to use for the prediction.
        machine_id (str): Optional. The ID of the machine for which the prediction is requested.

    Returns:
        PredictionResponse: The predicted P2V and RMS values for the machine with the given presets.
    """

    model = await get_model(model_id, model_version)

    # Prefill the response with the request information
    response = PredictionResponse(
        machine_id=machine_id,
        ai_model_id=model.id,
        ai_model_version=model.version,
        request_timestamp=datetime.now().isoformat(),
        response_time_s=0.0,
        segments=[],
    )

    try:
        # Placeholder for actual prediction logic
        grid = (1, 1)
        for i in range(grid[0] * grid[1]):
            x = i % grid[0]
            y = i // grid[1]
            response.segments.append(
                PredictionSegment(
                    id=i,
                    pred_p2v=random.uniform(0.0, 1.0),
                    pred_rms=random.uniform(0.0, 1.0),
                    x_range_mm=(x * 10, (x + 1) * 10),
                    y_range_mm=(y * 10, (y + 1) * 10),
                    xy_location=(x, y),
                )
            )

    except Exception as e:
        # Return an error response if an exception occurs
        msg = f"Error while predicting: {e}"
        logger.error(msg)
        raise HTTPException(status_code=500, detail=msg)

    return response


@app.post("/v1/pred/from_ps_and_ts/{model_id}/{model_version}/")
async def predict_from_presets_and_timeseries(
    model_id: str,
    model_version: str,
    presets: str = Form(...),
    file: UploadFile = File(...),
    machine_id: str = Form("Unknown"),
) -> PredictionResponse:
    """
    Endpoint for predicting the P2V and RMS values for a machine using presets and input sensor timeseries data.

    Args:
        model_id (str): The ID of the model that should be used for the prediction.
        model_version (str): The version of the model that should be used for the prediction.
        presets (MachinePreset): The machine presets to use for the prediction.
        file (UploadFile): The file containing the input data for the prediction. Must be a CSV file.
        machine_id (str): Optional. The ID of the machine for which the prediction is requested.

    Returns:
        PredictionResponse: The predicted P2V and RMS values for the machine with the given presets and input sensor timeseries data.
    """

    try:
        presets = MachinePreset.parse_raw(presets)
    except Exception as e:
        msg = f"Error while parsing presets: {e}"
        logger.error(msg)
        raise HTTPException(status_code=400, detail=msg)

    model = await get_model(model_id, model_version)
    model_input = await extract_timeseries_from_file(file)

    # Check model - content compatibility
    try:
        validate_columns(set(sorted(model.input_cols)), model_input)
    except ValueError as ve:
        msg = f"Error while validating input data: {ve}"
        logger.error(msg)
        raise HTTPException(status_code=400, detail=msg)

    # Clean the input data
    prepared_input = prepare_timeseries(model_input, model)

    # Prefill the response with the request information
    response = PredictionResponse(
        machine_id=machine_id,
        ai_model_id=model.id,
        ai_model_version=model.version,
        request_timestamp=datetime.now().isoformat(),
        response_time_s=0.0,
        segments=[],
    )

    try:
        # Placeholder for actual prediction logic
        grid = (1, 5)
        for i in range(grid[0] * grid[1]):
            x = i % grid[0]
            y = i // grid[1]
            response.segments.append(
                PredictionSegment(
                    id=i,
                    pred_p2v=random.uniform(0.0, 1.0),
                    pred_rms=random.uniform(0.0, 1.0),
                    x_range_mm=(x * 10, (x + 1) * 10),
                    y_range_mm=(y * 10, (y + 1) * 10),
                    xy_location=(x, y),
                )
            )

    except Exception as e:
        # Return an error response if an exception occurs
        msg = f"Error while predicting: {e}"
        logger.error(msg)
        raise HTTPException(status_code=500, detail=msg)

    return response


# endregion API Endpoints

# region Setup Debugging
if __name__ == "__main__":
    # Run the server
    import uvicorn

    HOST = "127.0.0.1"
    PORT = 8000

    logger.info(f"Starting uvivorn server at {HOST}:{PORT}...")
    uvicorn.run(app, host=HOST, port=PORT, log_config=str(CWD / "logging.conf"))
# endregion Setup Debugging
