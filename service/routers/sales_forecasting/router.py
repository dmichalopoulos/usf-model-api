from typing import List, Dict, Any
from uuid import uuid4
from datetime import datetime
from pathlib import Path
from http import HTTPStatus
from dateutil.parser import parse

from pydantic import field_validator
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

import pandas as pd

from usf_model_api.serving.base import PredictionRequest
from usf_model_api.utils import get_logger
from usf_model_api.serving.utils import MockDatabase


LOG = get_logger(__name__)
SAVED_MODEL_LOC = Path(__file__).parent / "assets"
SIMPLE_DB = MockDatabase(model_dir=SAVED_MODEL_LOC)


router = APIRouter(
    prefix="/sales-forecasting",
    tags=["ai_model"],
    dependencies=None,
)


class SalesForecastRequest(PredictionRequest):
    """
    A subclass of ``PredictionRequest`` used to represent a sales forecast request.

    Attributes
    ----------
    date : str
        The date for which the forecast is requested.
    store : int
        The store identifier.
    item : int
        The item identifier.
    """
    date: str
    store: int
    item: int

    @field_validator("date")
    def check_date(cls, date: str):
        """
        Checks that the ``date`` field is valid and correctly formatted.

        Parameters
        ----------
        date : str
            The date string to validate.

        Returns
        -------
        str
            The validated date string.

        Raises
        ------
        ValueError
            If the date string is not in a valid format.
        """
        try:
            parse(date)
        except ValueError:
            LOG.exception(
                "Invalid date value '%s'. Ensure you pass in a valid date with format 'yyyy-MM-dd'",
                date,
            )
            raise

        return date


@router.get("/")
def read_root():
    """
    Application root endpoint.

    Returns
    -------
    JSONResponse
        A JSON response with a welcome message.
    """
    return JSONResponse(
        status_code=HTTPStatus.OK,
        content={"message": "Welcome to the Sales Forecasting AI Model service!"},
    )


@router.get("/status", status_code=HTTPStatus.OK)
def get_app_status() -> JSONResponse:
    """
    This endpoint returns the status of the application.
    It can be used to check if the application is up and running.

    Returns
    -------
    JSONResponse
        A JSON response with the application status.
    """
    return JSONResponse(
        status_code=HTTPStatus.OK,
        content={"message": f"Status Code {HTTPStatus.OK}: The app is up and running."},
    )


def _predict(prediction_request: SalesForecastRequest | List[SalesForecastRequest]) -> List[Dict[str, Any]]:
    """
    Generates and stores (atomically; either all are successful or nothing is written) predictions
    for one or more sales forecast requests. Each request is allowed to request a specific model deployment
    based on its ``model_id`` field.

    Parameters
    ----------
    prediction_request : SalesForecastRequest | List[SalesForecastRequest]
        A single sales forecast request or a list of sales forecast requests.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries containing the predictions.
    """
    # Create and prepare dataframe for scoring
    to_score = prediction_request if isinstance(prediction_request, list) else [prediction_request]
    scoring_df = pd.DataFrame.from_records(map(lambda x: x.model_dump(), to_score))
    scoring_df.insert(0, column="prediction_id", value=None)
    scoring_df["prediction"] = None
    scoring_df["created_at"] = None
    features = sorted(
        set(scoring_df.columns) - {"model_id", "prediction_id", "prediction", "created_at"}
    )
    LOG.info("Identified model features: %s", features)

    # Score by model requested for each batch (also generalizes to one model)
    requested_models = scoring_df["model_id"].unique()
    LOG.info("Requested models: %s", requested_models.tolist())
    for m in requested_models:
        LOG.info("Running scoring with model '%s' ...", m)
        model = SIMPLE_DB.get_model(m)
        if not model:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"Model with ID '{m}' not found.")

        predictions = model.predict(X=scoring_df.loc[scoring_df["model_id"] == m, features])
        scoring_df.loc[scoring_df["model_id"] == m, "prediction"] = predictions
        scoring_df.loc[scoring_df["model_id"] == m, "prediction_id"] = [
            str(uuid4()) for _ in range(len(predictions))
        ]
        scoring_df.loc[scoring_df["model_id"] == m, "created_at"] = pd.to_datetime(
            datetime.utcnow()
        ).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    assert not scoring_df.isnull().values.any(), "Prediction dataframe contains NaN values."
    # Save predictions to database
    SIMPLE_DB.save_predictions(scoring_df)

    return scoring_df.to_dict(orient="records")


@router.post("/predict")
def predict(prediction_request: SalesForecastRequest | List[SalesForecastRequest]) -> JSONResponse:
    """
    This endpoint is used to get model predictions. The model(s) used to generate predictions
    is determined by the ``model_id`` field value in each ``SalesForecastRequest`` object.

    Parameters
    ----------
    prediction_request : SalesForecastRequest | List[SalesForecastRequest]
        A single sales forecast request or a list of sales forecast requests.

    Returns
    -------
    JSONResponse
        A JSON response containing the predictions.
    """
    predictions = _predict(prediction_request)

    return JSONResponse(
        status_code=HTTPStatus.OK,
        content={
            "message": f"Prediction request successful.",
            "predictions": predictions,
        },
    )
