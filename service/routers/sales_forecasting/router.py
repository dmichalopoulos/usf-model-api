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
    # responses={404: {"description": "Not found"}},
)


class SalesForecastRequest(PredictionRequest):
    date: str
    store: int
    item: int

    @field_validator("date")
    def check_date(cls, date: str):
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
    Application root
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
    """
    return JSONResponse(
        status_code=HTTPStatus.OK,
        content={"message": "The app is up and running."},
    )

def _predict(model_id: str, prediction_request: SalesForecastRequest | List[SalesForecastRequest]) -> List[Dict[str, Any]]:
    """
    This function should contain the logic to predict using the model.
    For now, it just returns a dummy prediction.
    """
    model = SIMPLE_DB.get_model(model_id)
    if not model:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"Model with ID '{model_id}' not found.")

    to_score = prediction_request if isinstance(prediction_request, list) else [prediction_request]
    scoring_df = pd.DataFrame.from_records(map(lambda x: x.model_dump(), to_score))
    scoring_df["prediction"] = model.predict(X=scoring_df.drop(columns=["model_id"]))
    scoring_df.insert(
        0, column="prediction_id", value=[str(uuid4()) for _ in range(len(scoring_df))]
    )
    scoring_df["created_at"] = pd.to_datetime(datetime.utcnow()).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    SIMPLE_DB.save_predictions(scoring_df)

    return scoring_df.to_dict(orient="records")


@router.post("/predict/{model_id}")
def predict(model_id: str, prediction_request: SalesForecastRequest | List[SalesForecastRequest]) -> JSONResponse:
    predictions = _predict(model_id, prediction_request)

    return JSONResponse(
        status_code=HTTPStatus.OK,
        content={
            "message": f"Prediction for model ID '{model_id}' was successful.",
            "predictions": predictions,
        },
    )
