import sys
from pathlib import Path

# Ugly hack, but it works for now
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from fastapi.testclient import TestClient
from http import HTTPStatus
from unittest.mock import patch, MagicMock

from service.routers.sales_forecasting.router import router, SalesForecastRequest, SIMPLE_DB

client = TestClient(router)


def test_read_root():
    response = client.get("/sales-forecasting")
    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"message": "Welcome to the Sales Forecasting AI Model service!"}


def test_get_app_status():
    response = client.get("/sales-forecasting/status")
    assert response.status_code == HTTPStatus.OK
    assert response.json() == {
        "message": f"Status Code {HTTPStatus.OK}: The app is up and running."
    }


@patch.object(SIMPLE_DB, "get_model", return_value=MagicMock(predict=lambda X: [0.5] * len(X)))
def test_predict_single_request(mock_get_model):
    request_data = {"date": "2023-01-01", "store": 1, "item": 1, "model_id": "test_model"}
    response = client.post("/sales-forecasting/predict", json=request_data)
    assert response.status_code == HTTPStatus.OK
    assert "predictions" in response.json()
    assert len(response.json()["predictions"]) == 1


@patch.object(SIMPLE_DB, "get_model", return_value=MagicMock(predict=lambda X: [0.5] * len(X)))
def test_predict_multiple_requests(mock_get_model):
    request_data = [
        {"date": "2023-01-01", "store": 1, "item": 1, "model_id": "test_model"},
        {"date": "2023-01-02", "store": 2, "item": 2, "model_id": "test_model"},
    ]
    response = client.post("/sales-forecasting/predict", json=request_data)
    assert response.status_code == HTTPStatus.OK
    assert "predictions" in response.json()
    assert len(response.json()["predictions"]) == 2


def test_sales_forecast_request_date_validation():
    with pytest.raises(ValueError):
        SalesForecastRequest(date="invalid-date", store=1, item=1, model_id="test_model")
