# pylint: disable=abstract-method, redefined-outer-name, unused-argument, protected-access
from unittest.mock import MagicMock, patch
import pytest

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

from usf_model_api.serving.utils import MockDatabase
from usf_model_api.models.base import PredictionModel


class LinearRegressionModel(PredictionModel):
    def __init__(self, model_id: str, preprocessor=None, predictor=None):
        super().__init__(
            model_id=model_id, preprocessor=SimpleImputer(), predictor=LinearRegression()
        )

        self._model = LinearRegression()


@pytest.fixture
def mock_model():
    model = MagicMock(spec=PredictionModel)
    model.model_id = "test_model"
    return model


@pytest.fixture
def mock_model_dir(tmp_path, mock_model):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / "test_model.pkl"
    LinearRegressionModel(model_id="test_model").serialize(model_path)

    return model_dir


def test_mock_database_initialization(mock_model_dir):
    db = MockDatabase(model_dir=mock_model_dir)
    assert isinstance(db.model_db, dict)
    assert isinstance(db.predictions_db, pd.DataFrame)


def test_load_models(mock_model_dir, mock_model):
    with patch("usf_model_api.models.base.PredictionModel.deserialize", return_value=mock_model):
        db = MockDatabase()
        db.load_models(mock_model_dir)
        assert "test_model" in db.model_db
        assert db.model_db["test_model"] == mock_model


def test_get_model(mock_model):
    db = MockDatabase()
    db._model_db["test_model"] = mock_model
    model = db.get_model("test_model")
    assert model == mock_model
    assert db.get_model("non_existent_model") is None


def test_save_predictions():
    db = MockDatabase()
    predictions = pd.DataFrame(
        {
            "prediction_id": ["1", "2"],
            "prediction": [0.5, 0.7],
            "created_at": ["2023-01-01 00:00:00.000", "2023-01-01 00:00:00.000"],
        }
    )
    db.save_predictions(predictions)
    assert not db.predictions_db.empty
    assert len(db.predictions_db) == 2
    assert "prediction_id" in db.predictions_db.columns
    assert "prediction" in db.predictions_db.columns
    assert "created_at" in db.predictions_db.columns
