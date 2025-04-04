import logging
from typing import Any

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_absolute_percentage_error
from sklearn import set_config

from usf_model_api.models.base import PredictionModel, ModelDataset


set_config(transform_output="pandas")
logging.basicConfig()

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

MODEL_PARAMS = {
    "verbose": 100,
    "loss_function": "RMSE",
    "learning_rate": 0.4, #0.3,
    "depth": 5,
    # "l2_leaf_reg": 30, #2, #5, #3, #5, #2.5, #5, #10, #0.1,
    "n_estimators": 3_000,
    "boost_from_average": True,
    "bootstrap_type": "MVS", #"No",
    "subsample": 0.8,
    "eval_metric": "MAPE",
}

TRAIN_DATA_LOC = "../../downloads/train.csv"
SAVE_MODEL_LOC = "../../service/assets/sales_forecast_model.pkl"
TRAIN_PCT = 0.8
TARGET = "sales"


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None) -> "DateFeatureExtractor":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X['day'] = pd.to_datetime(X['date']).dt.day
        X['month'] = pd.to_datetime(X['date']).dt.month
        X['year'] = pd.to_datetime(X['date']).dt.year
        X['day_of_week'] = pd.to_datetime(X['date']).dt.dayofweek
        return X.drop(columns=['date'])


class SalesDataset(ModelDataset):
    def __init__(self, data: pd.DataFrame, train_pct: float = 0.8, random_seed: int = 42):
        self.data = data
        self.train_pct = train_pct
        self.random_seed = random_seed
        self.splits = self._get_splits(data)

    def _get_splits(self, data: pd.DataFrame):
        train_df, test_df = train_test_split(
            data,
            train_size=self.train_pct,
            shuffle=True,
            random_state=self.random_seed
        )
        return {
            "train": train_df,
            "test": test_df,
        }

    def get_training_split(self):
        return self.splits["train"]

    def get_test_split(self):
        return self.splits["test"]


class SalesForecastingModel(PredictionModel):
    def __init__(self, preprocessor: Any, predictor: CatBoostRegressor):
        super().__init__(preprocessor=preprocessor, predictor=predictor)

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> float:
        check_is_fitted(self.model)

        predictions = self.model.predict(X)
        mae = mean_absolute_percentage_error(y_true=y, y_pred=predictions)

        return mae


if __name__ == "__main__":
    LOG.info("Loading training data from %s", TRAIN_DATA_LOC)
    data = pd.read_csv(TRAIN_DATA_LOC)

    LOG.info("Creating training and test splits")
    model_dataset = SalesDataset(data, train_pct=TRAIN_PCT)
    train_df = model_dataset.get_training_split()
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].to_numpy()

    LOG.info("Training model ...")
    model = SalesForecastingModel(
        preprocessor=DateFeatureExtractor(), predictor=CatBoostRegressor(**MODEL_PARAMS)
    )
    model.fit(X_train, y_train)

    LOG.info("Evaluating model ...")
    test_df = model_dataset.get_test_split()
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET].to_numpy()
    score = model.evaluate(X_test, y_test)
    LOG.info("Model evaluation score: %s", score)

    LOG.info("Saving model to %s", SAVE_MODEL_LOC)
    model.serialize(SAVE_MODEL_LOC)
