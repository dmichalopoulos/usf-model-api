import argparse
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np

from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_absolute_percentage_error

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from usf_model_api.models.base import PredictionModel, ModelDataset
from usf_model_api.utils import get_logger, load_yaml


set_config(transform_output="pandas")
LOG = get_logger(__name__)


# Model types
VALID_MODEL_TYPES = {"catboost", "lgbm"}


# Defaults (can be overridden by command line args)
DEFAULT_SCRIPT_PATH = Path(__file__).resolve().parent
DEFAULT_TRAIN_DATA_LOC = DEFAULT_SCRIPT_PATH.joinpath("../..", "downloads", "train.csv")
DEFAULT_SAVE_MODEL_LOC = DEFAULT_SCRIPT_PATH.joinpath(
    "../..",
    "service",
    "routers",
    "sales_forecasting",
    "assets"
)
DEFAULT_TRAIN_PCT = 0.8
DEFAULT_RANDOM_SEED = 42

# Model parameters
TARGET = "sales"
# MODEL_PARAMS = load_yaml("./params.yaml")
MODEL_PARAMS = load_yaml(DEFAULT_SCRIPT_PATH / "params.yaml")


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

    def _get_splits(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
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

    def get_training_split(self) -> pd.DataFrame:
        return self.splits["train"]

    def get_test_split(self) -> pd.DataFrame:
        return self.splits["test"]


class SalesForecastingModel(PredictionModel):
    def __init__(self, model_id: str, preprocessor: Any, predictor: CatBoostRegressor):
        super().__init__(model_id=model_id,preprocessor=preprocessor, predictor=predictor)

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> float:
        check_is_fitted(self.model)

        predictions = self.model.predict(X)
        mae = mean_absolute_percentage_error(y_true=y, y_pred=predictions)

        return mae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a sales forecasting model.")
    # parser.add_argument(
    #     "--model-name",
    #     type=str,
    #     required=True,
    #     help="Location of the training data CSV file.",
    # )
    parser.add_argument(
        "--model-name",
        action="append",
        help=f"Models to train. One of {VALID_MODEL_TYPES}.",
        type=str,
        default=[],
    )
    parser.add_argument(
        "--data-loc",
        type=str,
        default=DEFAULT_TRAIN_DATA_LOC,
        help="Location of the training data CSV file.",
    )
    parser.add_argument(
        "--save-loc",
        type=str,
        default=DEFAULT_SAVE_MODEL_LOC,
        help="Location to save the trained model.",
    )
    parser.add_argument(
        "--train-pct",
        type=float,
        default=DEFAULT_TRAIN_PCT,
        help="Percentage of data to use for training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args()


def train_models(args: argparse.Namespace):
    # If an unspecified model name is provided, raise an error
    if set(args.model_name) - VALID_MODEL_TYPES:
        raise ValueError(
            "Expected elements of 'model_name' to be one of %s, but found '%s'.",
            VALID_MODEL_TYPES,
            args.model_name
        )

    LOG.info("Loading training data from %s", args.data_loc)
    data = pd.read_csv(args.data_loc)

    LOG.info("Creating training and test splits")
    model_dataset = SalesDataset(data, train_pct=args.train_pct, random_seed=args.seed)
    train_df = model_dataset.get_training_split()
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].to_numpy()

    for name in args.model_name:
        LOG.info("Training %s model ...", name)
        params = MODEL_PARAMS[name]
        model = SalesForecastingModel(
            model_id=name,
            preprocessor=DateFeatureExtractor(),
            predictor=(CatBoostRegressor if name == "catboost" else LGBMRegressor)(**params)
        )
        model.fit(X_train, y_train)

        LOG.info("Evaluating model ...")
        test_df = model_dataset.get_test_split()
        X_test = test_df.drop(columns=[TARGET])
        y_test = test_df[TARGET].to_numpy()
        score = model.evaluate(X_test, y_test)
        LOG.info("Model evaluation score: %s", score)

        save_path = Path(args.save_loc).joinpath(f"{model.model_id}.pkl")
        LOG.info("Saving model to '%s'", save_path)
        model.serialize(save_path)


if __name__ == "__main__":
    parsed_args = parse_args()
    train_models(parsed_args)

