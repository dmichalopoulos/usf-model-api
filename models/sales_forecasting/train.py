import argparse
from pathlib import Path
import logging
from typing import Dict, Any

import pandas as pd
import numpy as np

from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_absolute_percentage_error

from catboost import CatBoostRegressor

from usf_model_api.models.base import PredictionModel, ModelDataset


set_config(transform_output="pandas")
logging.basicConfig()

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


# Defaults (can be overridden by command line args)
DEFAULT_SCRIPT_PATH = Path(__file__).resolve().parent
DEFAULT_TRAIN_DATA_LOC = DEFAULT_SCRIPT_PATH.joinpath("../..", "downloads", "train.csv")
# DEFAULT_SAVE_MODEL_LOC = DEFAULT_SCRIPT_PATH.joinpath("../..", "service", "assets")
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
MODEL_PARAMS = {
    "verbose": 100,
    "loss_function": "RMSE",
    "learning_rate": 0.4, #0.3,
    "depth": 5,
    # "l2_leaf_reg": 30, #2, #5, #3, #5, #2.5, #5, #10, #0.1,
    "n_estimators": 500, #3_000,
    "boost_from_average": True,
    "bootstrap_type": "MVS", #"No",
    "subsample": 0.8,
    "eval_metric": "MAPE",
}


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
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Location of the training data CSV file.",
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


if __name__ == "__main__":
    args = parse_args()

    LOG.info("Loading training data from %s", args.data_loc)
    data = pd.read_csv(args.data_loc)

    LOG.info("Creating training and test splits")
    model_dataset = SalesDataset(data, train_pct=args.train_pct, random_seed=args.seed)
    train_df = model_dataset.get_training_split()
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].to_numpy()

    LOG.info("Training model ...")
    model = SalesForecastingModel(
        model_id=args.model_name,
        preprocessor=DateFeatureExtractor(),
        predictor=CatBoostRegressor(**MODEL_PARAMS)
    )
    model.fit(X_train, y_train)

    LOG.info("Evaluating model ...")
    test_df = model_dataset.get_test_split()
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET].to_numpy()
    score = model.evaluate(X_test, y_test)
    LOG.info("Model evaluation score: %s", score)

    # LOG.info("Saving model to %s", os.path.join(args.save_loc, args.model_name))
    save_path = Path(args.save_loc).joinpath(f"{model.model_id}.pkl")
    LOG.info("Saving model to '%s'", save_path)
    model.serialize(save_path)
