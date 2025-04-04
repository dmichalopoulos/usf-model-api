from logging import getLogger
import pandas as pd

from models.sales_forecasting.data import SalesDataset
from models.sales_forecasting.model import SalesForecastingModel


LOG = getLogger(__name__)


MODEL_PARAMS = {
    "nthread": -1, #10,
     "max_depth": 5,
    #         "max_depth": 9,
    "task": "train",
    "boosting_type": "gbdt",
    "objective": "regression_l1",
    "metric": "mape", # this is abs(a-e)/max(1,a)
    #         "num_leaves": 39,
    "num_leaves": 64,
    "learning_rate": 0.2,
    "feature_fraction": 0.9,
    #         "feature_fraction": 0.8108472661400657,
    #         "bagging_fraction": 0.9837558288375402,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 3.097758978478437,
    "lambda_l2": 2.9482537987198496,
    #       "lambda_l1": 0.06,
    #       "lambda_l2": 0.1,
    "verbose": 1,
    "min_child_weight": 6.996211413900573,
    "min_split_gain": 0.037310344962162616,
}

TRAIN_DATA_LOC = "../../downloads/train.csv"
SAVE_MODEL_LOC = "../../service/assets/sales_forecast_model.pkl"
TRAIN_PCT = 0.8
TARGET = "sales"


if __name__ is "__main__":
    LOG.info("Loading training data from %s", TRAIN_DATA_LOC)
    data = pd.read_csv(TRAIN_DATA_LOC)

    # Create splits
    LOG.info("Creating training and test splits")
    model_dataset = SalesDataset(data, train_pct=TRAIN_PCT)
    train_df = model_dataset.get_training_split()
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].to_numpy()

    # Train model
    LOG.info("Training model ...")
    model = SalesForecastingModel(MODEL_PARAMS)
    model.fit(X_train, y_train)

    # Evaluate model
    LOG.info("Evaluating model ...")
    test_df = model_dataset.get_test_split()
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET].to_numpy()
    score = model.evaluate(X_test, y_test)
    LOG.info("Model evaluation score: %s", score)

    LOG.info("Saving model to %s", SAVE_MODEL_LOC)
    model.serialize(SAVE_MODEL_LOC)
