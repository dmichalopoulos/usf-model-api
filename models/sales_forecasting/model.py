from typing import Any, Type

import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_absolute_error

from usf_model_api.models.base import PredictionModel


class SalesForecastingModel(PredictionModel):
    def __init__(self, preprocessor: Any, predictor: BaseEstimator):
        super().__init__(preprocessor=preprocessor, predictor=predictor)

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> float:
        check_is_fitted(self.model)

        predictions = self.model.predict(X)
        mae = mean_absolute_error(y_true=y, y_pred=predictions)

        return mae
