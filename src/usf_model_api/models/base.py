from logging import getLogger
from typing import Any, Dict, Type, Optional

import cloudpickle
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline


LOG = getLogger()


class ModelDataset:
    def get_training_split(self):
        """
        Returns the training split of the dataset
        """
        raise NotImplementedError("get_training_split() is not implemented.")

    def get_validation_split(self):
        """
        Returns the validation split of the dataset
        """
        raise NotImplementedError("get_validation_split() is not implemented.")

    def get_test_split(self):
        """
        Returns the test split of the dataset
        """
        raise NotImplementedError("get_test_split() is not implemented.")


class PredictionModel(TransformerMixin, BaseEstimator):
    def __init__(self, preprocessor: Optional[Any], predictor: BaseEstimator):
        self._preprocessor = preprocessor
        self._predictor = predictor
        self._model = Pipeline(
            [
                ('preprocessor', self.preprocessor),
                ('model', self.predictor)
            ]
        )

    def __sklearn_tags__(self):
        if self.model and hasattr(self.model, "__sklearn_tags__"):
            return self.model.__sklearn_tags__()

        return super().__sklearn_tags__()

    @property
    def preprocessor(self) -> Any:
        """
        Returns the preprocessor
        """
        return self._preprocessor

    @property
    def predictor(self) -> BaseEstimator:
        """
        Returns the predictor
        """
        return self._predictor

    @property
    def model(self) -> Pipeline:
        """
        Returns the model
        """
        return self._model

    def fit(self, X: pd.DataFrame, y: np.ndarray, **fit_params) -> "PredictionModel":
        try:
            self.model.fit(X, y=y, **fit_params)
            self.is_fitted_ = True
        except AttributeError as e:
            raise NotImplementedError("Method fit(..) is not implemented.") from e

        return self

    def predict(self, X: pd.DataFrame, **predict_params) -> np.ndarray:
        try:
            check_is_fitted(self.model)
            return self.model.predict(X, **predict_params)
        except AttributeError as e:
            raise NotImplementedError(
                f"Method predict(..) is not implemented by {type(self.model)}."
            ) from e

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> float:
        raise NotImplementedError("Subclasses must implement this method.")

    def serialize(self, dir_path):
        """
        Serializes the model using pickle

        Parameters
        ----------
        dir_path : Location where the pickle object will be stored

        Notes
        -----
        In the case where the wrapped model contains unpickleable objects, or for any other reason needs
        to implement special behavior when saving the model, users may implement their own `serialize()`
        method. The only argument passed to this method must be a `file_like: str`
        """
        try:
            cloudpickle.dump(self, dir_path)
        except TypeError:
            with open(dir_path, "wb") as f:
                cloudpickle.dump(self, f)

    @classmethod
    def deserialize(cls, dir_path) -> "PredictionModel":
        """
        Deserializes the model using pickle

        Parameters
        ----------
        dir_path : Location where the pickle object is stored

        Returns
        -------
        Unpickled object

        Notes
        -----
        In the case where the wrapped model contains unpickleable objects, or for any other reason needs
        to implement special behavior when loading the model, users may implement their own `deserialize()`
        method. The only argument passed to this method must be a `file_like: str`
        """
        try:
            return cloudpickle.load(dir_path)
        except TypeError:
            with open(dir_path, "rb") as f:
                return cloudpickle.load(f)
