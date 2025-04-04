from typing import Any, Optional
from pathlib import Path

import cloudpickle
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline

from usf_model_api.utils import get_logger

LOG = get_logger(__name__)


class ModelDataset:
    """
    Base class for a dataset used in model training and evaluation.
    """

    def get_training_split(self):
        """
        Returns the training split of the dataset.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError("get_training_split() is not implemented.")

    def get_validation_split(self):
        """
        Returns the validation split of the dataset.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError("get_validation_split() is not implemented.")

    def get_test_split(self):
        """
        Returns the test split of the dataset.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError("get_test_split() is not implemented.")


class PredictionModel(TransformerMixin, BaseEstimator):
    """
    A class representing a prediction model that includes a preprocessor and a predictor.

    Attributes
    ----------
    model_id : str
        The unique identifier for the model.
    preprocessor : Any
        The preprocessor used in the model pipeline.
    predictor : BaseEstimator
        The predictor used in the model pipeline.
    model : Pipeline
        The scikit-learn pipeline combining the preprocessor and predictor.
    """

    def __init__(self, model_id: str, preprocessor: Optional[Any], predictor: BaseEstimator):
        """
        Initializes the PredictionModel with a model ID, preprocessor, and predictor.

        Parameters
        ----------
        model_id : str
            The unique identifier for the model.
        preprocessor : Optional[Any]
            The preprocessor used in the model pipeline.
        predictor : BaseEstimator
            The predictor used in the model pipeline.
        """
        self._model_id = model_id
        self._preprocessor = preprocessor
        self._predictor = predictor
        self._model = Pipeline(
            [
                ('preprocessor', self.preprocessor),
                ('model', self.predictor)
            ]
        )

    def __sklearn_tags__(self):
        """
        Returns the scikit-learn tags for the model.

        Returns
        -------
        dict
            The scikit-learn tags.
        """
        if self.model and hasattr(self.model, "__sklearn_tags__"):
            return self.model.__sklearn_tags__()

        return super().__sklearn_tags__()

    @property
    def model_id(self) -> str:
        """
        Returns the model ID.

        Returns
        -------
        str
            The unique identifier for the model.
        """
        return self._model_id

    @property
    def preprocessor(self) -> Any:
        """
        Returns the preprocessor.

        Returns
        -------
        Any
            The preprocessor used in the model pipeline.
        """
        return self._preprocessor

    @property
    def predictor(self) -> BaseEstimator:
        """
        Returns the predictor.

        Returns
        -------
        BaseEstimator
            The predictor used in the model pipeline.
        """
        return self._predictor

    @property
    def model(self) -> Pipeline:
        """
        Returns the model pipeline.

        Returns
        -------
        Pipeline
            The scikit-learn pipeline combining the preprocessor and predictor.
        """
        return self._model

    def fit(self, X: pd.DataFrame, y: np.ndarray, **fit_params) -> "PredictionModel":
        """
        Fits the model to the training data.

        Parameters
        ----------
        X : pd.DataFrame
            The training data.
        y : np.ndarray
            The target values.
        **fit_params : dict
            Additional parameters to pass to the fit method.

        Returns
        -------
        PredictionModel
            The fitted model.
        """
        try:
            self.model.fit(X, y=y, **fit_params)
            self.is_fitted_ = True
        except AttributeError as e:
            raise NotImplementedError("Method fit(..) is not implemented.") from e

        return self

    def predict(self, X: pd.DataFrame, **predict_params) -> np.ndarray:
        """
        Makes predictions using the fitted model.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        **predict_params : dict
            Additional parameters to pass to the predict method.

        Returns
        -------
        np.ndarray
            The predicted values.
        """
        try:
            check_is_fitted(self.model)
            return self.model.predict(X, **predict_params)
        except AttributeError as e:
            raise NotImplementedError(
                f"Method predict(..) is not implemented by {type(self.model)}."
            ) from e

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> float:
        """
        Evaluates the model on the given data.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : np.ndarray
            The target values.

        Returns
        -------
        float
            The evaluation metric.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def serialize(self, file_path: str | Path):
        """
        Serializes the model using pickle.

        Parameters
        ----------
        file_path : str
            Location where the pickle object will be stored.

        Notes
        -----
        In the case where the wrapped model contains unpickleable objects, or for any other reason needs
        to implement special behavior when saving the model, users may implement their own `serialize()`
        method. The only argument passed to this method must be a `file_like: str`.
        """
        try:
            cloudpickle.dump(self, file_path)
        except TypeError:
            with open(file_path, "wb") as f:
                cloudpickle.dump(self, f)

    @classmethod
    def deserialize(cls, file_path: str | Path) -> "PredictionModel":
        """
        Deserializes the model using pickle.

        Parameters
        ----------
        file_path : str
            Location where the pickle object is stored.

        Returns
        -------
        PredictionModel
            The unpickled model object.

        Notes
        -----
        In the case where the wrapped model contains unpickleable objects, or for any other reason needs
        to implement special behavior when loading the model, users may implement their own `deserialize()`
        method. The only argument passed to this method must be a `file_like: str`.
        """
        try:
            return cloudpickle.load(file_path)
        except TypeError:
            with open(file_path, "rb") as f:
                return cloudpickle.load(f)
