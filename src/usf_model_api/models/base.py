from logging import getLogger
from typing import Any, Dict, Union, Optional, Type

import cloudpickle
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


LOG = getLogger()


# NOTE: Per the sklearn docs, always make sure that all Mixin classes come before BaseEstimator in
# the inheritance order.
class USFModel(TransformerMixin, BaseEstimator):
    """
    A subclass of :py:class:`sklearn.base.BaseEstimator` and :py:class:`sklearn.base.TransformerMixin`,
    which serves as a base class for deployed models. It allows for lots of customization, including
    how a model instance is serialized/deserialized.
    """

    def __init__(self, *model_args, model_cls: Type = None, model_id: str = None, **model_kwargs):
        """
        Initializer.

        Notes
        -----
         * ``model_cls`` should be a class that adheres to ``sklearn`` standards for estimators. In particular,
           a ``model_cls`` instance should have an ``estimator_type`` tag value that will allow model scoring
           functions to correctly infer whether it's a classifier, regressor, or something else. See
           `here <https://scikit-learn.org/stable/modules/generated/sklearn.utils.Tags.html>`_ for more details.

        Parameters
        ----------
        model_args : Arguments required by the wrapped model
        model_cls : The class of the wrapped model
        model_id: The model id that identifies the model in the AI Factory registry
        model_kwargs: Keyword arguments accepted by the wrapped model
        """
        self._model_id = None

        if model_id is None:
            # generate model_id, could be from kfp
            pass
        else:
            self._model_id = model_id

        self._model = None
        self.model_cls = model_cls
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        super().__init__()

    def __sklearn_tags__(self):
        if self.model and hasattr(self.model, "__sklearn_tags__"):
            return self.model.__sklearn_tags__()

        return super().__sklearn_tags__()

    @property
    def model_id(self):
        """
        Returns the id currently associated with the model.

        Notes
        -----
        Because the model id can be provided arbitrarily to the constructor, it should not be assumed that
        the id returned by this method is a valid AI Factory model id.
        """
        return self._model_id

    @property
    def model(self):
        """
        Returns the wrapped model
        """
        return self._model

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **fit_params,
    ) -> "USFModel":
        """
        Invokes the fit method of the wrapped model

        Parameters
        ----------
        X : Training data features, each row is a data point and each column a feature
        y : Labels for each data point for supervised training
        fit_params: Fitting parameters required by the wrapped model
        """
        self._model = self.model_cls(*self.model_args, **self.model_kwargs)

        try:
            self._model.fit(X, y=y, **fit_params)
        except AttributeError as e:
            raise NotImplementedError("Method fit(..) is not implemented.") from e
        return self

    def partial_fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **fit_params,
    ) -> "USFModel":
        """
        Invokes the partial_fit method of the wrapped model

        Parameters
        ----------
        X : Training data features, each row is a data point and each column a feature
        y : Labels for each data point for supervised training
        fit_params: Fitting parameters required by the wrapped model
        """
        try:
            self._model.partial_fit(X, y=y, **fit_params)
        except AttributeError as e:
            raise NotImplementedError("Method partial_fit(..) is not implemented.") from e

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Invokes the transform method of the wrapped model

        Parameters
        ----------
        X : Data to transform, each row is a data point and each column a feature

        Returns
        -------
        Transformed input data X
        """
        if self._model is None:
            raise RuntimeError("transform called before fit(..)")

        try:
            X_tf = self._model.transform(X)

            if isinstance(X, pd.DataFrame) and X_tf.shape == X.shape:
                X_tf = pd.DataFrame(data=X_tf, columns=X.columns)

            return X_tf
        except AttributeError as e:
            raise NotImplementedError("Method transform(..) is not implemented.") from e

    def predict(self, X, **predict_params) -> Union[np.ndarray, OptimizationResponse]:
        """
        Invokes the predict method of the wrapped model

        Parameters
        ----------
        X : Data to predict from, each row is a data point and each column a feature
        predict_params: Parameters required by the predict method of the wrapped model

        Returns
        -------
        Predictions for input data X
        """
        if self._model is None:
            raise RuntimeError("Method predict(..) called before fit(..).")

        try:
            return self._model.predict(X, **predict_params)
        except AttributeError as e:
            raise NotImplementedError(
                f"Method predict(..) is not implemented by {type(self._model)}, {type(self.model_cls)}."
            ) from e

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray], **predict_params
    ) -> Union[np.ndarray, OptimizationResponse]:
        """
        Invokes the predict_proba method of the wrapped model

        Parameters
        ----------
        X : Data to predict from, each row is a data point and each column a feature
        predict_params: Parameters required by the predict method of the wrapped model

        Returns
        -------
        Probability predictions for input data X
        """
        if self._model is None:
            raise RuntimeError("Method predict_proba(..) called before fit(..).")

        try:
            return self._model.predict_proba(X, **predict_params)
        except AttributeError as e:
            raise NotImplementedError("Method predict_proba(..) is not implemented.") from e

    def explain(self, **explain_params):
        """
        Invokes the explain method of the wrapped model

        Parameters
        ----------
        explain_params: Explain parameters required by the wrapped model
        """
        if self._model is None:
            raise RuntimeError("Method explain(..) called before fit(..).")

        try:
            return self._model.explain(**explain_params)
        except AttributeError as e:
            raise NotImplementedError("explain is not implemented") from e

    def explain_predictions(self, X: Union[pd.DataFrame, np.ndarray], **explain_params):
        """
        Invokes the explain_predictions method of the wrapped model

        Parameters
        ----------
        X : Data to predict from, each row is a data point and each column a feature
        explain_params: Explain parameters required by the wrapped model
        """
        if self._model is None:
            raise RuntimeError("Method explain_prediction(..) called before fit")

        try:
            return self._model.explain_predictions(X, **explain_params)
        except AttributeError as e:
            raise NotImplementedError("Method explain_predictions(..) is not implemented.") from e

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
    def deserialize(cls, dir_path) -> "USFModel":
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
