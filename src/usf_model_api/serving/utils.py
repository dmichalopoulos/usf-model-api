from typing import Optional, Dict, Any
from pathlib import Path
import pandas as pd

from usf_model_api.utils import get_logger
from usf_model_api.models.base import PredictionModel


LOG = get_logger(__name__)


class MockDatabase:
    """
    A mock database class for managing in-memory model objects and predictions. The ``model_db``
    is meant to simulate a very simple in-memory model 'registry'.

    Attributes
    ----------
    model_db : dict
        A dictionary to store models with their model IDs as keys.
    predictions_db : pd.DataFrame
        A DataFrame to store predictions.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initializes the MockDatabase with a directory containing model files.

        Parameters
        ----------
        model_dir : Path
            The directory containing model files. If specified, models will be loaded from this
            directory. Otherwise, an empty database will be created, which can be populated at any
            time by calling ``load_models()``.
        """
        self._model_db = {}
        self._predictions_db = pd.DataFrame()

        if model_dir is not None:
            self.load_models(model_dir)

    @property
    def model_db(self) -> Dict[str, Any]:
        """
        Returns the model database, which is a simple dictionary.
        """
        return self._model_db

    @property
    def predictions_db(self) -> pd.DataFrame:
        """
        Returns the predictions database, which is a pandas DataFrame.
        """
        return self._predictions_db

    def load_models(self, dir_path: Path, overwrite: bool = True):
        """
        Loads models from the specified directory.

        Parameters
        ----------
        dir_path : Path
            The directory path to load models from.
        overwrite : bool, optional
            Whether to overwrite the existing models in the database (default is True).
        """
        db = {}
        for file in dir_path.glob("*.pkl"):
            LOG.info("Loading saved model file '%s'", file)
            model = PredictionModel.deserialize(file)
            db[model.model_id] = model

        if len(db) == 0:
            LOG.warning("No models found in '%s'. You need to train at least one first.", dir_path)

        if overwrite:
            self._model_db = db
            return

        self.model_db.update(db)

    def get_model(self, model_id: str) -> Optional[PredictionModel]:
        """
        Retrieves a model by its ID. Returns None if the model is not found.

        Parameters
        ----------
        model_id : str
            The unique identifier of the model.

        Returns
        -------
        Optional[PredictionModel]
            The model corresponding to the given ID, or None if not found.
        """
        return self.model_db.get(model_id)

    def save_predictions(self, predictions_df: pd.DataFrame):
        """
        Saves predictions to the predictions database.

        Parameters
        ----------
        predictions_df : pd.DataFrame
            The DataFrame containing predictions to be saved.
        """
        self._predictions_db = pd.concat(
            [self.predictions_db, predictions_df], axis=0, ignore_index=True
        )
