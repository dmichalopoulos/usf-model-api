from typing import Optional
from pathlib import Path
import pandas as pd

from usf_model_api.utils import get_logger
from usf_model_api.models.base import PredictionModel


LOG = get_logger(__name__)


class MockDatabase:
    def __init__(self, model_dir: Path):
        self.model_db = {}
        self.predictions_db = pd.DataFrame()

        if model_dir is not None:
            self.load_models(model_dir)

    def load_models(self, dir_path: Path, overwrite: bool = True):
        db = {}
        for file in dir_path.glob("*.pkl"):
            LOG.info("Loading saved model file '%s'", file)
            model = PredictionModel.deserialize(file)
            db[model.model_id] = model

        if len(db) == 0:
            LOG.warning("No models found in '%s'. You need to train at least one first.", dir_path)

        if overwrite:
            self.model_db = db
            return

        self.model_db.update(db)

    def get_model(self, model_id: str) -> Optional[PredictionModel]:
        return self.model_db.get(model_id)

    def save_predictions(self, predictions_df: pd.DataFrame):
        self.predictions_db = pd.concat(
            [self.predictions_db, predictions_df], axis=0, ignore_index=True
        )
