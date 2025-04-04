import pandas as pd
from sklearn.model_selection import train_test_split

from usf_model_api.models.base import ModelDataset


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
