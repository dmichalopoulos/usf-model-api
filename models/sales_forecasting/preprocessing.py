"""
Write an sklearn-style transformer that takes a pandas DataFrame as input. The model features are as
follows:
 * date
 * store
 * item

The transformer should apply basic feature engineering on the date column. Specifically, it should extract
day, month, and year features. Because there are other features in the DataFrame, your solution
should utilize the `ColumnTransformer` class from sklearn to apply the transformation only to the date column.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn import set_config

set_config(transform_output="pandas")


class DateFeatureExtractor(TransformerMixin, BaseEstimator):
    def fit(self, X: pd.DataFrame, y=None) -> "DateFeatureExtractor":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X['day'] = X['date'].dt.day
        X['month'] = X['date'].dt.month
        X['year'] = X['date'].dt.year
        # Drop the original date column
        return X.drop(columns=['date'])


preprocessor = ColumnTransformer(
    transformers=[
        ('date_extractor', DateFeatureExtractor(), 'date')
    ],
    remainder='passthrough'
)
