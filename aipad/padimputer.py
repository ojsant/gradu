import warnings
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class PADImputer(BaseEstimator, TransformerMixin):

    def __init__(self, method: Literal["mean", "fill_average", "interp"] = "mean",
                 bins: int = 8, axis: int = 1):
        self.method = method
        self.bins = bins
        self.axis = axis

        self._strategy_map = {
            "mean": self._mean_finder,
            "fill_average": self._fill_average_finder,
            "interp": self._interp_finder,
            "knn": self._knn_finder
            }

    def _mean_finder(self, a, axis):
        imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
        if axis == 0:
            return imputer.fit_transform(a)
        elif axis == 1:  # SimpleImputer only fills with column statistics
            return imputer.fit_transform(a.T).T
        else:
            raise ValueError("Axis must either be 0 or 1")

    def _fill_average_finder(self, a, axis):
        return ((pd.DataFrame(a).ffill(axis=axis)
                 + pd.DataFrame(a).bfill(axis=axis)) / 2).interpolate(axis=axis, limit_direction="both").to_numpy()

    def _interp_finder(self, a, axis):
        return (pd.DataFrame(a).interpolate(axis=axis, limit_direction="both")).to_numpy()

    def _knn_finder(self, a, axis):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        return self

    def transform(self, X: np.ndarray):
        """
        Impute missing values in pitch-angle data.

        Parameters
        ----------
        X : np.ndarray
            Input array with shape (n_rows, n_cols) where rows are timepoints.

        Returns
        -------
        arr_imputed : np.ndarray
            Array with missing values imputed.
        """
        X = np.array(X, copy=True)

        # Impute with given strategy
        if self.method not in self._strategy_map:
            raise ValueError(f"Method must be one of {list(self._strategy_map.keys())}")
        X_imputed = self._strategy_map[self.method](X, axis=self.axis)

        return X_imputed
