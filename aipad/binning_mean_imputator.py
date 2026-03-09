import warnings
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

class PADImputator(BaseEstimator, TransformerMixin):

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


def select_true_points(data_array: np.ndarray, size_pct: float, seed: int | None = None)\
                       -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly select a percentage of non-NaN elements from a 2D array.

    Parameters
    ----------
    data_array : numpy.ndarray
        Two-dimensional input array from which elements are sampled.
        NaN values are excluded from the sampling process.
    size_pct : float
        Fraction of the non-NaN elements to sample. Must satisfy
        0 <= size_pct <= 1. For example, 0.1 selects 10% of all
        valid (non-NaN) elements.
    seed : int or None, optional
        Seed for the random number generator. If provided, sampling
        is reproducible. If None (default), a fresh, unpredictable
        random sequence is used.

    Returns
    -------
    rand_rows : numpy.ndarray
        One-dimensional array of row indices of the sampled elements.
    rand_cols : numpy.ndarray
        One-dimensional array of column indices of the sampled elements.
    true_values : numpy.ndarray
        One-dimensional array containing the sampled values from
        `data_array`. Has the same dtype as `data_array`.
    """

    if not (0 <= size_pct <= 1):
        raise ValueError("size_pct should be between 0 and 1")
    rand_generator = np.random.default_rng(seed)
    non_nan_indices = np.flatnonzero(~np.isnan(data_array))
    size = int(size_pct * non_nan_indices.size)

    # Choose random indices
    chosen_indices = rand_generator.choice(non_nan_indices, replace=False, size=size)
    rand_rows, rand_cols = np.unravel_index(chosen_indices, shape=data_array.shape)
    true_values = data_array[rand_rows, rand_cols]
    return rand_rows, rand_cols, true_values


def knn_impute(imputer, data: np.ndarray):
    # Mask all rows (temporal columns) that are all nan and do not impute those.
    mask = np.isnan(data).all(axis=1)
    x_imputed = data.copy()
    x_imputed[~mask] = imputer.transform(data[~mask])
    return x_imputed
