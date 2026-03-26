import numpy as np
import pandas as pd

from numpy.typing import NDArray
from typing import Literal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer


class PADImputer(BaseEstimator, TransformerMixin):

    def __init__(self, method: Literal["mean", "fill_average", "interp", "knn"] = "mean",
                 axis: int = 1, knn_neighbors: int = 5,
                 knn_weight: Literal["uniform", "distance"] = "uniform"):
        self.method = method
        self.axis = axis

        self._knn_neighbors = knn_neighbors
        self._knn_weigth = knn_weight

        self._strategy_map = {
            "mean": self._mean_finder,
            "fill_average": self._fill_average_finder,
            "interp": self._interp_finder,
            "knn": self._knn_finder
            }

        self.true = None
        self.reduced = None
        self.imputed = None
        self.predicted = None
        self.target = None

    def _mean_finder(self, a: NDArray, axis: Literal[0, 1]):
        imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
        if axis == 0:
            return imputer.fit_transform(a)
        elif axis == 1:     # SimpleImputer only fills with column statistics
            return imputer.fit_transform(a.T).T
        else:
            raise ValueError("Axis must either be 0 or 1")

    def _fill_average_finder(self, a: NDArray, axis: Literal[0, 1]):
        fill_av = ((pd.DataFrame(a).ffill(axis=axis) + pd.DataFrame(a).bfill(axis=axis)) / 2)
        return fill_av.interpolate(axis=axis, limit_direction="both").to_numpy()

    def _interp_finder(self, a: NDArray, axis: Literal[0, 1]):
        return (pd.DataFrame(a).interpolate(axis=axis, limit_direction="both")).to_numpy()

    def _knn_finder(self, a: NDArray, axis: Literal[0, 1]):
        imputer = KNNImputer(n_neighbors=self._knn_neighbors, weights=self._knn_weigth,
                             keep_empty_features=True)
        if axis == 0:
            return imputer.fit_transform(a)
        elif axis == 1:
            return imputer.fit_transform(a.T).T
        else:
            raise ValueError("Axis must either be 0 or 1")

    def load_data(self):
        pass

    @classmethod    # class or instance?
    def target_from_prediction(cls, true, reduced, imputed):
        pred = np.where((np.isnan(reduced)
                        & np.isfinite(true)
                        & np.meshgrid(np.isfinite(reduced).any(axis=1), np.arange(0, reduced.shape[1]),
                                        indexing="ij")[0]),
                        imputed, np.nan)
        target = np.where(np.isfinite(pred), true, np.nan)
        return target, pred

    @classmethod
    def calc_scores(cls, target, pred):
        pass

    def plot_results(self):
        pass

    def fit(self, X: NDArray, y: NDArray | None = None):
        return self

    def transform(self, X: NDArray):
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
