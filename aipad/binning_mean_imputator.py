import warnings
from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class BinningImputator(BaseEstimator, TransformerMixin):

    def __init__(self, method: Literal["angular", "temporal"] = "angular", bins: int = 8):
        self.method = method
        self.bins = bins

        self._strategy_map = {
            "angular": self._angular_finder,
            "temporal": self._temporal_finder
            }

    def _angular_finder(self, bin_means: np.ndarray) -> np.ndarray:
        """
        Fully vectorized angular nearest-bin fill.
        For each row, fill NaN bins using the nearest non-empty bin in that row.
        """
        n_rows, n_bins = bin_means.shape
        filled = bin_means.copy()

        # Mask of valid bins
        valid = ~np.isnan(bin_means)
        # Rows with at least one valid bin
        rows_with_data = np.where(valid.any(axis=1))[0]
        # Precompute angular distance matrix (n_bins x n_bins)
        bin_idx = np.arange(n_bins)
        dist_matrix = np.abs(bin_idx[:, None] - bin_idx[None, :])  # dist[i,j] = |i-j|

        for r in rows_with_data:
            row_valid = valid[r]
            if row_valid.all():
                continue  # no NaNs in this row
            nan_bins = np.where(~row_valid)[0]   # bins that are NaN
            valid_bins = np.where(row_valid)[0]  # bins that have data
            # Compute distances from each NaN bin to valid bins
            distances = dist_matrix[nan_bins[:, None], valid_bins[None, :]]
            # Find nearest valid bin for each NaN
            nearest_idx = valid_bins[np.argmin(distances, axis=1)]
            # Fill the NaNs
            filled[r, nan_bins] = filled[r, nearest_idx]

        return filled

    def _temporal_finder(self, bin_means: np.ndarray) -> np.ndarray:
        """
        Fully vectorized temporal nearest-bin fill along rows (time) per bin.
        Forward-fill missing values using the last valid row for each bin.
        """
        n_rows, n_bins = bin_means.shape

        # Mask of valid values
        valid_mask = ~np.isnan(bin_means)
        # Create an array of indices for each row
        row_idx = np.arange(n_rows)[:, None]  # shape (n_rows, 1)
        # Replace NaN indices with -1 for cumulative max trick
        last_valid_idx = np.where(valid_mask, row_idx, -1)
        # Forward-fill last valid index along rows
        last_valid_idx_ff = np.maximum.accumulate(last_valid_idx, axis=0)
        # Build the filled array by indexing into bin_means
        # For rows where there is no previous valid (-1), remain NaN
        filled = np.where(
            last_valid_idx_ff >= 0,
            bin_means[last_valid_idx_ff, np.arange(n_bins)],
            np.nan
        )

        return filled

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        return self

    def transform(self, X: np.ndarray):
        """
        Impute missing values in pitch-angle data using bin means.

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
        n_rows, n_cols = X.shape

        # Step 1 — Split columns into bins
        col_bins = np.array_split(np.arange(n_cols), self.bins)

        # Step 2 — Compute initial bin means per row
        bin_means = np.full((n_rows, self.bins), np.nan)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            for i, cols in enumerate(col_bins):
                bin_means[:, i] = np.nanmean(X[:, cols], axis=1)

        # Step 3 — Fill empty bins
        if self.method not in self._strategy_map:
            raise ValueError(f"Method must be one of {list(self._strategy_map.keys())}")
        bin_means_filled = self._strategy_map[self.method](bin_means)

        # Step 4 — Fill the original array using bin_means_filled
        arr_imputed = X.copy()
        for i, cols in enumerate(col_bins):
            arr_imputed[:, cols] = np.where(
                np.isnan(arr_imputed[:, cols]),
                bin_means_filled[:, i][:, None],
                arr_imputed[:, cols]
            )

        return arr_imputed


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
