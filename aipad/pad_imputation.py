# TODO 9.3. move imputation related functions to padimputer and
# preprocessing related to preprocessing

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import copy
import gc

from os import PathLike
from sklearn.impute import KNNImputer
from sklearn.impute._base import _BaseImputer
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.dates import HourLocator, DateFormatter
from numpy.typing import NDArray

from anisotropy import run_SEPevent
from aipad.spacecrafts import SoloConstants, WindConstants


# TODO: there still needs to be a better way to handle shape mismatches.
def coverage_overlap(cov1: pd.DataFrame, cov2: pd.DataFrame):
    """AND mask two coverage arrays to find where they overlap. If there's length
    mismatch, shorten the longer one. 1st return is the overlapping coverage, 2nd
    is the shortened one. If no mismatch, 2nd return is the 1st coverage.

    Args:
        cov1 (np.array): 1st coverage
        cov2 (np.array): 2nd coverage

    Returns:
        array: overlapping coverage
        array: reshaped coverage
    """
    mismatch = cov2.shape[0] - cov1.shape[0]

    if mismatch > 0:
        reshaped_cov2 = cov2[:len(cov2)-mismatch]
        return cov1 & reshaped_cov2, reshaped_cov2
    else:
        reshaped_cov1 = cov1[:len(cov1)-mismatch]
        return cov2 & reshaped_cov1, reshaped_cov1


def pad_histogram(sc, I_data: np.ndarray, coverage: pd.DataFrame, bins: int) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Code partly adapted from SOLER anisotropy tools SEPEvent.overview_plot()
    method (maintained by Jan Gieseler)
    https://github.com/soler-he/sep_tools/tree/main/anisotropy commit 7567a98
    """
    intensity = copy.copy(I_data)

    # Working with "wrong" indexing since NumPy broadcasting rules are funky
    X, Y = np.meshgrid(coverage.index.values,
                       np.linspace(0.5*180/bins, (bins-0.5)*180/bins, bins))
    hist = np.zeros(np.shape(X))
    hist_counts = np.zeros(np.shape(X))

    # Extract flux wherever the coverage is finite,
    # fill with those values between the min and max coverage,
    # and sum every sector's histogram together.
    # Could probably be written better.
    for i, sector in enumerate(sc.sectors):
        intensity_per_sector = intensity[:, i]
        cov_arr = coverage[sector].to_numpy()
        cov_finite = coverage[sector].notna().to_numpy()
        av_flux = np.where(cov_finite[:, 1], intensity_per_sector, np.nan)
        new_hist = np.where((Y > cov_arr[:, 0]) & (Y < cov_arr[:, 2]), av_flux, 0)
        hist = hist + new_hist
        hist_counts = hist_counts + np.where(new_hist > 0, 1, 0)   # Overlapping bins as averages

    hist = hist / hist_counts
    hist = np.where(hist > 0, hist, np.nan)

    return X.T, Y.T, hist.T      # Take the transpose to get correct indices


def convert_to_bool_coverage(cov, sc, bins=8):
    directions = sc.sectors

    X, Y = np.meshgrid(cov.index.values,
                       np.linspace(0.5*180/bins, (bins-0.5)*180/bins, bins),
                       indexing="ij")
    cov_arr = np.zeros_like(Y, dtype=np.bool_)

    for direction in directions:
        # replace missing values with -1 to exclude from comparison in the loop
        dataf = cov[direction].mask(cov[direction].isna(), -1)
        for index, data in dataf.reset_index().iterrows():
            covered = np.ma.masked_inside(Y[index], data["min"], data["max"])
            cov_arr[index] = cov_arr[index] | covered.mask

    return X, Y, cov_arr


def load_random_file(path: PathLike):
    r_file = np.random.choice(os.listdir(path)).tolist()
    return np.load(path / r_file)


def load_wind_event(*args, remove_peaks=False, n_lim=2, **kwargs):
    wind_event = run_SEPevent(*args, **kwargs)
    if remove_peaks:
        wind_event.wind_peak_removal(n_lim=n_lim)
    return wind_event


def make_train_test(*args, n=3, shift=0) -> list:
    """Form a train-test split by taking every nth item to the test set. Shift parameter controls
    which of the n samples is taken.

    Args:
        n (int, optional): how manyth sample is taken to the test set, default=3
        shift (int, optional): add to modulo operation for different splits, default=0
    """
    rets = []

    n_tot = len(args[0])
    for i, arg in enumerate(args):
        if len(arg) != n_tot:
            raise ValueError("Arguments not equal-length")
        trains = []
        tests = []
        for j, item in enumerate(arg):
            if (j + shift) % 3 == 0:
                tests.append(item)
            else:
                trains.append(item)
        rets.append(trains)
        rets.append(tests)
    return rets


def form_test_matrices(model: _BaseImputer, X_test: NDArray,
                       y_test: np.ndarray) -> list:
    true = X_test
    test = y_test
    reduced = np.where(np.isfinite(test), true, np.nan)
    if isinstance(model, KNNImputer):
        reduced_reshaped = reduced.reshape((144, 900))
        pred_full_reshaped = model.transform(reduced_reshaped)
        pred_full = pred_full_reshaped.reshape((720, 180))

    pred = np.where((np.isnan(reduced)
                    & np.isfinite(true)
                    & np.meshgrid(np.isfinite(reduced).any(axis=1), np.arange(0, reduced.shape[1]),
                                  indexing="ij")[0]),
                    pred_full, np.nan)
    target = np.where(np.isfinite(pred), true, np.nan)

    return [true, reduced, pred_full, pred, target]


def target_from_prediction(true, reduced, imputed):
    pred = np.where((np.isnan(reduced)
                    & np.isfinite(true)
                    & np.meshgrid(np.isfinite(reduced).any(axis=1), np.arange(0, reduced.shape[1]),
                                  indexing="ij")[0]),
                    imputed, np.nan)
    target = np.where(np.isfinite(pred), true, np.nan)
    return target, pred

