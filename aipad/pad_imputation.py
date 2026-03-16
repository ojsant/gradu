# TODO 9.3. move imputation related functions to padimputer and
# preprocessing related to preprocessing

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import copy

from pathlib import Path
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.impute._base import _BaseImputer
from sklearn.metrics import r2_score, root_mean_squared_error
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
    sectors = sc.sectors

    # Working with "wrong" indexing since NumPy broadcasting rules are funky
    X, Y = np.meshgrid(coverage.index.values,
                       np.linspace(0.5*180/bins, (bins-0.5)*180/bins, bins))
    hist = np.zeros(np.shape(X))
    hist_counts = np.zeros(np.shape(X))

    # Extract flux wherever the coverage is finite,
    # fill with those values between the min and max coverage,
    # and sum every sector's histogram together.
    # Could probably be written better.
    for i, direction in enumerate(sectors):
        intensity_per_sector = intensity[:, i]
        cov_arr = coverage[direction].to_numpy()
        cov_finite = coverage[direction].notna().to_numpy()
        av_flux = np.where(cov_finite[:, 1], intensity_per_sector, np.nan)
        new_hist = np.where((Y > cov_arr[:, 0]) & (Y < cov_arr[:, 2]), av_flux, 0)
        hist = hist + new_hist
        hist_counts = hist_counts + np.where(new_hist > 0, 1, 0)   # Overlapping bins as averages

    hist = hist / hist_counts
    hist = np.where(hist > 0, hist, np.nan)

    return X.T, Y.T, hist.T


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


def induce_missingness(cov: pd.DataFrame, p: float, seed: int = 123) \
        -> tuple[pd.DataFrame, np.ndarray]:
    """Set a percentage of given coverage as missing. Returns a copy of the coverage array without
    the randomly picked values and the missingness mask.

    Args:
        cov (pd.DataFrame): coverage with columns ("min", "center", "max") for each sector
        p (float): proportion of data to set as missing
    """
    random = np.random.default_rng(seed)
    mask = random.choice([False, True], size=(cov.shape[0], cov.shape[1] // 3), p=[1-p, p])
    mask = np.repeat(mask, 3, axis=1)
    masked_cov = cov.where(~mask, np.nan)    # DataFrame.where() replaces where condition is False
    return masked_cov, mask


def load_random_file(path: Path):
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
    ind = np.arange(0, n_tot, 1)
    for arg in args:
        if len(arg) != n_tot:
            raise ValueError("Arguments not equal-length")

        test = arg[(ind + shift) % 3 == 0]
        train = arg[(ind + shift) % 3 != 0]
        rets.append(train)
        rets.append(test)

    return rets


def form_test_matrices(model: _BaseImputer, X_test: np.ndarray,
                       y_test: np.ndarray, transpose=False) -> list:
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


def calculate_scores(target: np.ndarray, pred: np.ndarray, score: str) -> float:
    """Calculate either coefficient of determination or mean square error
    as score between prediction and target.

    Args:
        model (_BaseImputer): KNNImputer, SimpleImputer, IterativeImputer
        true (np.ndarray): target values
        pred (np.ndarray): predicted values
        score (str): "r2" or "rmse"

    Returns:
        float
    """

    pred = pred.ravel()
    target = target.ravel()

    if np.any(np.isfinite(target)) and np.any(np.isfinite(pred)):
        if score == "r2":
            return r2_score(
                target[np.isfinite(target)], pred[np.isfinite(pred)]
                )

        elif score == "rmse":
            return root_mean_squared_error(
                target[np.isfinite(target)], pred[np.isfinite(pred)]
                )

        else:
            raise ValueError("Give a valid score!")
    else:
        return 0.0


def plot_results(model: _BaseImputer, res: list, score: str, intensities: np.ndarray,
                 sc: WindConstants, cov_sc: SoloConstants,
                 save_plot=True, save_path=None) -> None:

    true, reduced, pred_full, pred, target = res

    # Calculate score over time
    scores = []
    for tr, pr in zip(target, pred):
        pred_score = calculate_scores(tr, pr, score)
        scores.append(pred_score)

    # Score over whole 12 hrs
    total_score = calculate_scores(target, pred, score)

    true_miss_percent = np.sum(np.where(np.isnan(true), 1, 0)) \
        / (true.shape[0] * true.shape[1]) * 100
    reduced_miss_percent = np.sum(np.where(np.isnan(reduced), 1, 0)) \
        / (reduced.shape[0] * reduced.shape[1]) * 100

    X, Y = np.meshgrid(np.arange(0, true.shape[0]), np.arange(0, true.shape[1]), indexing="ij")

    norm = LogNorm(np.nanmin(true), np.nanmax(true))
    diff_norm = LogNorm(1e-3, 1e3)

    fig, axs = plt.subplots(nrows=8, figsize=(16, 32), sharex=True)

    for i in range(8):
        axs[0].plot(intensities[:, i], label=sc.sectors[i])
    axs[0].set_title("Intensities")
    axs[0].set_yscale("log")
    axs[0].legend(loc="upper right")

    axs[1].pcolormesh(X, Y, true, norm=norm, cmap="inferno")
    axs[1].set_title(f"WIND PAD, total {true_miss_percent:.2f} % missing")

    axs[2].pcolormesh(X, Y, reduced, norm=norm, cmap="inferno")
    axs[2].set_title(f"WIND PAD w/ reduction, {reduced_miss_percent:.2f} % missing")

    axs[3].pcolormesh(X, Y, pred_full, norm=norm, cmap="inferno")
    if isinstance(model, KNNImputer):
        axs[3].set_title(f"k-NN imputed values, k = {model.get_params()["n_neighbors"]}")
    elif isinstance(model, SimpleImputer):
        axs[3].set_title("Mean imputed values")

    axs[4].pcolormesh(X, Y, pred, norm=norm, cmap="inferno")
    axs[4].set_title("Predicted values")

    axs[5].pcolormesh(X, Y, target, norm=norm, cmap="inferno")
    axs[5].set_title("Target values")

    diff = np.abs(target - pred)

    mesh = axs[6].pcolormesh(X, Y, diff, norm=diff_norm)
    axs[6].set_title("Absolute difference")
    axins = inset_axes(axs[6], width="100%", height="100%", loc="center",
                       bbox_to_anchor=(1.01, 0, 0.03, 1), bbox_transform=axs[6].transAxes,
                       borderpad=0.2)
    fig.colorbar(mesh, cax=axins, orientation="vertical")

    axs[7].plot(np.arange(0, 720, 1), scores)

    axs[7].set_title(f"RMSE score: (total = {total_score})")

    if save_plot:
        plt.savefig(save_path)
        plt.close()

    else:
        plt.show()


def save_results(model, res, meta, scorer, path) -> pd.DataFrame:

    reduced, pred, target = res[1], res[3], res[4]
    score = calculate_scores(target, pred, scorer)
    reduced_miss_percent = np.sum(np.where(np.isnan(reduced), 1, 0)) \
        / (reduced.shape[0] * reduced.shape[1]) * 100
    cols = ["event_dt", "miss_percent", "score", "scorer", "model_params"]

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=cols)

    row = [pd.to_datetime(meta["onset_datetime"], format="%Y%m%d-%H%M%S"),
           reduced_miss_percent, score, scorer, (model.n_neighbors, model.weights)]

    df.loc[-1] = row
    df.index = df.index + 1
    df = df.sort_index()

    df.to_csv(path, index=False)

    return df
