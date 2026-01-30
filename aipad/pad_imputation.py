import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import copy
import traceback

from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.impute._base import _BaseImputer
from sklearn.metrics import r2_score, mean_squared_error
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


def pad_histogram(sc, I_data: np.ndarray, coverage: pd.DataFrame, bin_width: int) -> np.ndarray:
    """
    Code partly adapted from SOLER anisotropy tools SEPEvent.overview_plot()
    method (maintained by Jan Gieseler)
    https://github.com/soler-he/sep_tools/tree/main/anisotropy commit 7567a98
    """
    intensity = copy.copy(I_data)
    sectors = sc.sectors

    X, Y = np.meshgrid(coverage.index.values, np.linspace(0, 180, int(180 / bin_width)))
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
        new_hist = np.where(((Y > cov_arr[:, 0]) & (Y < cov_arr[:, 2])), av_flux, 0)
        hist = hist + new_hist
        hist_counts = hist_counts + np.where(new_hist > 0, 1, 0)   # Overlapping bins as averages

    hist = hist / hist_counts
    hist = np.where(hist > 0, hist, np.nan)

    return hist.T       # The indices flip somewhere?? Take the transpose for (time, angle) shape


def convert_to_bool_coverage(cov, sc, bin_width_deg=1):
    directions = sc.sectors

    X, Y = np.meshgrid(cov.index.values,
                       np.linspace(0, 180, int(180 / bin_width_deg)),
                       indexing="ij")
    cov_arr = np.zeros_like(Y, dtype=np.bool_)

    for direction in directions:
        # replace missing values with -1 to exclude from comparison in the loop
        dataf = cov[direction].mask(cov[direction].isna(), -1)
        for index, data in dataf.reset_index().iterrows():
            covered = np.ma.masked_inside(Y[index], data["min"], data["max"])
            cov_arr[index] = cov_arr[index] | covered.mask

    return X, Y, cov_arr


def load_random_file(path: Path):
    r_file = np.random.choice(os.listdir(path)).tolist()
    return np.load(path / r_file)


def load_wind_event(*args, remove_peaks=False, n_lim=2, **kwargs):
    wind_event = run_SEPevent(*args, **kwargs)
    if remove_peaks:
        wind_event.wind_peak_removal(n_lim=n_lim)
    return wind_event


def read_npzs(load_path):
    hist_list = []
    reduced_hist_list = []
    intensity_list = []
    metadata = []

    # TODO handle metadata

    for file in load_path.iterdir():
        try:
            npz = np.load(file)
            hist = npz["full"]
            reduced_hist = npz["reduced"]
            intensity = npz["intensity_data"]
            hist_list.append(hist)
            reduced_hist_list.append(reduced_hist)
            intensity_list.append(intensity)
            metadata.append(str(file))

        except ValueError:
            print(traceback.format_exc())
            continue

    hist_arr = np.array(hist_list)
    reduced_hist_arr = np.array(reduced_hist_list)
    intensity_arr = np.array(intensity_list)
    metadata_arr = np.array(metadata)

    return hist_arr, reduced_hist_arr, intensity_arr, metadata_arr


def make_train_test(*args, n=3):
    """Form a train-test split by taking every nth item to the test set.

    Args:
        n (int, optional):
    """
    rets = []
    n_tot = len(args[0])
    ind = np.arange(0, n_tot, 1)
    for arg in args:
        if len(arg) != n_tot:
            raise ValueError("Arguments not equal-length")

        test = arg[ind % 3 == 0]
        train = arg[ind % 3 != 0]
        rets.append(train)
        rets.append(test)

    return rets


def form_test_matrices(model: _BaseImputer, X_test: np.ndarray,
                       y_test: np.ndarray, transpose=False) -> list:
    true = X_test
    test = y_test
    reduced = np.where(np.isfinite(test), true, np.nan)
    if isinstance(model, KNNImputer):
        # reduced_reshaped = reduced.reshape((144,900))
        pred_full = model.transform(reduced)
        # pred_full = pred_full_reshaped.reshape((720,180))

    elif isinstance(model, SimpleImputer) or isinstance(model, IterativeImputer):
        if transpose:
            pred_full = model.fit_transform(reduced.T).T
        else:
            pred_full = model.fit_transform(reduced)

    pred = np.where((np.isnan(reduced)
                    & np.isfinite(true)
                    & np.meshgrid(np.isfinite(reduced).any(axis=1), np.arange(0, reduced.shape[1]),
                                  indexing="ij")[0]),
                    pred_full, np.nan)
    target = np.where(np.isfinite(pred), true, np.nan)

    return [true, reduced, pred_full, pred, target]


def calculate_scores(model: _BaseImputer, res: list) -> list:
    pred = res[3]
    target = res[4]

    if isinstance(model, KNNImputer):
        pred = pred.reshape((144, 900))
        target = target.reshape((144, 900))

    scores = []
    mse_scores = []
    for i, _ in enumerate(pred):
        if np.any(np.isfinite(target[i])) and np.any(np.isfinite(pred[i])):
            scores.append(r2_score(
                target[i][np.isfinite(target[i])], pred[i][np.isfinite(pred[i])]
                )
            )
            mse_scores.append(-mean_squared_error(
                target[i][np.isfinite(target[i])], pred[i][np.isfinite(pred[i])]
                )
            )
        else:
            scores.append(np.nan)
            mse_scores.append(np.nan)

    return [scores, mse_scores]


def plot_results(model: _BaseImputer, res: list, intensities: np.ndarray,
                 sc: WindConstants, cov_sc: SoloConstants, transpose=True,
                 save_plot=True, save_path=None) -> None:

    scores, mse_scores = calculate_scores(model, res)

    true, reduced, pred_full, pred, target = res
    true_miss_percent = np.sum(np.where(np.isnan(true), 1, 0)) \
        / (true.shape[0] * true.shape[1]) * 100
    reduced_miss_percent = np.sum(np.where(np.isnan(reduced), 1, 0)) \
        / (reduced.shape[0] * reduced.shape[1]) * 100

    fig, axs = plt.subplots(nrows=9, figsize=(16, 36), sharex=True)

    X, Y = np.meshgrid(np.arange(0, true.shape[0]), np.arange(0, true.shape[1]), indexing="ij")

    norm = LogNorm(np.nanmin(true), np.nanmax(true))
    diff_norm = LogNorm(1e-3, 1e3)

    for i in range(8):
        axs[0].plot(intensities[:, i], label=sc.sectors[i])
    axs[0].set_title("Intensities")
    axs[0].set_yscale("log")
    axs[0].legend(loc="upper right")

    axs[1].pcolormesh(X, Y, true, norm=norm, cmap="inferno")
    axs[1].set_title(f"WIND PAD, {true_miss_percent:.2f} % missing")

    axs[2].pcolormesh(X, Y, reduced, norm=norm, cmap="inferno")
    axs[2].set_title(f"WIND PAD w/ reduction, {reduced_miss_percent:.2f} % missing")

    axs[3].pcolormesh(X, Y, pred_full, norm=norm, cmap="inferno")
    if isinstance(model, KNNImputer):
        axs[3].set_title(f"k-NN imputed values, k = {model.get_params()["n_neighbors"]}")
    elif isinstance(model, SimpleImputer):
        axs[3].set_title("Mean imputed values")
    elif isinstance(model, IterativeImputer):
        axs[3].set_title(f"Iteratively imputed values with {model.estimator}")

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

    if isinstance(model, KNNImputer):
        axs[7].plot(np.arange(0, 720, 5), scores)
        axs[7].set_title("R2 score per 5 x 180 block")

        axs[8].plot(np.arange(0, 720, 5), mse_scores)
        axs[8].set_title("Negative MSE score per 5 x 180 block")

    else:
        axs[7].plot(np.arange(0, 720, 1), scores)
        axs[7].set_title("R2 score")

        axs[8].plot(np.arange(0, 720, 1), mse_scores)
        axs[8].set_title("Negative MSE score")

    if save_plot:
        plt.savefig(save_path)
        plt.close()

    else:
        plt.show()
