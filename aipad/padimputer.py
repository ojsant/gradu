import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import h5py

# from os import PathLike   TODO
from typing import Literal
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import griddata
from pathlib import Path
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.dates import HourLocator, DateFormatter

from aipad.pad_imputation import pad_histogram

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in log10",
                        category=RuntimeWarning)

DEFAULT_FILE_PATH = Path("./data/hdf5")
DEFAULT_RESULTS_PATH = Path("./results")
DEFAULT_PLOT_PATH = Path("./plots")


def convert_timestamps(attrs):
    """Convert Epoch timestamps to NumPy datetime."""
    epoch = attrs["Epoch"]
    t_unit = attrs["t_unit"]
    return epoch.astype(f"datetime64[{t_unit}]")


class PADImputer(BaseEstimator, TransformerMixin):
    def __init__(self, spacecraft, bins: int = 8,
                 method: Literal["mean", "fill_average", "interp",
                                 "knn", "cubic1d", "cubic2d"] = "mean",
                 axis: int = 1, knn_neighbors: int = 5,
                 knn_weight: Literal["uniform", "distance"] = "uniform",
                 random_seed: int = 123):
        self.method = method
        self.axis = axis
        self.spacecraft = spacecraft
        self.bins = bins
        self.pa_bins = np.linspace(0.5*180/self.bins, (self.bins-0.5)*180/self.bins, self.bins)

        self._knn_neighbors = knn_neighbors
        self._knn_weigth = knn_weight

        self._strategy_map = {
            "mean": self._mean_finder,
            "fill_average": self._fill_average_finder,
            "interp": self._interp_finder,
            "knn": self._knn_finder,
            "cubic1d": self._cubic1d_finder,
            "cubic2d": self._cubic2d_finder
            }

        if random_seed is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = np.random.default_rng(random_seed)

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

    def _cubic1d_finder(self, a: NDArray, axis: Literal[0, 1]):
        return (pd.DataFrame(a).interpolate(method="cubic",
                                            axis=axis, limit_direction="both")).to_numpy()

    def _cubic2d_finder(self, a: NDArray, axis: Literal[0, 1]):
        X, Y = np.meshgrid(np.arange(a.shape[0]), np.arange(a.shape[1]), indexing="ij")
        points = np.column_stack((X.ravel(), Y.ravel()))
        values = a.ravel()

        mask = np.isfinite(values)
        points_masked = points[mask]
        values_masked = values[mask]
        return griddata(points_masked, values_masked,
                        points, method='cubic').reshape(a.shape)

    def load_event_data(self, hdf, event_name, sc):
        """Load datasets and attributes for a given event."""
        event_group = hdf[event_name]

        # Load datasets
        intensity_attrs = dict(event_group["Intensity"].attrs)
        intensity_attrs["Epoch"] = convert_timestamps(intensity_attrs)
        intensity = pd.DataFrame(event_group["Intensity"][:], index=intensity_attrs["Epoch"],
                                 columns=sc.sectors)

        mag_field_attrs = dict(event_group["MagField"].attrs)
        mag_field_attrs["Epoch"] = convert_timestamps(mag_field_attrs)
        mag_field = pd.DataFrame(event_group["MagField"][:], index=mag_field_attrs["Epoch"],
                                 columns=["Bx", "By", "Bz", "B"])

        ind = pd.MultiIndex.from_product([sc.sectors, ["min", "center", "max"]])
        coverage_attrs = dict(event_group["Coverage"].attrs)
        coverage_attrs["Epoch"] = convert_timestamps(coverage_attrs)
        coverage = pd.DataFrame(event_group["Coverage"][:], index=coverage_attrs["Epoch"],
                                columns=ind)

        self.event_data = {
            "Intensity": intensity,
            "MagField": mag_field,
            "Coverage": coverage,
            "Intensity_attrs": intensity_attrs,
            "MagField_attrs": mag_field_attrs,
            "Coverage_attrs": coverage_attrs
        }

    @classmethod
    def _induce_missingness(cls, cov: pd.DataFrame, pct: float, rng: np.random.Generator) \
            -> tuple[pd.DataFrame, np.ndarray]:
        """Set a percentage of given coverage as missing. Returns a copy of the coverage array
        without the randomly picked values and the missingness mask.

        Args:
            cov (pd.DataFrame): coverage with columns ("min", "center", "max") for each sector
            p (float): proportion of data to set as missing
        """
        p = pct / 100
        mask = rng.choice([False, True], size=(cov.shape[0], cov.shape[1] // 3), p=[1-p, p])
        mask = np.repeat(mask, 3, axis=1)
        masked_cov = cov.where(~mask, np.nan)  # DataFrame.where() replaces where condition is False
        return masked_cov, mask

    @classmethod
    def _target_and_prediction(cls, true, reduced, imputed):
        pred = np.where((np.isnan(reduced)
                        & np.isfinite(true)
                        & np.meshgrid(np.isfinite(reduced).any(axis=1),
                                      np.arange(0, reduced.shape[1]),
                                      indexing="ij")[0]),
                        imputed, np.nan)
        target = np.where(np.isfinite(pred), true, np.nan)
        return target, pred

    @classmethod
    def _calculate_scores(cls, target, pred, score: str) -> float:
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

    def plot_results(self, I_data, B_data, true, reduced, imputed, score: str, scale,
                     save_plot=True, show_plot=True, save_path=None) -> None:

        target, pred = PADImputer._target_and_prediction(true, reduced, imputed)

        # Score over whole 12 hrs
        total_score = PADImputer._calculate_scores(target, pred, score)

        true_miss_percent = np.sum(np.where(np.isnan(true), 1, 0)) \
            / (true.shape[0] * true.shape[1]) * 100
        reduced_miss_percent = np.sum(np.where(np.isnan(reduced), 1, 0)) \
            / (reduced.shape[0] * reduced.shape[1]) * 100

        X, Y = np.meshgrid(I_data.index.values, self.pa_bins, indexing="ij")

        diff = target - pred

        norm = LogNorm(np.nanmin(true), np.nanmax(true))

        fig, axs = plt.subplots(nrows=8, figsize=(16, 32), sharex=True)

        for b in range(B_data.values.shape[1]):
            axs[0].plot(B_data.index.values, B_data.iloc[:, b], label=B_data.columns[b])

        for i in range(I_data.shape[1]):
            axs[1].plot(I_data.index.values, I_data.iloc[:, i], label=self.spacecraft.sectors[i])

        axs[0].legend(loc="upper right")
        axs[0].set_ylabel("B [nT]")
        axs[0].xaxis.set_major_locator(HourLocator(range(24)))
        axs[0].xaxis.set_major_formatter(DateFormatter("%d %b\n%H:%M"))
        axs[0].set_title("Magnetic field (GSE coordinates)")

        axs[1].set_yscale("log")
        axs[1].legend(loc="upper right")
        axs[1].set_ylabel(r"I $[\mathrm{(cm^2\ s\ sr\ MeV)^{-1}}]$")
        axs[1].set_title("Intensities")

        int_mesh = axs[2].pcolormesh(X, Y, true, norm=norm, cmap="inferno")
        axs[2].set_title(f"{self.spacecraft.name.upper()} PAD, ",
                         f"total {true_miss_percent:.2f} % missing")
        axs[2].set_yticks(np.linspace(0, 180, 9))
        axs[2].set_ylabel(f"Pitch angle [{u"\u03b8"}]")

        axs[3].pcolormesh(X, Y, reduced, norm=norm, cmap="inferno")
        axs[3].set_title(f"{self.spacecraft.name.upper()} PAD w/ reduction, ",
                         f" {reduced_miss_percent:.2f} % missing")
        axs[3].set_yticks(np.linspace(0, 180, 9))
        axs[3].set_ylabel(f"Pitch angle [{u"\u03b8"}]")

        axs[4].pcolormesh(X, Y, imputed, norm=norm, cmap="inferno")
        axs[4].set_title("Imputed values")
        axs[4].set_yticks(np.linspace(0, 180, 9))
        axs[4].set_ylabel(f"Pitch angle [{u"\u03b8"}]")

        axs[5].pcolormesh(X, Y, pred, norm=norm, cmap="inferno")
        axs[5].set_title("Predicted values")
        axs[5].set_yticks(np.linspace(0, 180, 9))
        axs[5].set_ylabel(f"Pitch angle [{u"\u03b8"}]")

        axs[6].pcolormesh(X, Y, target, norm=norm, cmap="inferno")
        axs[6].set_title(f"Target values (RMSE = {total_score})")
        axs[6].set_yticks(np.linspace(0, 180, 9))
        axs[6].set_ylabel(f"Pitch angle [{u"\u03b8"}]")

        if scale is None:
            log_data = np.sign(diff) * np.log10(np.abs(diff) + 1e-10)  # Add small offset to avoid log(0)
            vmax = np.nanmax(np.log10(np.abs(diff) + 1e-10))
            vmin = -vmax
            diff_mesh = axs[7].pcolormesh(X, Y, log_data, norm=Normalize(vmin=vmin, vmax=vmax),
                                          cmap="bwr")
            axs[7].set_title(r"log|target - prediction| (signed)")

        else:
            vmax = np.nanmax(diff)
            vmin = -vmax
            diff_mesh = axs[7].pcolormesh(X, Y, diff, norm=Normalize(vmin=vmin, vmax=vmax),
                                          cmap="bwr")
            axs[7].set_title(r"Difference (TODO: this)")

        axins1 = inset_axes(axs[2], width="100%", height="100%", loc="center",
                            bbox_to_anchor=(1.01, 0, 0.03, 1), bbox_transform=axs[2].transAxes,
                            borderpad=0.2)
        axins2 = inset_axes(axs[7], width="100%", height="100%", loc="center",
                            bbox_to_anchor=(1.01, 0, 0.03, 1), bbox_transform=axs[7].transAxes,
                            borderpad=0.2)
        fig.colorbar(int_mesh, cax=axins1, ax=axs[2])
        fig.colorbar(diff_mesh, cax=axins2, ax=axs[7])
        axs[7].set_yticks(np.linspace(0, 180, 9))
        axs[7].set_ylabel(f"Pitch angle [{u"\u03b8"}]")
        axs[7].set_xlabel(f"Date (in {I_data.index[0].strftime("%Y")})")

        axs[0].set_xlim((I_data.index[0], I_data.index[-1]))
        if save_plot:
            plt.savefig(save_path)
            plt.clf()
            plt.close()
            del fig, axs, axins1, axins2, X, Y, target, pred
            gc.collect()

        else:
            if not show_plot:
                plt.show()

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
        self.imputed = X_imputed
        return X_imputed

    def run_analysis(self, event_key, miss_pct, repeats=100,
                     scale: Literal[None, "log", "min_max"] = None,
                     save_plot: bool = True, fname: str = "plot.png",
                     show_plot: bool = True,
                     file_path: Path = DEFAULT_FILE_PATH,
                     results_path: Path = DEFAULT_RESULTS_PATH,
                     plot_path: Path = DEFAULT_PLOT_PATH):

        index = pd.MultiIndex.from_product([[self.method], ["time", "pitch_angle"]],
                                           names=["method", "axis"])
        columns = pd.MultiIndex.from_product([["rmse_mean", "rmse_std"], [str(miss_pct)]],
                                             names=["stat", "miss_percent"])

        axis_str = "time" if self.axis == 0 else "pitch_angle"
        sc = self.spacecraft
        scaler = None

        with h5py.File(file_path, 'r') as hdf:
            try:
                df = pd.read_csv(results_path / f"{event_key}.csv", header=[0, 1], index_col=[0, 1])
            except FileNotFoundError:
                df = pd.DataFrame(index=index, columns=columns)

            self.load_event_data(hdf, event_key, sc)

            self.event_key = event_key

            I_data = self.event_data["Intensity"]
            B_data = self.event_data["MagField"]
            cov_data = self.event_data["Coverage"]
            # times = self.event_data["Intensity_attrs"]["Epoch"]

            if scale == "min_max":
                scaler = MinMaxScaler()
                intensity = scaler.fit_transform(I_data.values)
            elif scale == "log":
                intensity = np.log10(I_data.values)
            else:
                intensity = I_data.values

            X, Y, true = pad_histogram(sc, intensity, cov_data, self.bins)
            self.true = true
            X, Y = np.meshgrid(np.arange(true.shape[0]), np.arange(true.shape[1]), indexing="ij")
            scores = []

            if repeats < 10:
                print("The analysis is repeated a number of times to better estimate the effect of "
                      "random sampling on the score. Consider using "
                      f"more repeats than {repeats}")

            for i in range(repeats):
                reduced_cov, mask = PADImputer._induce_missingness(cov_data, miss_pct, self._rng)
                Xr, Yr, reduced = pad_histogram(sc, intensity, reduced_cov, self.bins)
                imputed = self.transform(reduced)
                if scale == "log":  # undo scaling for error measure calculation and plotting
                    imputed = 10 ** imputed
                    reduced = 10 ** reduced
                    true_ = 10 ** true
                elif scale == "min_max":
                    imputed = scaler.inverse_transform(imputed)
                    reduced = scaler.inverse_transform(reduced)
                    true_ = scaler.inverse_transform(true)
                else:
                    true_ = true
                target, pred = self._target_and_prediction(true_, reduced, imputed)
                score = self._calculate_scores(target, pred, "rmse")
                scores.append(score)

            mean = np.mean(scores)
            std = np.std(scores, ddof=1)
            print(f"Event {event_key} with {miss_pct} % MCAR missingness: results using method "
                  f"'{self.method}' along {axis_str} axis")
            print(f"RMSE mean: {mean:.3f}")
            print(f"RMSE std: {std:.3f}")

            df.loc[(self.method, axis_str), ("rmse_mean", str(miss_pct))] = mean
            df.loc[(self.method, axis_str), ("rmse_std", str(miss_pct))] = std

            if save_plot or show_plot:
                self.plot_results(I_data, B_data, true, reduced, imputed, "rmse", scale=scale,
                                  save_plot=save_plot,
                                  save_path=plot_path / fname, show_plot=show_plot)

            df.to_csv(results_path / f"{event_key}.csv")


class KNNPadImputer(PADImputer):
    def __init__(self, spacecraft, knn_neighbors, knn_weigth, bins=8, random_seed=123):
        super().__init__(spacecraft=spacecraft, bins=bins, method="knn", axis=0,
                         knn_neighbors=knn_neighbors, knn_weight=knn_weigth,
                         random_seed=random_seed)

    def run_analysis(self, event_key, miss_pct, repeats=10,
                     scale: None | Literal['log', 'min_max'] = None, scaler=None,
                     save_plot: bool = True, fname: str = "plot.png", show_plot: bool = True,
                     file_path: Path = DEFAULT_FILE_PATH, results_path: Path = DEFAULT_RESULTS_PATH,
                     plot_path: Path = DEFAULT_PLOT_PATH):

        index = pd.MultiIndex.from_product([["knn", "knn_with_traintest"], ["time", "pitch_angle"],
                                            ["1n", "5n", "15n", "20n"]],
                                           names=["method", "axis", "n_neighbors"])
        columns = pd.MultiIndex.from_product([["rmse_mean", "rmse_std"], [str(miss_pct)]],
                                             names=["stat", "miss_percent"])

        axis_str = "time" if self.axis == 0 else "pitch_angle"
        sc = self.spacecraft

        with h5py.File(file_path, 'r') as hdf:
            try:
                df = pd.read_csv(results_path / f"{event_key}.csv",
                                 header=[0, 1], index_col=[0, 1, 2])
            except FileNotFoundError:
                df = pd.DataFrame(index=index, columns=columns)

            self.load_event_data(hdf, event_key, sc)

            self.event_key = event_key

            I_data = self.event_data["Intensity"]
            B_data = self.event_data["MagField"]
            cov_data = self.event_data["Coverage"]
            # times = self.event_data["Intensity_attrs"]["Epoch"]

            if scale == "min_max":
                if isinstance(scaler, MinMaxScaler):
                    intensity = scaler.transform(I_data.values)
                else:
                    raise TypeError("pass a fitted MinMaxScaler object")

            elif scale == "log":
                intensity = np.log10(I_data.values)
            else:
                intensity = I_data.values

            X, Y, true = pad_histogram(sc, intensity, cov_data, self.bins)
            self.true = true
            X, Y = np.meshgrid(np.arange(true.shape[0]), np.arange(true.shape[1]), indexing="ij")
            scores = []

            if repeats < 10:
                print("The analysis is repeated a number of times to better estimate the effect of "
                      "random sampling on the score. Consider using "
                      f"more repeats than {repeats}")

            for i in range(repeats):
                reduced_cov, mask = PADImputer._induce_missingness(cov_data, miss_pct, self._rng)
                Xr, Yr, reduced = pad_histogram(sc, intensity, reduced_cov, self.bins)
                imputed = self.transform(reduced)
                if scale == "log":  # undo scaling for error measure calculation and plotting
                    imputed = 10 ** imputed
                    reduced = 10 ** reduced
                    true_ = 10 ** true
                elif scale == "min_max":
                    imputed = scaler.inverse_transform(imputed)
                    reduced = scaler.inverse_transform(reduced)
                    true_ = scaler.inverse_transform(true)
                else:
                    true_ = true
                target, pred = self._target_and_prediction(true_, reduced, imputed)
                score = self._calculate_scores(target, pred, "rmse")
                scores.append(score)

            mean = np.mean(scores)
            std = np.std(scores, ddof=1)
            print(f"Event {event_key} with {miss_pct} % MCAR missingness: results using method "
                  f"'{self.method}' along {axis_str} axis")
            print(f"RMSE mean: {mean:.3f}")
            print(f"RMSE std: {std:.3f}")

            df.loc[(self.method, axis_str, f"{self._knn_neighbors}n"),
                   ("rmse_mean", str(miss_pct))] = mean
            df.loc[(self.method, axis_str, f"{self._knn_neighbors}n"),
                   ("rmse_std", str(miss_pct))] = std

            if save_plot or show_plot:
                self.plot_results(I_data, B_data, true, reduced, imputed, "rmse", scale=scale,
                                  save_plot=save_plot,
                                  save_path=plot_path / fname, show_plot=show_plot)

            df.to_csv(results_path / f"{event_key}.csv")
