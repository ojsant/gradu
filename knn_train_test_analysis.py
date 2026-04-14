import numpy as np
import pandas as pd
import h5py
import sys

from os import PathLike
from pathlib import Path
from sklearn.impute import KNNImputer
from aipad.spacecrafts import SoloConstants, WindConstants
from aipad.pad_imputation import (
    pad_histogram, make_train_test)
from aipad.padimputer import PADImputer, KNNPadImputer

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in log10", category=RuntimeWarning)


def KNN_traintest(imputer: PADImputer, file_path: PathLike, size_pct: float, seed: int, scale="log") -> list:
    trues = []
    reduceds = []
    times = []
    mags = []
    intensities = []

    with h5py.File(file_path, 'r') as hdf:
        event_list = list(hdf.keys())

        for event_no in range(len(event_list)):
            imputer.load_event_data(hdf, event_list[event_no], wind)
            I_data = imputer.event_data["Intensity"]
            cov_data = imputer.event_data["Coverage"]
            B_data = imputer.event_data["MagField"]
            time = imputer.event_data["Intensity_attrs"]["Epoch"]
            X, Y, true = pad_histogram(wind, np.log10(I_data.values) if scale == "log" else I_data.values, cov_data, 8)
            reduced_cov, mask = PADImputer._induce_missingness(cov_data, size_pct, np.random.default_rng(seed))
            X, Y, reduced = pad_histogram(wind, np.log10(I_data.values) if scale == "log" else I_data.values, reduced_cov, 8)

            trues.append(true)
            reduceds.append(reduced)
            times.append(time)
            mags.append(B_data)
            intensities.append(I_data)

    return make_train_test(trues, reduceds, times, mags, intensities, event_list)


args = sys.argv
print(args)

file_path = Path("./data/hdf5") / "wind_1min_8bins.h5"
plot_path = Path("./knn_results/plots/knn_with_traintest")
plot_path.mkdir(exist_ok=True, parents=True)

solo = SoloConstants()
wind = WindConstants()

bins = 8
neighbors = int(args[2])
n_splits = 3
miss_pct = int(args[1])
size_pct = miss_pct / 100
pa = np.linspace(0.5*180/bins, (bins-0.5)*180/bins, bins)
imputer = KNNImputer(n_neighbors=neighbors, weights="uniform", keep_empty_features=True)
pad_imputer = KNNPadImputer(wind, knn_neighbors=neighbors, knn_weigth="uniform")
(X_train, X_test, y_train, y_test, t_train, t_test,
 B_train, B_test, I_train, I_test, e_train, e_test) = KNN_traintest(pad_imputer, file_path, size_pct, 123, scale="none")

X_train_stacked = np.vstack(X_train)

imputer.fit(X_train_stacked)

for j in range(len(e_test)):
    # KNN with train test
    df = None
    event_key = e_test[j]
    results_path = Path("./knn_results")
    results_path.mkdir(exist_ok=True)
    print(f"Event {event_key}")
    try:
        df = pd.read_csv(results_path / f"{event_key}.csv", header=[0, 1], index_col=[0, 1, 2])

    except FileNotFoundError:
        df = pd.DataFrame(index=pd.MultiIndex.from_product([
            ("knn", "knn_with_traintest"), ("time", "pitch_angle"),
            ("1n", "5n", "15n", "20n")], names=["method", "axis", "n_neighbors"]),
            columns=pd.MultiIndex.from_product([
                ["rmse_mean", "rmse_std"],
                ["10", "20", "30", "40", "50"]], names=["stat", "miss_percent"]))

    true = X_test[j]
    reduced = y_test[j]
    imputed = imputer.transform(reduced)
    target, pred = PADImputer._target_and_prediction(true, reduced, imputed)
    score = PADImputer._calculate_scores(target, pred, "rmse")

    df.loc[("knn_with_traintest", "time", f"{neighbors}n"),
           ("rmse_mean", str(miss_pct))] = score
    # df.loc[(imputer.method, args[3]), ("rmse_std", miss_percent)] = std
    fname = f"{event_key}_knn_with_traintest_axis0_{miss_pct}pct_{neighbors}nbors"
    # plot_results(pa, I_test[j], B_test[j], true, reduced, imputed, "rmse", wind, solo,
    #              save_path=plot_path / fname)
    del true, reduced, imputed, target, pred
    df.to_csv(results_path / f"{event_key}.csv")

    # without train test
    pad_imputer.run_analysis(event_key, miss_pct, repeats=10, scale=None, save_plot=False,
                             show_plot=False,
                             file_path=file_path,
                             results_path=results_path)
