import numpy as np
import pandas as pd
import h5py
import sys

from os import PathLike
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from aipad.spacecrafts import SoloConstants, WindConstants
from aipad.pad_imputation import (
    calculate_PAD, make_train_test, convert_to_bool_coverage)
from aipad.padimputer import PADImputer, KNNPadImputer

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in log10", category=RuntimeWarning)

args = sys.argv
print(args)
random = np.random.default_rng(seed=int(args[5]))


def load_knn_data(sc, imputer: PADImputer, file_path: PathLike, event_csv: PathLike, size_pct: float) -> list:
    trues = []
    reduceds = []
    times = []
    mags = []
    intensities = []
    events = pd.read_csv(event_csv, header=0).values.ravel()    # consider only events given in a separate csv

    with h5py.File(file_path, 'r') as hdf:

        for event_key in events:
            imputer.load_event_data(hdf, event_key, sc)
            I_data = imputer.event_data["Intensity"]
            cov_data = imputer.event_data["Coverage"]
            B_data = imputer.event_data["MagField"]
            time = imputer.event_data["Intensity_attrs"]["Epoch"]
            X, Y, true = calculate_PAD(wind, I_data.values, cov_data, 8)
            reduced_cov, mask = PADImputer._induce_missingness(cov_data, size_pct, random)
            X, Y, reduced = calculate_PAD(wind, I_data.values, reduced_cov, 8)

            trues.append(true)
            reduceds.append(reduced)
            times.append(time)
            mags.append(B_data)
            intensities.append(I_data)

    return make_train_test(trues, reduceds, times, mags, intensities, events.tolist())


solo = SoloConstants()
wind = WindConstants()

bins = 8
neighbors = int(args[3])
n_splits = 3
missingness = args[1]
miss_pct = int(args[2])
size_pct = miss_pct / 100

if args[4] in ["log", "log+min_max", "min_max"]:
    scale = args[4]

else:
    scale = "none"

# bins = 8
# neighbors = 5
# n_splits = 3
# missingness = "mar"
# miss_pct = 20
# size_pct = miss_pct / 100
# scale = "log"

file_path = Path("./data/hdf5") / "wind_1min_8bins.h5"
plot_path = Path(f"./plots/{missingness}/knn_with_traintest")
plot_path.mkdir(exist_ok=True, parents=True)

pa = np.linspace(0.5*180/bins, (bins-0.5)*180/bins, bins)
imputer = KNNImputer(n_neighbors=neighbors, weights="uniform", keep_empty_features=True)
pad_imputer = KNNPadImputer(wind, knn_neighbors=neighbors, knn_weigth="uniform")
pad_imputer.missingness = missingness
(X_train, X_test, y_train, y_test, t_train, t_test,
 B_train, B_test, I_train, I_test, e_train, e_test) = load_knn_data(wind, pad_imputer, file_path,
                                                                    Path("./events.csv"), size_pct)

X_train_stacked = np.vstack(X_train)
if scale == "min_max":
    scaler = MinMaxScaler()
    X_train_stacked = scaler.fit_transform(X_train_stacked)

elif scale == "log":
    scaler = None
    X_train_stacked = np.log10(X_train_stacked)

elif scale == "log+min_max":
    X_train_stacked = np.log10(X_train_stacked)
    scaler = MinMaxScaler()
    X_train_stacked = scaler.fit_transform(X_train_stacked)

else:
    scaler = None

imputer.fit(X_train_stacked)

for j in range(len(y_test)):
    # KNN with train test
    event_key = e_test[j]
    results_path = Path(f"./results/knn_results/{missingness}")
    results_path.mkdir(exist_ok=True, parents=True)

    print(f"Event {event_key}")
    try:
        df = pd.read_csv(results_path / f"{event_key}.csv", header=[0, 1], index_col=[0, 1])

    except FileNotFoundError:
        df = pd.DataFrame(index=pd.MultiIndex.from_product([[], []], names=["method", "n_neighbors"]),
                          columns=pd.MultiIndex.from_product([[], []], names=["miss_pct", "scale"]))

    true = X_test[j]

    if missingness == "mar":
        B_data = B_test[j]
        onset_dt = pd.to_datetime(event_key)
        start_index = len(B_data[(B_data.index < onset_dt)])
        B_data = B_data.iloc[start_index-2*60:start_index+10*60]
        reduced_cov = PADImputer.calc_coverage(B_data, "solo", opening=52.8)
        # TODO sc_like -> corresponding SC constants
        Xr, Yr, reduced_cov_bool = convert_to_bool_coverage(reduced_cov, SoloConstants())
        repeats = 1
        reduced = np.where(reduced_cov_bool, true, np.nan)
        reduced_miss_percent = np.sum(np.where(np.isnan(reduced), 1, 0)) \
            / (reduced.shape[0] * reduced.shape[1]) * 100

    elif missingness == "mcar":
        reduced = y_test[j]

    if scale == "log":
        reduced_ = np.log10(reduced)
        imputed = 10 ** imputer.transform(reduced_)

    elif scale == "min_max":
        reduced_ = scaler.transform(reduced)
        imputed = scaler.inverse_transform(imputer.transform(reduced_))

    elif scale == "log+min_max":
        reduced_ = scaler.transform(np.log10(reduced))
        imputed = 10 ** scaler.inverse_transform(imputer.transform(reduced_))

    else:
        imputed = imputer.transform(reduced)

    target, pred = PADImputer._target_and_prediction(true, reduced, imputed)
    score = PADImputer._calculate_scores(target, pred, normalize=True)

    df.loc[("knn_with_traintest", neighbors), (str(miss_pct), scale)] = score
    fname = f"{event_key}_knn_with_traintest_axis0_{miss_pct}pct_{neighbors}neighbors"

    # plot_results(pa, I_test[j], B_test[j], true, reduced, imputed, "rmse", wind, solo,
    #              save_path=plot_path / fname)
    del true, reduced, imputed, target, pred
    df.to_csv(results_path / f"{event_key}.csv")

    pad_imputer.run_analysis(event_key, "solo", miss_pct, repeats=1, opening=52.8, scale=scale,
                             scaler=scaler,
                             normalize_rmse=True, save_plot=False, show_plot=False,
                             file_path=file_path, results_path=results_path)
