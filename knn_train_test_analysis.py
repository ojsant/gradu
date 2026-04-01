import numpy as np
import pandas as pd
import h5py
import sys
import subprocess

from os import PathLike
from pathlib import Path
from sklearn.impute import KNNImputer
from aipad.spacecrafts import SoloConstants, WindConstants
from aipad.pad_imputation import (
    calculate_scores,  pad_histogram, induce_missingness,
    load_event_data, plot_results, target_from_prediction, make_train_test)


def KNN_traintest(file_path: PathLike, size_pct: float, seed: int) -> list:
    trues = []
    reduceds = []
    times = []
    mags = []
    intensities = []

    with h5py.File(file_path, 'r') as hdf:
        event_list = list(hdf.keys())

        for event_no in range(len(event_list)):
            event_data = load_event_data(hdf, event_list[event_no], wind)
            I_data = event_data["Intensity"]
            cov_data = event_data["Coverage"]
            B_data = event_data["MagField"]
            time = event_data["Intensity_attrs"]["Epoch"]
            X, Y, true = pad_histogram(wind, I_data.values, cov_data, 8)
            reduced_cov, mask = induce_missingness(cov_data, size_pct, seed)
            X, Y, reduced = pad_histogram(wind, I_data.values, reduced_cov, 8)

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
miss_percent = int(args[1])
size_pct = miss_percent / 100
pa = np.linspace(0.5*180/bins, (bins-0.5)*180/bins, bins)
imputer = KNNImputer(n_neighbors=neighbors, weights="uniform", keep_empty_features=True)

(X_train, X_test, y_train, y_test, t_train, t_test,
 B_train, B_test, I_train, I_test, e_train, e_test) = KNN_traintest(file_path, size_pct, 123)

X_train_stacked = np.vstack(X_train)

imputer.fit(X_train_stacked)

for j in range(30):
    # KNN with train test
    df = None
    event_key = e_test[j]
    print(f"Event {event_key}")
    try:
        df = pd.read_csv(f"./knn_results/{event_key}.csv", header=0, index_col=[0, 1])

    except FileNotFoundError:
        df = pd.DataFrame(index=pd.MultiIndex.from_product([("knn", "knn_with_traintest"),
                                                            ("1n", "5n", "10n", "15n")],
                          names=["method", "n_neighbors"]),
                          columns=pd.Series(["10", "20", "50", "80"], name="miss_percent"))

    true = X_test[j]
    reduced = y_test[j]
    imputed = imputer.transform(reduced)
    target, pred = target_from_prediction(true, reduced, imputed)
    score = calculate_scores(target, pred, "rmse")

    df.loc[("knn_with_traintest", f"{neighbors}n"),
           str(miss_percent)] = score
    # df.loc[(imputer.method, args[3]), ("rmse_std", miss_percent)] = std
    fname = f"{event_key}_knn_with_traintest_axis0_{miss_percent}pct_{neighbors}nbors"
    plot_results(pa, I_test[j], B_test[j], true, reduced, imputed, "rmse", wind, solo,
                 save_path=plot_path / fname)
    del true, reduced, imputed, target, pred
    df.to_csv(f"./knn_results/{event_key}.csv")

    # without train test
    method = "knn"
    axis = "time"
    rc = subprocess.run(("python3", "knn_analysis.py", event_key,
                         str(miss_percent), str(neighbors)))
