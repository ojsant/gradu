import numpy as np
import pandas as pd
import h5py
import sys

from pathlib import Path

from aipad.padimputer import PADImputer
from aipad.spacecrafts import SoloConstants, WindConstants
from aipad.pad_imputation import (
    calculate_scores,  pad_histogram, induce_missingness,
    load_event_data, plot_results, target_from_prediction)

args = sys.argv

file_path = Path("./data/hdf5") / "wind_1min_8bins.h5"
plot_path = Path(f"./plots/imputers/{args[2]}/{args[3]}")
plot_path.mkdir(exist_ok=True)
csv_path = Path("./results")
csv_path.mkdir(exist_ok=True)

imputer = PADImputer()
solo = SoloConstants()
wind = WindConstants()
miss_percents = ["10", "20", "50", "80"]
repeats = 50
index = pd.MultiIndex.from_product([list(imputer._strategy_map.keys()), ["time", "pitch_angle"]],
                                   names=["method", "axis"])
columns = pd.MultiIndex.from_product([["rmse_mean", "rmse_std"], miss_percents],
                                     names=["stat", "miss_percent"])

print(args)
df = None

with h5py.File(file_path, 'r') as hdf:
    try:
        df = pd.read_csv(f"./results/{args[1]}.csv", header=[0, 1], index_col=[0, 1])
    except FileNotFoundError:
        df = pd.DataFrame(index=index, columns=columns)
    event_data = load_event_data(hdf, args[1], wind)

    I_data = event_data["Intensity"]
    B_data = event_data["MagField"]
    cov_data = event_data["Coverage"]
    times = event_data["Intensity_attrs"]["Epoch"]

    X, Y, true = pad_histogram(wind, I_data.values, cov_data, 8)
    imputer.method = args[2]
    imputer.axis = 0 if args[3] == "time" else 1
    miss_percent = args[4]
    scores = []
    for i in range(repeats):
        reduced_cov, mask = induce_missingness(cov_data, int(miss_percent) / 100)
        X, Y, reduced = pad_histogram(wind, I_data.values, reduced_cov, 8)
        imputed = imputer.fit_transform(reduced)
        target, pred = target_from_prediction(true, reduced, imputed)
        score = calculate_scores(target, pred, "rmse")
        scores.append(score)

    scores = np.array(scores)
    mean = scores.mean()
    std = scores.std(ddof=1)
    df.loc[(imputer.method, args[3]), ("rmse_mean", miss_percent)] = mean
    df.loc[(imputer.method, args[3]), ("rmse_std", miss_percent)] = std
    fname = f"{args[1]}_{imputer.method}_axis{imputer.axis}_{miss_percent}pct"
    plot_results(Y[0], I_data, B_data, true, reduced, imputed, "rmse", wind, solo,
                 save_path=plot_path / fname)

df.to_csv(csv_path / f"{args[1]}.csv")
