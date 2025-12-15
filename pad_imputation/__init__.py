import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import copy

from dataclasses import dataclass, field
from pathlib import Path
from anisotropy import run_SEPevent


@dataclass(frozen=True)
class SoloConstants:
    sectors: list[str] = field(default_factory=lambda: ["sun", "asun", "north", "south"])
    mission_start: dt.datetime = dt.datetime(2021, 1, 1)
    mission_end: dt.datetime = dt.datetime(2025, 5, 31)
    native_cadence: str = "1min"
    pitch_angle_mu_columns: list[str] = field(default_factory=lambda: [f"Pitch_Angle_{dir}" for dir in ["S", "A", "N", "D"]])
    pitch_angle_sigma_columns: list[str] = field(default_factory=lambda: [f"Pitch_Angle_Sigma_{dir}" for dir in ["S", "A", "N", "D"]])


@dataclass(frozen=True)
class WindConstants:
    sectors: list[str] = field(default_factory=lambda: [f"P{i}" for i in range(8)])
    mission_start: dt.datetime = dt.datetime(2005, 1, 1)
    mission_end: dt.datetime = dt.datetime(2025, 5, 31)
    native_cadence: str = "12s"


def coverage_overlap(cov1, cov2):   # TODO handle mismatch (remove from coverage with higher amount of timestamps) (this is due to timestamp-based indexing)
    return cov1 & cov2


def intensity_histogram(sc, I_data, coverage, bin_width):
    """
    Code partly adapted from SOLER anisotropy tools SEPEvent.overview_plot() method (maintained by Jan Gieseler)
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
        intensity_per_sector = intensity[:,i]
        cov_arr = coverage[direction].to_numpy()
        cov_finite = coverage[direction].notna().to_numpy()
        av_flux = np.where(cov_finite[:,1], intensity_per_sector, np.nan)  
        new_hist = np.where(((Y > cov_arr[:,0]) & (Y < cov_arr[:,2])), av_flux, 0)
        hist = hist + new_hist    
        hist_counts = hist_counts + np.where(new_hist > 0, 1, 0)   # Overlapping bins are calculated as averages

    hist = hist / hist_counts
    hist = np.where(hist > 0, hist, np.nan)

    return hist


def convert_to_bool_coverage(cov, sc, bin_width_deg=1):
    directions = sc.sectors
    
    X, Y = np.meshgrid(cov.index.values, np.linspace(0, 180, int(180 / bin_width_deg)), indexing="ij")
    cov_arr = np.zeros_like(Y, dtype=np.bool_)

    for direction in directions:
        dataf = cov[direction].mask(cov[direction].isna(), -1)     # replace missing values with -1 to exclude from comparison in the loop
        for index, data in dataf.reset_index().iterrows():
            covered = np.ma.masked_inside(Y[index], data["min"], data["max"])
            cov_arr[index] = cov_arr[index] | covered.mask

    return X, Y, cov_arr


def load_random_file(path):
    r_file = np.random.choice(os.listdir(path)).tolist()
    return np.load(path + os.sep + r_file)


def load_wind_event(remove_peaks=False, n_lim=2, *args, **kwargs):
    wind_event = run_SEPevent(*args, **kwargs)
    if remove_peaks:
        wind_event.wind_peak_removal(n_lim=n_lim)
    return wind_event


solo = SoloConstants()
wind = WindConstants()
