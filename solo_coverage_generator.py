# TODO 2.12. check for empty data files

from solo_epd_loader import epd_load

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import sys
import logging
filehandler = logging.FileHandler(filename="solo_cov.log", encoding="utf-8")
streamhandler = logging.StreamHandler(sys.stdout)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S', handlers=[filehandler, streamhandler], force=True)

SOLO_SECTORS = ["sun", "asun", "north", "south"]
WIND_SECTORS = [f"P{i}" for i in range(8)]

SOLO_START = dt.datetime(2020, 11, 2)
SOLO_END = dt.datetime(2025, 5, 31)

def get_sectors(sc):
    if sc.lower() == "solo":
        return SOLO_SECTORS

    elif sc.lower() == "wind":
        return WIND_SECTORS

def convert_to_bool_coverage(cov, sc, bin_width_deg=1):
    directions = get_sectors(sc)
    
    X, Y = np.meshgrid(cov.index.values, np.linspace(0, 180, int(180 / bin_width_deg)), indexing="ij")
    cov_arr = np.zeros_like(Y, dtype=np.bool_)

    for direction in directions:
        dataf = cov[direction].mask(cov[direction].isna(), -1)     # replace missing values with -1 to exclude from comparison in the loop
        for index, data in dataf.reset_index().iterrows():
            covered = np.ma.masked_inside(Y[index], data["min"], data["max"])
            cov_arr[index] = cov_arr[index] | covered.mask

    return X, Y, cov_arr


if __name__ == "__main__":
    sc = "solo"
    data_path = f"{os.getcwd()}{os.sep}data"
    max_days = (SOLO_END - SOLO_START).days

    try:
        start = int(sys.argv[1])
    except (IndexError, ValueError):
        start = 0
    logging.info(f"Run start: {start}")

    
    for i in range(start, max_days):
        logging.info(f"Day {i}")
        save_path = os.getcwd() + os.sep + "coverages" + os.sep + sc + os.sep
        os.makedirs(save_path, exist_ok=True)
        date = SOLO_START + dt.timedelta(days=i)
        logging.info(f"Forming SolO coverages from date {date}")
        try:
            df, df_rtn, df_hci, energies_dict, metadata_dict = epd_load("ept", startdate=date, 
                                                                        level="l3", autodownload=True,
                                                                        path=data_path, pos_timestamp="start")

            if (len(df.index) // 60) < 12:
                logging.info(f"Day {i}: Data file not long enough to generate 12 hour coverage, skipping...")
                continue

            pa_cols = [f"Pitch_Angle_{dir}" for dir in ["S", "A", "N", "D"]]
            pa_sigma_cols = [f"Pitch_Angle_Sigma_{dir}" for dir in ["S", "A", "N", "D"]]
            ind = pd.MultiIndex.from_product([SOLO_SECTORS, ["min", "center", "max"]])
            df_cov = pd.DataFrame(index=df.index, columns=ind)
            for direction, pa_col, pa_sigma_col in zip(SOLO_SECTORS, pa_cols, pa_sigma_cols):
                df_cov[(direction, "center")] = df[pa_col]
                df_cov[(direction, "min")] = df[pa_col] - df[pa_sigma_col]
                df_cov[(direction, "max")] = df[pa_col] + df[pa_sigma_col]

            X1, Y1, solo_cov = convert_to_bool_coverage(df_cov, sc="SolO")

            # Split day in half to generate two 12 hour coverages
            if len(solo_cov) >= 24 * 60:
                cov1 = solo_cov[:12*60,:]
                cov2 = solo_cov[12*60:24*60, :]
                np.save(f"{save_path}{date.strftime("%Y%m%d")}_1", arr=cov1, allow_pickle=False)
                np.save(f"{save_path}{date.strftime("%Y%m%d")}_2", arr=cov2, allow_pickle=False)
                logging.info(f"Day {i}: 2 coverages")

            # If length is less than 24 but more than 12 hours, generate just one (randomly picked)
            elif len(solo_cov) >= 12*60 and len(solo_cov) < 24*60:
                random_start = np.random.randint(0, len(solo_cov) - 12*60)
                cov = solo_cov[random_start:random_start+12*60]
                np.save(f"{save_path}{date.strftime("%Y%m%d")}_1", arr=cov, allow_pickle=False)
                logging.info(f"Day {i}: 1 coverage")

        except UnboundLocalError:
            logging.info(f"No data was found, skipping...")
