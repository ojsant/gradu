import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess

from scipy.ndimage import gaussian_filter1d
from pyonset import Onset, BootstrapWindow
from anisotropy import run_SEPevent

import sys
import logging
filehandler = logging.FileHandler(filename="wind_events.log", encoding="utf-8")
streamhandler = logging.StreamHandler(sys.stdout)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', 
                    datefmt='%m/%d/%Y %H:%M:%S', handlers=[filehandler, streamhandler], force=True)


def check_for_gaps(data, start, end, max_gap=1):
    flux_finite = data.dropna()
    for i in range(len(flux_finite[start:end]) - 1):
        if flux_finite.index[i + 1] - flux_finite.index[i] > pd.Timedelta(hours=max_gap):
            return True
    return False

data_path = f"{os.getcwd()}{os.sep}data"
csv_path = f"{os.getcwd()}{os.sep}wind_events.csv"

rc = subprocess.call(["./download.sh", "l1", "2023"])

for d in range(365):
    
    date = pd.to_datetime("2023-1-1") + pd.Timedelta(days=d)
    logging.info(f"Analyzing flux for {date}")
    start = date - pd.Timedelta(days=1)
    end = date + pd.Timedelta(days=2)    # off by one, so three days are loaded now
    try:
        event = run_SEPevent(data_path, spacecraft_instrument="Wind 3DP", starttime=start, endtime=end, 
                            species="e", channels=3, averaging="60min")         # maybe wget SFPD, SOPD from CDAWeb if SSL Berkeley is down
    except ValueError:
        continue

    # Data smoothing and finite difference optimization
    data = pd.Series(event.I_data[:,3], index=event.I_times)
    data_middle = data[date:(date + pd.Timedelta(days=1))]  # only consider the middle day for onset detection
    data_smooth = gaussian_filter1d(data_middle, sigma=1)   
    data_smooth_diff = np.diff(data_smooth)
    arg_optima = []
    for i in range(len(data_smooth_diff)-1):
        if (data_smooth_diff[i + 1] > 0 and data_smooth_diff[i] < 0) or (data_smooth_diff[i + 1] < 0 and data_smooth_diff[i] > 0):
            arg_optima.append(i+1)

    # Plot data, smoothed data and optima
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.plot(data_middle.index, data_smooth)
    ax.plot(data_middle.index[arg_optima], data_smooth[arg_optima], "k+")
    ax.set_yscale("log")
    ax.set_ylabel("Intensity")
    fig.suptitle(f"Optima for {start} - {end}")
    fig.tight_layout()
    plot_dir = f"{os.getcwd()}{os.sep}plots"
    os.makedirs(plot_dir, exist_ok=True)
    fname = plot_dir + os.sep + f"Wind_{start.strftime("%Y%m%d")}-{end.strftime("%Y%m%d")}"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)

    # Rough onset determination with Poisson-CUSUM
    onset = Onset(start, end, "Wind", "3DP", "e", "l3", viewing="3", data_path=data_path)
    for i in range(len(arg_optima) - 1):
        n = 0
        opt1 = arg_optima[i]
        opt2 = arg_optima[i+1]
        if (data_middle.iloc[opt2] / data_middle.iloc[opt1] > 10) \
        and (data_middle.iloc[opt2] > data_middle.iloc[opt1]) \
        and check_for_gaps(data_middle, data_middle.index[opt1], data_middle.index[opt2]) is False:
            
            logging.info(f"Found a potential event between {data_middle[arg_optima].index[i]} - {data_middle[arg_optima].index[i+1]}")
            bg_end = data_middle[arg_optima].index[i]
            bg_start = bg_end - pd.Timedelta(hours=8)
            background = BootstrapWindow(bg_start, bg_end, bootstraps=1000)
            single_onset_stats, flux_series = onset.cusum_onset(channels=[3], background_range=background, 
                                                                        viewing="3", resample="1min", cusum_minutes=60,
                                                                        plot=False)
            try:
                onset_date = single_onset_stats[-1].date()
                onset_time = single_onset_stats[-1].time()
                logging.info(f"Poisson-CUSUM result: found onset {onset_date} {onset_time}\n" \
                             f"using background {bg_start} - {bg_end}")
                
                # Write to CSV
                with open(csv_path, "+r", encoding="utf-8") as fp:
                    event_no = len(fp.readlines())
                    fp.write(f"{event_no},{onset_date},{onset_time},{bg_start},{bg_end}\n")
                    logging.info(f"Results successfully saved to {csv_path}")
                    fp.close()
                
                n += 1

            except ValueError:
                logging.info(f"Sufficient minimum ({opt1}) and maximum ({opt2}) were identified, but no onset was found")

    logging.info(f"Analysis for {date} finished, {n} onsets found")
