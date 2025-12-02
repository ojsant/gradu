import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess
import traceback
from time import sleep
import requests

from scipy.signal import medfilt
from pyonset import Onset, BootstrapWindow
from seppy.util import resample_df
from anisotropy import run_SEPevent

import sys
import logging
filehandler = logging.FileHandler(filename="wind_events.log", encoding="utf-8")
streamhandler = logging.StreamHandler(sys.stdout)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', 
                    datefmt='%m/%d/%Y %H:%M:%S', handlers=[filehandler, streamhandler], force=True)

import warnings

warnings.filterwarnings("ignore", "WARNING: background mean is nan!", category=UserWarning, module="pyonset")

def check_for_gaps(data, start, end, max_gap=1):
    flux_finite = data.dropna()
    for i in range(len(flux_finite[start:end]) - 1):
        if flux_finite.index[i + 1] - flux_finite.index[i] > pd.Timedelta(hours=max_gap):
            return True
    return False

data_path = f"{os.getcwd()}{os.sep}data"
csv_path = f"{os.getcwd()}{os.sep}wind_events.csv"
plot_dir = f"{os.getcwd()}{os.sep}plots_sigma2"
os.makedirs(plot_dir, exist_ok=True)     

#year = sys.argv[1]
#rc = subprocess.call(f'wget -r -nv --show-progress --tries=10 -nc -nH -np -nd -P "data" -A "*.cdf" "https://cdaweb.gsfc.nasa.gov/pub/data/wind/3dp/3dp_sfpd/{year}/"'.split(" "))

for d in range(180):
    try:
        date = pd.to_datetime(f"2023-01-01") + pd.Timedelta(days=d)
        logging.info(f"Analyzing flux for {date}")

        start = date - pd.Timedelta(days=1)
        end = date + pd.Timedelta(days=2)    # off by one, so three days are loaded
        
        event = run_SEPevent(data_path, spacecraft_instrument="Wind 3DP", starttime=start, endtime=end, 
                            species="e", channels=3, averaging="5min")         # maybe wget SFPD, SOPD from CDAWeb if SSL Berkeley is down
        
        
        # Do median filtering with kernel size of 3. This alters the underlying data statistics but easily removes erroneous peak values.
        # The point is only to get a rough estimate for the onset.
        data = pd.Series(event.I_data[:,3], index=event.I_times)

        # Rough onset determination with median value from "rolling" Poisson-CUSUM:
        # slide a window over the whole day, calculating Poisson-CUSUM estimate with each window.
        # Mode of determined onsets is taken to be the (or an) onset
        onset = Onset(start, end, "Wind", "3DP", "e", "l3", viewing="3", data_path=data_path)
        resample = "5min"
        window_len = 6
        n = 0
        onsets = []
        onset_found = False

        #background = BootstrapWindow(window_start, window_end, bootstraps=1000)
        
        for i in range(24):
            window_end = date + pd.Timedelta(hours=i) # from start of the middle day
            window_start = window_end - pd.Timedelta(hours=window_len)
            #background = BootstrapWindow(window_start, window_end, bootstraps=1000)
            single_onset_stats, flux_series = onset.cusum_onset(channels=3, background_range=[window_start, window_end], 
                                                                        viewing="3", resample=resample, cusum_minutes=120,
                                                                        sigma_multiplier=2, plot=False, erase=True)     # erase = True -> median filtering with kernel_size=5 (my own mod)
            
            if isinstance(single_onset_stats["onset_time"], pd._libs.tslibs.NaTType):
                continue
            else:
                logging.info(f"Found onset with background {window_start} - {window_end}: {single_onset_stats["onset_time"]}")
                onsets.append(single_onset_stats["onset_time"])
                onset_found = True

        if onset_found:
            onset_times = pd.Series(onsets) 
            onset_result = onset_times.mode()[0] 

            # NOTE: for now, only take the most common value. Decide how to handle days with multiple events, if there are any
            # if len(onset_modes) > 1:
            #     pass

            logging.info(f"Determined onset: {onset_result}")

            # Write to CSV
            with open(csv_path, "+r", encoding="utf-8") as fp:
                event_no = len(fp.readlines())
                fp.write(f"{event_no},{onset_result.date()},{onset_result.time()}\n")
                logging.info(f"Results successfully saved to {csv_path}")
                fp.close()

        else:
            logging.info(f"No onsets found for {date}")

        fig, ax = plt.subplots()

        ax.step(flux_series.index, flux_series)
        if onset_found:
            ax.axvline(onset_result, color="red")

        ax.set_yscale("log")
        ax.set_ylabel("Intensity")
        fig.suptitle(f"Onset determination for {date}")
        fig.tight_layout()
        
        fname = plot_dir + os.sep + f"Wind_{date.strftime("%Y%m%d")}"
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
    except requests.exceptions.ReadTimeout:
        d -= 1
        sleep(60)
    except Exception:
        logging.info(f"{traceback.format_exc()}")
