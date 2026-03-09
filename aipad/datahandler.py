"""
Author: Juho Lankinen (jumila@utu.fi)
"""

import h5py
import glob
import numpy as np
from pathlib import Path
from itertools import product


class WindDataHandler:
    def __init__(self):
        pass

    @classmethod
    def create_file(cls, filename: str, npz_files_path="./data/intensities/1min_1deg/*.npz"):
        if not filename.endswith(".h5"):
            raise RuntimeError("Provide the filename as .h5 file.")
        npz_files = glob.glob(npz_files_path)

        with h5py.File(filename, "w") as h5file:
            for i, fn in enumerate(npz_files):
                data = np.load(fn)
                # Keys will be the filenames without 1min_1deg.npz ending.
                root_name = Path(fn).name[:-14]
                group = h5file.create_group(root_name)

                for key in ["full", "reduced", "intensity_data"]:
                    group.create_dataset(key, data=data[key], compression='gzip')

    @classmethod
    def save_event(cls, filename: str, groupname: str, arrays: dict):
        with h5py.File(filename, "a") as file:
            # require_group() returns an existing group or creates one if it doesn't exist
            group = file.require_group(groupname)
            for key in arrays.keys():
                group.create_dataset(key, data=arrays[key], compression='gzip')

    @classmethod
    def load_file(cls, filename: str) -> dict:
        d = {}
        keys = ["full", "intensities", "times"] \
            + [f"reduced_{p:.1f}" for p in np.arange(1, 6) / 10] \
            + [f"reduced_{p:.1f}_mask" for p in np.arange(1, 6) / 10]
        
        for key in keys:
            d[key] = []

        with h5py.File(filename, "r") as file:
            for k1, k2 in product(file.keys(), keys):
                d[k2].append(file[k1][k2][:])

        return d


if __name__ == '__main__':

    # To create the h5-file (created to where this script is run)
    print("Creating h5 file...")
    WindDataHandler.create_file("test_file.h5")

    # To load the files:
    print("Loading h5 file...")
    hist_list, reduced_hist_list, intensity_list = WindDataHandler.load_file("test_file.h5")

    print(f"Some loaded value: {hist_list[0][0][100]}")
