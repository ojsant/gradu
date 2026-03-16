import h5py

KEYS = ["Intensity", "MagField", "Coverage"]


class WindDataHandler:
    def __init__(self):
        pass

    @classmethod
    def save_event(cls, filename: str, groupname: str, arrays: dict, attrs: dict):
        with h5py.File(filename, "a") as file:
            # require_group() returns an existing group or creates one if it doesn't exist
            group = file.require_group(groupname)
            for key in arrays.keys():
                dset = group.create_dataset(key, data=arrays[key], compression='gzip')
                # Keys of attribute dict are the metadata quantities
                for attr in attrs[key].keys():
                    dset.attrs[attr] = attrs[key][attr]

    @classmethod
    def load_file(cls, filename: str) -> dict:
        d = {}
        with h5py.File(filename, "r") as file:
            for event in file.keys():
                for dset in KEYS:
                    d[dset] = file[event][dset][:]
                    d[f"{dset}Meta"] = {}
                    for attr in file[event][dset].attrs.keys():
                        d[f"{dset}Meta"][attr] = file[event][dset].attrs[attr]

        return d
