# Useful utility functions
import os
import errno
import numpy as np
import yaml as ym

from galaxywinds import constants


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file
            raise  # re-raise exception if a different error occurred


def load_config(file):
    with open(file, "r") as f:
        params = ym.safe_load(f)
    return params


def save_config(configObj, file):
    with open(file, "w") as yaml_file:
        ym.dump(configObj, yaml_file, default_flow_style=None)
    with open(file, "r") as original:
        data = original.read()
    with open(file, "w") as modified:
        modified.write("--- " + data)


class Wind:
    def __init__(self, params: dict):
        [setattr(self, key, value) for key, value in params.items()]
        self.rho_cloud = (
            self.P_wind
            * (constants.MU * constants.M_P)
            / (constants.K_B * self.T_cloud)
        )

    def get_fileparams(self):
        return np.array(
            [
                (self.T_wind, self.T_cloud),
                (self.rho_wind, self.rho_cloud),
                (self.v_wind, self.v_cloud),
            ]
        )


def find_nearest(array, values):
    array = np.asarray(array)
    values = values.reshape(values.shape[0], 1)
    idx = np.abs(array - values).argmin(axis=1)
    return array[idx], idx


def F_r(r, L0):
    return L0 / (4 * np.pi * (r) ** 2)
