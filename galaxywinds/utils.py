# Useful utility functions

import numpy as np
import yaml as ym

from galaxywinds import constants


def load_config(file):
    with open(file, "r") as f:
        params = ym.safe_load(f)
    return params


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
