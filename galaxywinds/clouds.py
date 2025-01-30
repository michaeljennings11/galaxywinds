# generate clouds

import os

import h5py
import numpy as np
import pandas as pd
import yaml as ym

from galaxywinds import config, constants, utils

enzo_to_colt_dir = config.colt_cubes_dir
colt_out_dir = config.colt_out_dir
bpass_dir = config.bpass_dir
config_files_dir = config.ion_config_dir

cloud_config_file = config.cloud_param_file


def create_coltfile(data, filename):
    enzo_to_colt_file = os.path.join(enzo_to_colt_dir, filename)
    with h5py.File(enzo_to_colt_file, "w") as f:
        # f.attrs['time'] = np.float64(time.to('s'))  # Current simulation time
        f.attrs["nx"] = np.int32(data["nx"])
        f.attrs["ny"] = np.int32(data["ny"])
        f.attrs["nz"] = np.int32(data["nz"])
        f.attrs["n_cells"] = np.int32(data["n_cells"])
        f.create_dataset("bbox", data=data["bbox"])  # Bounding box [cm]
        f["bbox"].attrs["units"] = b"cm"
        f.create_dataset(
            "v",
            data=np.vstack(
                [data["vx"].flatten(), data["vy"].flatten(), data["vz"].flatten()]
            ).T,
            dtype=np.float64,
        )  # Velocities [cm/s]
        f["v"].attrs["units"] = b"cm/s"
        f.create_dataset(
            "rho", data=data["rho"].flatten(), dtype=np.float64
        )  # Density [g/cm^3]
        f["rho"].attrs["units"] = b"g/cm^3"
        f.create_dataset(
            "T", data=data["T"].flatten(), dtype=np.float64
        )  # Temperature [K]
        f["T"].attrs["units"] = b"K"


def sphere(shape, radius, position):
    """Generate an n-dimensional spherical mask."""
    # assume shape and position have the same length and contain ints
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    assert len(position) == len(shape)
    n = len(shape)
    semisizes = (radius,) * len(shape)

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        # this can be generalized for exponent != 2
        # in which case `(x_i / semisize)`
        # would become `np.abs(x_i / semisize)`
        arr += (x_i / semisize) ** 2

    # the inner part of the sphere will have distance below or equal to 1
    return arr <= 1.0


def create_datacube(shape, radius, center, xw, xc):
    cube = np.ones(shape)
    mask = sphere(shape, radius, center)
    cube[~mask] *= xw
    cube[mask] *= xc
    return cube


def all_cubes(shape, radius, center, params):
    cubes = []
    for i, pars in enumerate(params):
        pw, pc = pars
        dc = create_datacube(shape, radius, center, pw, pc)
        cubes.append(dc)
    return cubes


def get_bpass_L0(file=bpass_dir + "/spectra-bin-imf135_100.z020.dat.gz", age=6.0):
    col_names = np.array(["wavelength"])
    ages_n = np.arange(2, 52 + 1, 1)
    ages = np.power(10, 6 + 0.1 * (ages_n - 2))
    log_ages = np.log10(ages)
    log_ages_s = np.array(["%.1f" % number for number in log_ages])
    col_names = np.concatenate((col_names, log_ages_s))
    sed_df = pd.read_csv(file, sep=r"\s+", names=col_names)
    sed_waves = sed_df["wavelength"].values

    fluxes = sed_df[str(age)]
    L0 = np.sum(fluxes) * constants.L_SUN
    return L0


class Ion_config(ym.YAMLObject):
    yaml_tag = "!ionization"

    def __init__(self, options: dict):
        [setattr(self, key, value) for key, value in options.items()]


def generate_ion_config(
    init_dir="ics",
    init_base="colt",
    output_dir="output",
    output_base="ion",
    abundances_base=None,
    abundances_output_base="states",
    cosmological=False,
    n_photons=1e6,
    max_iter=25,
    max_error=1e-3,
    plane_direction="+x",
    Sbol_plane=1e-1,
    UVB_model="HM12",
    free_free=True,
    free_bound=True,
    two_photon=True,
    source_file_Z_age="/Users/mjennings/colt/tables/bpass-spectra-bin-imf135_100.hdf5",
    single_Z=2.0e-2,
    single_age=1.0,
    output_photons=False,
    output_abundances=True,
    output_photoionization=False,
    dust_model="/Users/mjennings/colt/tables/MW_WeingartnerDraine.hdf5",
    metallicity=0.01295,
    silicon_metallicity=7e-4,
    dust_to_metal=0.4,
    silicon_ions=True,
    ion_bins=True,
):
    template_dict = {
        "init_dir": init_dir,
        "init_base": init_base,
        "output_dir": output_dir,
        "output_base": output_base,
        "abundances_base": abundances_base,
        "abundances_output_base": abundances_output_base,
        "cosmological": cosmological,
        "n_photons": int(n_photons),
        "max_iter": max_iter,
        "max_error": max_error,
        "plane_direction": plane_direction,
        "Sbol_plane": Sbol_plane,
        "UVB_model": UVB_model,
        "free_free": free_free,
        "free_bound": free_bound,
        "two_photon": two_photon,
        "source_file_Z_age": source_file_Z_age,
        "single_Z": single_Z,
        "single_age": single_age,
        "output_photons": output_photons,
        "output_abundances": output_abundances,
        "output_photoionization": output_photoionization,
        "dust_model": dust_model,
        "metallicity": metallicity,
        "silicon_metallicity": silicon_metallicity,
        "dust_to_metal": dust_to_metal,
        "silicon_ions": silicon_ions,
        "ion_bins": ion_bins,
    }
    config_dict = {key: val for key, val in template_dict.items() if val is not None}
    return Ion_config(config_dict)


def save_config(configObj, file):
    with open(file, "w") as yaml_file:
        ym.dump(configObj, yaml_file, default_flow_style=False)
    with open(file, "r") as original:
        data = original.read()
    with open(file, "w") as modified:
        modified.write("--- " + data)


def generate_clouds(
    wind_solution, cloud_config_file=cloud_config_file, bpass_model="default"
):

    cloud_params = utils.load_config(cloud_config_file)

    r_arr = np.array(cloud_params["r_array"]) * constants.KPC

    # get wind solution arrays
    rwinds, i_r = utils.find_nearest(
        wind_solution.r, r_arr
    )  # find nearest matching r values
    Mclouds = wind_solution.M_cloud[i_r]
    rclouds = (Mclouds / (4 * np.pi * wind_solution.rho_cloud[i_r] / 3)) ** (1 / 3)
    Tclouds, rho_clouds, vclouds = wind_solution.get_fileparams()[:, :, i_r]

    rc = rclouds
    px_per_rc = cloud_params["res_rcloud"]
    px_sizes = rclouds / px_per_rc
    rc_to_rbox = cloud_params["cloud_box_ratio"]
    rbox_in_px = max(
        round(px_per_rc / rc_to_rbox), px_per_rc + 1
    )  # ensure r_box is at least 1px larger than r_cloud
    rboxs = px_sizes * rbox_in_px

    nx_cube, ny_cube, nz_cube = (rbox_in_px * 2, rbox_in_px * 2, rbox_in_px * 2)
    cube_shape = (nx_cube, ny_cube, nz_cube)

    pos_center = tuple(np.asarray(cube_shape) // 2)

    # create cubes
    for i, idx in enumerate(i_r):
        bbox_cube = np.array(
            [[-rboxs[i], -rboxs[i], -rboxs[i]], [+rboxs[i], +rboxs[i], +rboxs[i]]]
        )
        T_cube, rho_cube, vx_cube = all_cubes(
            cube_shape, px_per_rc, pos_center, wind_solution.get_fileparams()[:, :, idx]
        )
        vy_cube = np.copy(vx_cube) * 0
        vz_cube = vy_cube
        data_dict = {
            "nx": nx_cube,
            "ny": ny_cube,
            "nz": nz_cube,
            "n_cells": nx_cube * ny_cube * nz_cube,
            "bbox": bbox_cube,
            "vx": vx_cube,
            "vy": vy_cube,
            "vz": vz_cube,
            "rho": rho_cube,
            "T": T_cube,
        }
        out_file = f"cube_sphere_{i:04}"
        create_coltfile(data_dict, out_file + ".hdf5")

    #######################
    # create config files #
    #######################
    if bpass_model == "default":
        sed_file = bpass_dir + "/spectra-bin-imf135_100.z020.dat.gz"
        L0 = get_bpass_L0(sed_file)
    Sbols = utils.F_r(rwinds, L0)

    config_files_list = []
    for i, Sbol in enumerate(Sbols):
        ab_out_file = f"states_sphere_{i:04}"
        conf_out_file = f"ion_sphere_{i:04}"
        full_config_file = config_files_dir + "/ion_configs/" + conf_out_file + ".yaml"
        config_files_list.append(full_config_file)
        init_file = f"cube_sphere_{i:04}"
        config_i = generate_ion_config(
            init_dir=enzo_to_colt_dir,
            init_base=init_file,
            output_dir=os.path.join(colt_out_dir, "ionization"),
            output_base=conf_out_file,
            Sbol_plane=float(Sbol),
            abundances_output_base=ab_out_file,
        )
        save_config(config_i, full_config_file)
    return config_files_list
