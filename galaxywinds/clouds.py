# generate clouds

import os

import h5py
import numpy as np

from galaxywinds import utils

enzo_to_colt_dir = "/Users/mjennings/Projects/galaxy_winds/data/interim/enzo_to_colt"


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
            ),
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


def generate_clouds(r_arr, wind_solution, bpass_model="default"):
    # get wind solution arrays
    rwinds, i_r = utils.find_nearest(
        wind_solution.r, r_arr
    )  # find nearest matching r values
    Mclouds = wind_solution.M_cloud[i_r]
    rclouds = (Mclouds / (4 * np.pi * wind_solution.rho_cloud[i_r] / 3)) ** (1 / 3)
    Tclouds, rho_clouds, vclouds = wind_solution.get_fileparams()[:, :, i_r]

    rc = rclouds
    px_per_rc = 8
    px_sizes = rclouds / px_per_rc
    rc_to_rbox = 0.75
    rbox_in_px = max(
        round(px_per_rc / rc_to_rbox), px_per_rc + 1
    )  # ensure r_box is at least 1px larger than r_cloud
    rboxs = px_sizes * rbox_in_px

    nx_cube, ny_cube, nz_cube = (rbox_in_px, rbox_in_px, rbox_in_px)
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
    # if bpass_model == "default":
    #     sed_file = bpass_dir + "/spectra-bin-imf135_100.z020.dat.gz"
    #     L0 = get_bpass_L0(sed_file)
    # Sbols = F_r(rwinds, L0)

    # for i, Sbol in enumerate(Sbols):
    #     ab_out_file = f"states_sphere_{i:04}"
    #     conf_out_file = f"ion_sphere_{i:04}"
    #     init_file = f"cube_sphere_{i:04}"
    #     config_i = generate_ion_config(
    #         init_base=init_file,
    #         output_base=conf_out_file,
    #         Sbol_plane=float(Sbol),
    #         abundances_output_base=ab_out_file,
    #     )
    #     save_config(
    #         config_i, config_files_dir + "/ion_configs/" + conf_out_file + ".yaml"
    #     )
