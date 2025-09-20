# generate clouds

import os

import h5py
import numpy as np
import pandas as pd
import yaml as ym

from galaxywinds import config, constants, utils

enzo_to_colt_dir = config.colt_cubes_dir
colt_out_dir = config.colt_out_dir
model_outs_dir = config.model_outs_dir
model = config.model
model_dir = os.path.join(model_outs_dir, model)
data_cube_dir = os.path.join(model_dir, "data_cubes")


bpass_dir = config.bpass_dir
config_files_dir = os.path.join(model_dir, "config_files")
ionization_dir = os.path.join(model_dir, "ionization")
line_dir = os.path.join(model_dir, "line")

cloud_config_file = config.cloud_param_file


def create_coltfile(data, filename):
    enzo_to_colt_file = os.path.join(data_cube_dir, filename)
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


class Line_config(ym.YAMLObject):
    yaml_tag = "!mcrt"

    def __init__(self, options: dict):
        [setattr(self, key, value) for key, value in options.items()]


def generate_ion_config(
    init_dir="ics",
    init_base="colt",
    output_dir="output",
    output_base="ion",
    abundances_dir=None,
    abundances_output_dir=None,
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
        "abundances_dir": abundances_dir,
        "abundances_base": abundances_base,
        "abundances_output_dir": abundances_output_dir,
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


def generate_line_config(
    init_dir="ics",
    init_base="colt",
    output_dir="output",
    output_subdir="SiII-1260",
    output_base="SiII-1260_sphere",
    abundances_dir=None,
    abundances_base="states_sphere",
    plane_direction="+x",
    L_plane_cont=1,
    plane_beam=True,
    plane_radius_y_bbox=0.4,
    plane_radius_z_bbox=0.4,
    continuum_range=[-600, 1500],
    cosmological=False,
    line="SiII-1260",
    v_turb_kms=None,
    output_mcrt_emission=False,
    output_mcrt_attenuation=False,
    output_proj_emission=False,
    output_proj_attenuation=False,
    metallicity=0.01295,
    silicon_metallicity=0.0007,
    dust_to_metal=0.4,
    dust_model="MW",
    n_photons=10000000,
    output_photons=True,
    photon_file="photons",
    freq_range=[-3000, 3000],
    n_bins=1000,
    image_radius_bbox=1,
    n_pixels=160,
    cameras=[[1, 0, 0]],
):
    template_dict = {
        "init_dir": init_dir,
        "init_base": init_base,
        "output_dir": output_dir,
        "output_subdir": output_subdir,
        "output_base": output_base,
        "abundances_dir": abundances_dir,
        "abundances_base": abundances_base,
        "plane_direction": plane_direction,
        "L_plane_cont": L_plane_cont,
        "plane_beam": plane_beam,
        "plane_radius_y_bbox": plane_radius_y_bbox,
        "plane_radius_z_bbox": plane_radius_z_bbox,
        "continuum_range": continuum_range,
        "cosmological": cosmological,
        "line": line,
        "v_turb_kms": v_turb_kms,
        "output_mcrt_emission": output_mcrt_emission,
        "output_mcrt_attenuation": output_mcrt_attenuation,
        "output_proj_emission": output_proj_emission,
        "output_proj_attenuation": output_proj_attenuation,
        "metallicity": metallicity,
        "silicon_metallicity": silicon_metallicity,
        "dust_to_metal": dust_to_metal,
        "dust_model": dust_model,
        "n_photons": int(n_photons),
        "output_photons": output_photons,
        "photon_file": photon_file,
        "freq_range": freq_range,
        "n_bins": n_bins,
        "image_radius_bbox": image_radius_bbox,
        "n_pixels": n_pixels,
        "cameras": cameras,
    }
    config_dict = {key: val for key, val in template_dict.items() if val is not None}
    return Line_config(config_dict)


def generate_clouds(
    wind_solution, cloud_config_file=cloud_config_file, bpass_model="default"
):
    r = wind_solution.r
    # r_cloud = wind_solution.r_cloud
    r_cloud_start, idx_cloud_start = utils.find_nearest(
        wind_solution.r, np.asarray([wind_solution.r_cloud_start])
    )

    cloud_params = utils.load_config(cloud_config_file)
    if "r_array" in cloud_params:
        r_arr = np.array(cloud_params["r_array"]) * constants.KPC
    elif "n_shells" in cloud_params:
        n_shells = cloud_params["n_shells"]  # number of radial shells
        ir = np.arange(0, wind_solution.r.shape[0])  # radial index array
        ir_partitions = np.array_split(
            ir[0:-1], n_shells
        )  # radial index array partitioned into n shells
        delta_ir = np.array(
            [len(x) for x in ir_partitions]
        )  # width of radial shells in indices
        ir_starts = [x[0] for x in ir_partitions]  # starting index of each radial shell
        idx_clouds = ir_starts + (delta_ir // 2)
        r_arr = r[idx_clouds]
    else:
        raise ValueError(
            "Either r_array or n_shells must be provided in cloud_params file!"
        )
    print(f"r_arr: {r_arr/constants.KPC}")
    print(f"v_c: {wind_solution.v_cloud[idx_clouds]/1e5}")
    print(f"v_w: {wind_solution.v_wind[idx_clouds]/1e5}")

    vturb = cloud_params["vturb"]
    print(f"vturb: {vturb}")
    model_name = cloud_params["model_name"]

    # get wind solution arrays
    rwinds, i_r = utils.find_nearest(
        wind_solution.r, r_arr
    )  # find nearest matching r values
    print(f"Generating clouds at r = {rwinds / constants.KPC} kpc...")
    Mclouds = wind_solution.M_cloud[i_r]
    rclouds = (Mclouds / (4 * np.pi * wind_solution.rho_cloud[i_r] / 3)) ** (1 / 3)
    Tclouds, rho_clouds, vclouds = wind_solution.get_fileparams()[:, :, i_r]
    Zcloud = wind_solution.Z_cloud[i_r]

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
    for i, idx in enumerate(idx_clouds):
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

    ###########################
    # create ion config files #
    ###########################
    if "Sbol" in cloud_params:  # Check if Sbol is hardcoded for each radii
        Sbol = cloud_params["Sbol"]
        Sbols = Sbol * np.ones(rwinds.shape)
    else:  # Calculate Sbol from radius and bpass luminosity
        if bpass_model == "default":
            sed_file = bpass_dir + "/spectra-bin-imf135_100.z020.dat.gz"
            L0 = get_bpass_L0(sed_file)
        Sbols = utils.F_r(rwinds, L0)

    ion_config_files_list = []
    for i, Sbol in enumerate(Sbols):
        ab_out_file = f"states_sphere_{i:04}"
        conf_out_file = f"ion_sphere_{i:04}"
        full_config_file = config_files_dir + "/ion_configs/" + conf_out_file + ".yaml"
        ion_config_files_list.append(full_config_file)
        init_file = f"cube_sphere_{i:04}"
        ab_in_dir = None
        ab_in_file = None
        Z = Zcloud[i]
        if i > 0:
            ab_in_dir = ionization_dir
            ab_in_file = f"states_sphere_{i-1:04}"
        config_i = generate_ion_config(
            init_dir=data_cube_dir,  # initial conditions data cube directory
            init_base=init_file,  # initial conditions data cube base file
            output_dir=ionization_dir,  # output ion file directory
            output_base=conf_out_file,  # output ion file base file
            Sbol_plane=float(Sbol),
            abundances_output_dir=ionization_dir,  # abundances output directory
            abundances_output_base=ab_out_file,  # abundances output file base
            abundances_dir=ab_in_dir,  # abundances initial conditions directory
            abundances_base=ab_in_file,  # abundances initial conditions file base
            metallicity=float(Z),
        )
        utils.save_config(config_i, full_config_file)

    ############################
    # create line config files #
    ############################
    spec_line = "SiII-1260"
    freq_range = np.array([-3000, 3000])
    vwinds = wind_solution.v_wind[i_r]
    line_config_files_list = []
    for i, vwind in enumerate(vwinds):
        cont_range = freq_range + vwind / constants.KM
        conf_out_file = f"{spec_line}_sphere_{i:04}"
        full_config_file = config_files_dir + "/line_configs/" + conf_out_file + ".yaml"
        line_config_files_list.append(full_config_file)
        init_file = f"cube_sphere_{i:04}"
        ab_in_file = f"states_sphere_{i:04}"
        config_i = generate_line_config(
            init_dir=enzo_to_colt_dir,
            init_base=init_file,
            output_dir=line_dir,
            output_subdir=spec_line,
            output_base=conf_out_file,
            abundances_dir=ionization_dir,  # abundances initial conditions directory
            abundances_base=ab_in_file,
            continuum_range=[float(cont_range[0]), float(cont_range[1])],
            line=spec_line,
            v_turb_kms=vturb,
        )
        utils.save_config(config_i, full_config_file)

    return ion_config_files_list, line_config_files_list
