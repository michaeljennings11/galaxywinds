import os

import h5py
import numpy as np
from matplotlib import pyplot as plt

PROJ_ROOT = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(PROJ_ROOT, "data")
CLOUD_DIR = os.path.join(DATA_DIR, "cloud_data")
LINE_DIR = os.path.join(DATA_DIR, "line_data")


def read_boxvx(cloud_file="cloud-data-000.h5"):
    filename = f"{CLOUD_DIR}/{cloud_file}"
    with h5py.File(filename, "r") as f:
        box_vx, _, _ = f[list(f.keys())[0]].attrs["frame_velocity"]
    return box_vx


def read_photons(line="SiII-1260"):
    """
    Returns dataset of photon information from COLT output for desired line.
    """
    filename = f"{LINE_DIR}/{line}_photons.hdf5"
    ds = {"filename": filename}
    with h5py.File(filename, "r") as f:
        for key in f.attrs.keys():
            ds[key] = f.attrs[key]
        ds["direction"] = f["direction"][:]
        ds["frequency"] = f["frequency"][:]
    return ds


def mu_alpha(npoints: int, alpha: float = 90) -> np.ndarray:
    if alpha == 90:
        xs = np.random.uniform(-1, 1, npoints)
    else:
        xs_left = np.cos(np.radians(alpha))
        xs = np.concatenate(
            (
                np.random.uniform(-1, -xs_left, npoints // 2),
                np.random.uniform(xs_left, 1, npoints - npoints // 2),
            )
        )
    return xs


def mu_psi(xs: np.ndarray, psi: float = 0) -> np.ndarray:
    thetas = np.arccos(xs)  # get angles in radians from x-axis
    ys = np.sin(thetas) * np.random.choice([-1, 1], len(xs))  # generate y-coords
    xs_r = xs * np.cos(np.radians(psi)) - ys * np.sin(
        np.radians(psi)
    )  # x-coords of rotated points around z-axis by angle psi
    return xs_r


def xr_points(npoints: int, alpha: float = 90, psi: float = 0) -> np.ndarray:
    """
    Return x-coordinates of points uniformly distributed on the surface of unit-sphere.

    Parameters
    ----------
    npoints : int
        Number of points to generate.
    alpha : float, optional
        Half-angle solid angle in degrees. Default is 90 equating to full 4pi radian sphere.
    psi : float, optional
        Angle in degrees to rotate solid angle away from x-axis. Default is 0.
    """
    if not ((0 <= alpha <= 90) & (0 <= psi <= 90)):
        raise ValueError("Angles alpha and psi must be within 0 and 90 degrees")
    xs = mu_alpha(npoints, alpha)
    if (psi != 0) & (alpha != 90):
        xs = mu_psi(xs, psi)
    return xs


def build_spec(ds: dict, mu_dist: np.ndarray, dv: float = 3):
    mu_all = ds["direction"][:, 0]  # all mu=cos(theta) of photons
    freq_all = ds["frequency"]  # all frequencies in velocity-space (km/s) of photons
    freq_min = np.amin(freq_all)
    freq_max = np.amax(freq_all)
    dmu = 0.01
    bins = np.arange(freq_min, freq_max, dv)
    flux_arr = np.zeros(len(bins) - 1)
    for i, mu in enumerate(mu_dist):
        mu_left = mu - dmu
        mu_right = mu + dmu
        mask = (mu_all >= mu_left) & (mu_all <= mu_right)
        freq = freq_all[mask]
        hist = np.histogram(freq, bins=bins)
        flux_mu = hist[0]
        flux_arr += flux_mu

    wave_arr = hist[1][:-1]
    mask_cont = (wave_arr < -400) | ((wave_arr > 200) & (wave_arr < 900))
    flux_cont = np.median(flux_arr[mask_cont])
    flux_norm = flux_arr / flux_cont

    return wave_arr, flux_norm


if __name__ == "__main__":
    seed = np.random.seed(0)

    # Get COLT and sim data
    ds_photons = read_photons()
    box_vx = read_boxvx()

    # Place clouds
    nclouds = 500
    alpha = 90  # spherical outflow
    psi = 0
    mu_clouds = xr_points(nclouds, alpha, psi)

    # Build spectrum
    wave, flux = build_spec(ds_photons, mu_clouds)

    # Plot spectrum
    xlims = [(-200, 200), (920, 1250)]

    fig, axes = plt.subplots(
        1, 2, sharey=True, gridspec_kw={"wspace": 0.03}, figsize=(6, 3)
    )

    for xlim, ax in zip(xlims, axes):
        ax.axhline(1, color="k", alpha=0.2)
        ax.step(wave, flux, where="mid")
        ax.set_xlim(xlim)
        ax.set_ylim(0)
        ax.set_xlabel(r"velocity (km/s)")
    axes[0].set_ylabel("normalized flux")
    plt.show()
