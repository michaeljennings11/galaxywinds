# Construct a single wind shell at r

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


def randSphericalCap(npoints: int = 100, alpha: float = 90) -> np.ndarray:
    alpha = np.radians(alpha)
    phi = np.random.uniform(0, 1, npoints) * 2 * np.pi
    x = np.random.uniform(0, 1, npoints) * (1 - np.cos(alpha)) + np.cos(alpha)
    y = np.sqrt(1 - x**2) * np.cos(phi)
    z = np.sqrt(1 - x**2) * np.sin(phi)
    r = np.asarray([x, y, z])
    return r


def rotateWind(r: np.ndarray, psi: float = 0) -> np.ndarray:
    if psi != 0:
        psi = np.radians(psi)
        cp = np.cos(psi)
        sp = np.sin(psi)
        Rz = np.matrix([[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]])
        r = np.asarray(Rz * r)
    return r


def randWindPoints(npoints: int = 100, alpha: float = 90, psi: float = 0):
    """
    Return x,y,z coordinates of points uniformly distributed on the surface of unit-sphere.

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
    r1 = randSphericalCap(npoints // 2, alpha)  # generate 1st cone coordinates
    r2 = randSphericalCap(
        npoints - npoints // 2, alpha
    )  # generate 2nd cone coordinates
    if psi != 0:
        r1 = rotateWind(r1, psi)  # rotate 1st cone
    r2 = rotateWind(r2, psi + 180)  # rotate 2nd cone coordinates
    return np.concatenate((r1, r2), axis=1)


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
    mu_clouds, _, _ = randWindPoints(nclouds, alpha, psi)

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
