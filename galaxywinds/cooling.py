# Cooling functions for FB20


import h5py
import numpy as np
from scipy import interpolate

from galaxywinds import constants

# Cooling curve as a function of density, temperature, metallicity

Cooling_File = "/Users/mjennings/Projects/MultiphaseGalacticWind/CoolingTables/z_0.000.hdf5"  ### From Wiersma et al. (2009) appropriate for z=0 UVB
f = h5py.File(Cooling_File, "r")
i_X_He = -3
Metal_free = f.get("Metal_free")
Total_Metals = f.get("Total_Metals")
log_Tbins = np.array(np.log10(Metal_free["Temperature_bins"]))
log_nHbins = np.array(np.log10(Metal_free["Hydrogen_density_bins"]))
Cooling_Metal_free = np.array(Metal_free["Net_Cooling"])[
    i_X_He
]  ##### what Helium_mass_fraction to use    Total_Metals = f.get('Total_Metals')
Cooling_Total_Metals = np.array(Total_Metals["Net_cooling"])
HHeCooling = interpolate.RectBivariateSpline(log_Tbins, log_nHbins, Cooling_Metal_free)
ZCooling = interpolate.RectBivariateSpline(log_Tbins, log_nHbins, Cooling_Total_Metals)
f.close()
Zs = np.logspace(-2, 1, 31)
Lambda_tab = np.array(
    [
        [
            [HHeCooling.ev(lT, ln) + Z * ZCooling.ev(lT, ln) for Z in Zs]
            for lT in log_Tbins
        ]
        for ln in log_nHbins
    ]
)
Lambda_z0 = interpolate.RegularGridInterpolator(
    (log_nHbins, log_Tbins, Zs), Lambda_tab, bounds_error=False, fill_value=-1e-30
)


def tcool_P(T, P, metallicity):
    """
    cooling time function
    T in units of K
    P in units of K * cm**-3
    metallicity in units of solar metallicity
    """
    T = np.where(T > 10**8.98, 10**8.98, T)
    T = np.where(T < 10**2, 10**2, T)
    nH_actual = P / T * (constants.MU / constants.MU_H)
    nH = np.where(nH_actual > 1, 1, nH_actual)
    nH = np.where(nH < 10**-8, 10**-8, nH)
    return (
        (1.0 / (constants.GAMMA - 1.0))
        * (constants.MU_H / constants.MU)
        * constants.K_B
        * T
        / (nH_actual * Lambda_z0((np.log10(nH), np.log10(T), metallicity)))
    )


def Lambda_T_P(T, P, metallicity):
    """
    cooling curve function as a function of
    T in units of K
    P in units of K * cm**-3
    metallicity in units of solar metallicity
    above nH = 0.9 * cm**-3 there is no more density dependence
    """
    nH = P / T * (constants.MU / constants.MU_H)
    if nH > 0.9:
        nH = 0.9
    return Lambda_z0((np.log10(nH), np.log10(T), metallicity))


Lambda_T_P = np.vectorize(Lambda_T_P)


def Lambda_P_rho(P, rho, metallicity):
    """
    cooling curve function as a function of
    P in units of erg * cm**-3
    rho in units of g * cm**-3
    metallicity in units of solar metallicity
    above nH = 0.9 * cm**-3 there is no more density dependence
    """
    nH = rho / (constants.MU_H * constants.M_P)
    T = P / constants.K_B / (rho / (constants.MU * constants.M_P))
    if nH > 0.9:
        nH = 0.9
    return Lambda_z0((np.log10(nH), np.log10(T), metallicity))


Lambda_P_rho = np.vectorize(Lambda_P_rho)
