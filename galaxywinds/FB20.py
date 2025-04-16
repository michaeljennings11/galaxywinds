# Code to solve for Fielding & Bryan (2020) multiphase wind solution
# Original code written by Drummond Fielding


import numpy as np
import yaml as ym
from scipy import interpolate
from scipy.integrate import solve_ivp

from galaxywinds import config, constants, cooling, utils

wind_config_file = config.wind_param_file


def run_FB20(wind_config_file=wind_config_file):

    wind_params = utils.load_config(wind_config_file)

    # param groups
    global_params = wind_params["global"]
    feedback_params = wind_params["feedback"]
    model_params = wind_params["model"]
    sonic_params = wind_params["sonic"]
    singlephase_params = wind_params["singlephase"]
    multiphase_params = wind_params["multiphase"]

    # global params
    SFR = global_params["SFR"] * constants.M_SUN / constants.YR
    eta_E = global_params["eta_E"]
    eta_M = global_params["eta_M"]
    eta_M_cold = global_params["eta_M_cold"]
    log_M_cloud_init = global_params["log_M_cloud_init"]

    # feedback params
    E_SN = feedback_params["E_SN"]  ## energy per SN in erg
    mstar = feedback_params["mstar"] * constants.M_SUN  ## mass of stars formed per SN
    M_cloud_min = (
        feedback_params["M_cloud_min"] * constants.M_SUN
    )  ## minimum mass of clouds
    Mdot = eta_M * SFR
    Edot = eta_E * (E_SN / mstar) * SFR

    # model params
    CoolingAreaChiPower = model_params["CoolingAreaChiPower"]
    ColdTurbulenceChiPower = model_params["ColdTurbulenceChiPower"]
    TurbulentVelocityChiPower = model_params["TurbulentVelocityChiPower"]
    Mdot_coefficient = model_params["Mdot_coefficient"]
    Cooling_Factor = model_params["Cooling_Factor"]
    drag_coeff = model_params["drag_coeff"]
    f_turb0 = model_params["f_turb0"]
    v_circ0 = model_params[
        "v_circ0"
    ]  # gravitational potential assuming isothermal potential
    Z_wind_init = model_params["Z_wind_init"] * constants.Z_SOLAR
    half_opening_angle = model_params["half_opening_angle"] * np.pi
    Omwind = 4 * np.pi * (1.0 - np.cos(half_opening_angle))

    # sonic params
    r_sonic = sonic_params["r_sonic"] * constants.PC
    epsilon = sonic_params[
        "epsilon"
    ]  ## define a small number to jump above and below sonic radius / mach = 1
    Mach0 = 1.0 + epsilon
    v0 = np.sqrt(Edot / Mdot) * (1 / ((constants.GAMMA - 1) * Mach0) + 1 / 2.0) ** (
        -1 / 2.0
    )  ## velocity at sonic radius
    rho0 = Mdot / (Omwind * r_sonic**2 * v0)  ## density at sonic radius
    P0 = rho0 * v0**2 / Mach0**2 / constants.GAMMA  ## pressure at sonic radius
    rhoZ0 = rho0 * Z_wind_init
    # print(
    #     "v_wind = %.1e km/s  n_wind = %.1e cm^-3  P_wind = %.1e kb K cm^-3"
    #     % (v0 / 1e5, rho0 / (constants.MU * constants.M_P), P0 / constants.K_B)
    # )
    Edot_per_Vol = Edot / (4 / 3.0 * np.pi * r_sonic**3)  # source terms from SN
    Mdot_per_Vol = Mdot / (4 / 3.0 * np.pi * r_sonic**3)  # source terms from SN

    def Multiphase_Wind_Evo(r, state):
        """
        Calculates the derivative of v_wind, rho_wind, Pressure, rhoZ_wind, M_cloud, v_cloud, and Z_cloud.
        Used with solve_ivp to calculate steady state structure of multiphase wind.
        """
        v_wind = state[0]
        rho_wind = state[1]
        Pressure = state[2]
        rhoZ_wind = state[3]
        M_cloud = state[4]
        v_cloud = state[5]
        Z_cloud = state[6]

        # wind properties
        cs_sq_wind = constants.GAMMA * Pressure / rho_wind
        Mach_sq_wind = v_wind**2 / cs_sq_wind
        Z_wind = rhoZ_wind / rho_wind
        vc = v_circ0 * np.where(r < r_sonic, r / r_sonic, 1.0)

        # source term from inside galaxy
        Edot_SN = Edot_per_Vol * np.where(Mach_sq_wind < 1, 1.0, 0.0)
        Mdot_SN = Mdot_per_Vol * np.where(Mach_sq_wind < 1, 1.0, 0.0)

        # cloud properties
        Ndot_cloud = Ndot_cloud_init * np.where(
            r < cold_cloud_injection_radial_extent,
            (r / cold_cloud_injection_radial_extent)
            ** cold_cloud_injection_radial_power,
            1.0,
        )
        number_density_cloud = Ndot_cloud / (Omwind * v_cloud * r**2)
        cs_cl_sq = (
            constants.GAMMA * constants.K_B * T_cloud / (constants.MU * constants.M_P)
        )

        # cloud transfer rates
        rho_cloud = (
            Pressure * (constants.MU * constants.M_P) / (constants.K_B * T_cloud)
        )  # cloud in pressure equilibrium
        chi = rho_cloud / rho_wind  # density contrast
        r_cloud = (M_cloud / (4 * np.pi / 3.0 * rho_cloud)) ** (1 / 3.0)
        v_rel = v_wind - v_cloud
        v_turb = f_turb0 * v_rel * chi**TurbulentVelocityChiPower
        T_wind = Pressure / constants.K_B * (constants.MU * constants.M_P / rho_wind)
        T_mix = (T_wind * T_cloud) ** 0.5
        Z_mix = (Z_wind * Z_cloud) ** 0.5
        t_cool_layer = cooling.tcool_P(
            T_mix, Pressure / constants.K_B, Z_mix / constants.Z_SOLAR
        )[()]
        t_cool_layer = np.where(t_cool_layer < 0, 1e10 * constants.MYR, t_cool_layer)
        ksi = r_cloud / (v_turb * t_cool_layer)
        AreaBoost = chi**CoolingAreaChiPower
        v_turb_cold = v_turb * chi**ColdTurbulenceChiPower
        Mdot_grow = (
            Mdot_coefficient
            * 3.0
            * M_cloud
            * v_turb
            * AreaBoost
            / (r_cloud * chi)
            * np.where(ksi < 1, ksi**0.5, ksi**0.25)
        )
        Mdot_loss = Mdot_coefficient * 3.0 * M_cloud * v_turb_cold / r_cloud
        Mdot_cloud = np.where(M_cloud > M_cloud_min, Mdot_grow - Mdot_loss, 0)

        # density
        drhodt = number_density_cloud * Mdot_cloud
        drhodt_plus = number_density_cloud * Mdot_loss
        drhodt_minus = number_density_cloud * Mdot_grow

        # momentum
        p_dot_drag = (
            0.5
            * drag_coeff
            * rho_wind
            * np.pi
            * v_rel**2
            * r_cloud**2
            * np.where(M_cloud > M_cloud_min, 1, 0)
        )
        dpdt_drag = number_density_cloud * p_dot_drag

        # energy
        e_dot_cool = (
            0.0
            if (Cooling_Factor == 0)
            else (rho_wind / (constants.MU_H * constants.M_P)) ** 2
            * cooling.Lambda_P_rho(Pressure, rho_wind, Z_wind / constants.Z_SOLAR)
        )

        # metallicity
        drhoZdt = -1.0 * (
            number_density_cloud * (Z_wind * Mdot_grow + Z_cloud * Mdot_loss)
        )

        # wind gradients
        # velocity
        dv_dr = 2 / Mach_sq_wind
        dv_dr += -((vc / v_wind) ** 2)
        dv_dr += drhodt_minus / (rho_wind * v_wind / r) * (1 / Mach_sq_wind)
        dv_dr += -drhodt_plus / (rho_wind * v_wind / r) * (1 / Mach_sq_wind)
        dv_dr += -drhodt_plus / (rho_wind * v_wind / r) * v_rel / v_wind
        dv_dr += (
            -drhodt_plus
            / (rho_wind * v_wind / r)
            * (constants.GAMMA - 1)
            / 2.0
            * (v_rel / v_wind) ** 2
        )
        dv_dr += (
            -drhodt_plus
            / (rho_wind * v_wind / r)
            * (-(cs_sq_wind - cs_cl_sq) / v_wind**2)
        )
        dv_dr += (constants.GAMMA - 1) * e_dot_cool / (rho_wind * v_wind**3 / r)
        dv_dr += -(constants.GAMMA - 1) * dpdt_drag * v_rel / (rho_wind * v_wind**3 / r)
        dv_dr += -dpdt_drag / (rho_wind * v_wind**2 / r)
        dv_dr *= (v_wind / r) / (1.0 - (1.0 / Mach_sq_wind))

        # density
        drho_dr = -2
        drho_dr += (vc / v_wind) ** 2
        drho_dr += -drhodt_minus / (rho_wind * v_wind / r)
        drho_dr += drhodt_plus / (rho_wind * v_wind / r)
        drho_dr += drhodt_plus / (rho_wind * v_wind / r) * v_rel / v_wind
        drho_dr += (
            drhodt_plus
            / (rho_wind * v_wind / r)
            * (constants.GAMMA - 1)
            / 2.0
            * (v_rel / v_wind) ** 2
        )
        drho_dr += (
            drhodt_plus
            / (rho_wind * v_wind / r)
            * (-(cs_sq_wind - cs_cl_sq) / v_wind**2)
        )
        drho_dr += -(constants.GAMMA - 1) * e_dot_cool / (rho_wind * v_wind**3 / r)
        drho_dr += (
            (constants.GAMMA - 1) * dpdt_drag * v_rel / (rho_wind * v_wind**3 / r)
        )
        drho_dr += dpdt_drag / (rho_wind * v_wind**2 / r)
        drho_dr *= (rho_wind / r) / (1.0 - (1.0 / Mach_sq_wind))

        # pressure
        dP_dr = -2
        dP_dr += (vc / v_wind) ** 2
        dP_dr += -drhodt_minus / (rho_wind * v_wind / r)
        dP_dr += drhodt_plus / (rho_wind * v_wind / r)
        dP_dr += drhodt_plus / (rho_wind * v_wind / r) * v_rel / v_wind
        dP_dr += (
            drhodt_plus
            / (rho_wind * v_wind / r)
            * (constants.GAMMA - 1)
            / 2.0
            * (v_rel**2 / cs_sq_wind)
        )
        dP_dr += (
            drhodt_plus
            / (rho_wind * v_wind / r)
            * (-(cs_sq_wind - cs_cl_sq) / cs_sq_wind)
        )
        dP_dr += (
            -(constants.GAMMA - 1) * e_dot_cool / (rho_wind * v_wind * cs_sq_wind / r)
        )
        dP_dr += (
            (constants.GAMMA - 1)
            * dpdt_drag
            * v_rel
            / (rho_wind * v_wind * cs_sq_wind / r)
        )
        dP_dr += dpdt_drag / (rho_wind * v_wind**2 / r)
        dP_dr *= (Pressure / r) * constants.GAMMA / (1.0 - (1.0 / Mach_sq_wind))

        drhoZ_dr = drho_dr * (rhoZ_wind / rho_wind) + (rhoZ_wind / r) * drhodt_plus / (
            rho_wind * v_wind / r
        ) * (Z_cloud / Z_wind - 1)

        # cloud gradients
        dM_cloud_dr = Mdot_cloud / v_cloud

        dv_cloud_dr = (
            (p_dot_drag + v_rel * Mdot_grow - M_cloud * vc**2 / r)
            / (M_cloud * v_cloud)
            * np.where(M_cloud > M_cloud_min, 1, 0)
        )

        dZ_cloud_dr = (
            (Z_wind - Z_cloud)
            * Mdot_grow
            / (M_cloud * v_cloud)
            * np.where(M_cloud > M_cloud_min, 1, 0)
        )

        return np.r_[
            dv_dr, drho_dr, dP_dr, drhoZ_dr, dM_cloud_dr, dv_cloud_dr, dZ_cloud_dr
        ]

    def Single_Phase_Wind_Evo(r, state):
        """
        Calculates the derivative of v_wind, rho_wind, Pressure for a single phase wind.
        Used with solve_ivp to calculate steady state structure of a single phase wind with no cooling and no gravity.
        """
        v_wind = state[0]
        rho_wind = state[1]
        Pressure = state[2]

        # wind properties
        cs_sq_wind = constants.GAMMA * Pressure / rho_wind
        Mach_sq_wind = v_wind**2 / cs_sq_wind

        # source term from inside galaxy
        Edot_SN = Edot_per_Vol * np.where(r < r_sonic, 1.0, 0.0)
        Mdot_SN = Mdot_per_Vol * np.where(r < r_sonic, 1.0, 0.0)

        # density
        drhodt = Mdot_SN

        # momentum
        dpdt = 0

        # energy
        dedt = Edot_SN

        dv_dr = (
            (v_wind / r)
            / (1.0 - (1.0 / Mach_sq_wind))
            * (
                2.0 / Mach_sq_wind
                - 1
                / (rho_wind * v_wind / r)
                * (
                    drhodt * (constants.GAMMA + 1) / 2.0
                    + (constants.GAMMA - 1) * dedt / v_wind**2
                )
            )
        )
        drho_dr = (
            (rho_wind / r)
            / (1.0 - (1.0 / Mach_sq_wind))
            * (
                -2.0
                + 1
                / (rho_wind * v_wind / r)
                * (
                    drhodt * (constants.GAMMA + 3) / 2.0
                    + (constants.GAMMA - 1) * dedt / v_wind**2
                    - drhodt / Mach_sq_wind
                )
            )
        )
        dP_dr = (
            (Pressure / r)
            * constants.GAMMA
            / (1.0 - (1.0 / Mach_sq_wind))
            * (
                -2.0
                + 1
                / (rho_wind * v_wind / r)
                * (
                    drhodt
                    + drhodt * (constants.GAMMA - 1) / 2.0 * Mach_sq_wind
                    + (constants.GAMMA - 1) * Mach_sq_wind * dedt / v_wind**2
                )
            )
        )

        return np.r_[dv_dr, drho_dr, dP_dr]

    def cloud_ksi(r, state):
        """
        function to calculate the value of ksi = t_mix / t_cool
        """
        v_wind = state[0]
        rho_wind = state[1]
        Pressure = state[2]
        rhoZ_wind = state[3]
        Z_wind = rhoZ_wind / rho_wind
        M_cloud = state[4]
        v_cloud = state[5]
        Z_cloud = state[6]
        rho_cloud = (
            Pressure * (constants.MU * constants.M_P) / (constants.K_B * T_cloud)
        )  # cloud in pressure equilibrium
        chi = rho_cloud / rho_wind
        r_cloud = (M_cloud / (4 * np.pi / 3.0 * rho_cloud)) ** (1 / 3.0)
        v_rel = v_wind - v_cloud
        v_turb = f_turb0 * v_rel * chi**TurbulentVelocityChiPower
        T_wind = Pressure / constants.K_B * (constants.MU * constants.M_P / rho_wind)
        T_mix = (T_wind * T_cloud) ** 0.5
        Z_mix = (Z_wind * Z_cloud) ** 0.5
        t_cool_layer = cooling.tcool_P(
            T_mix, Pressure / constants.K_B, Z_mix / constants.Z_SOLAR
        )[()]
        t_cool_layer = np.where(t_cool_layer < 0, 1e10 * constants.MYR, t_cool_layer)
        ksi = r_cloud / (v_turb * t_cool_layer)
        return ksi, r_cloud, v_turb

    def supersonic(r, z):
        return z[0] / np.sqrt(constants.GAMMA * z[2] / z[1]) - (1.0 + epsilon)

    supersonic.terminal = True

    def subsonic(r, z):
        return z[0] / np.sqrt(constants.GAMMA * z[2] / z[1]) - (1.0 - epsilon)

    subsonic.terminal = True

    def cold_wind(r, z):
        return np.sqrt(constants.GAMMA * z[2] / z[1]) / np.sqrt(
            constants.GAMMA * constants.K_B * T_cloud / (constants.MU * constants.M_P)
        ) - (1.0 + epsilon)

    cold_wind.terminal = True

    def cloud_stop(r, z):
        return z[5] - 10e5

    cloud_stop.terminal = True

    ##########################
    # integrate single phase #
    ##########################

    # singlephase params
    r_init = (
        singlephase_params["r_init"] * constants.PC
    )  ### inner radius for hot solution

    ## calculate gradients right at sonic radius
    dv_dr0, drho_dr0, dP_dr0 = Single_Phase_Wind_Evo(r_sonic, np.r_[v0, rho0, P0])

    ## interpolate to within the subsonic region
    dlogvdlogr = dv_dr0 * r_sonic / v0
    dlogrhodlogr = drho_dr0 * r_sonic / rho0
    dlogPdlogr = dP_dr0 * r_sonic / P0
    dlogr0 = 1e-8

    v0_sub = 10 ** (np.log10(v0) - dlogvdlogr * dlogr0)
    rho0_sub = 10 ** (np.log10(rho0) - dlogrhodlogr * dlogr0)
    P0_sub = 10 ** (np.log10(P0) - dlogPdlogr * dlogr0)

    ### integrate (single phase only) from sonic radius to r_init in the subsonic region.
    sol = solve_ivp(
        Single_Phase_Wind_Evo,
        [10 ** (np.log10(r_sonic) - dlogr0), r_init],
        np.r_[v0_sub, rho0_sub, P0_sub],
        events=[supersonic],
        dense_output=True,
        rtol=1e-12,
        atol=[1e-3, 1e-7 * constants.M_P, 1e-2 * constants.K_B],
    )

    r_init = sol.t[-1]
    v_init = sol.y[0][-1]
    rho_init = sol.y[1][-1]
    P_init = sol.y[2][-1]
    rhoZ_init = rho_init * Z_wind_init

    ## interpolate to within the supersonic region
    v0_sup = 10 ** (np.log10(v0) + dlogvdlogr * dlogr0)
    rho0_sup = 10 ** (np.log10(rho0) + dlogrhodlogr * dlogr0)
    P0_sup = 10 ** (np.log10(P0) + dlogPdlogr * dlogr0)

    ### integrate (single phase only) from sonic radius to 100x sonic radius
    sol_sup = solve_ivp(
        Single_Phase_Wind_Evo,
        [10 ** (np.log10(r_sonic) + dlogr0), 10**2 * r_sonic],
        np.r_[v0_sup, rho0_sup, P0_sup],
        events=[supersonic],
        dense_output=True,
        rtol=1e-12,
        atol=[1e-3, 1e-7 * constants.M_P, 1e-2 * constants.K_B],
    )

    r_hot_only = np.append(sol.t[::-1], sol_sup.t)
    v_wind_hot_only = np.append(sol.y[0][::-1], sol_sup.y[0])
    rho_wind_hot_only = np.append(sol.y[1][::-1], sol_sup.y[1])
    P_wind_hot_only = np.append(sol.y[2][::-1], sol_sup.y[2])

    Mdot_wind_hot_only = (
        Omwind
        * r_hot_only**2
        * rho_wind_hot_only
        * v_wind_hot_only
        / (constants.M_SUN / constants.YR)
    )
    cs_wind_hot_only = np.sqrt(constants.GAMMA * P_wind_hot_only / rho_wind_hot_only)
    T_wind_hot_only = (
        P_wind_hot_only
        / constants.K_B
        / (rho_wind_hot_only / (constants.MU * constants.M_P))
    )
    K_wind_hot_only = (P_wind_hot_only / constants.K_B) / (
        rho_wind_hot_only / (constants.MU * constants.M_P)
    ) ** constants.GAMMA
    Pdot_wind_hot_only = (
        Omwind
        * r_hot_only**2
        * rho_wind_hot_only
        * v_wind_hot_only**2
        / (1e5 * constants.M_SUN / constants.YR)
    )
    Edot_wind_hot_only = (
        Omwind
        * r_hot_only**2
        * rho_wind_hot_only
        * v_wind_hot_only
        * (0.5 * v_wind_hot_only**2 + 1.5 * cs_wind_hot_only**2)
        / (1e5**2 * constants.M_SUN / constants.YR)
    )

    #########################
    # integrate multiphase #
    #########################

    # multiphase params
    T_cloud = multiphase_params["T_cloud"]
    log_eta_M_cold = np.log10(eta_M_cold)
    cold_cloud_injection_radial_power = np.inf
    cold_cloud_injection_radial_extent = (
        multiphase_params["cold_cloud_injection_radial_extent"] * r_sonic
    )
    cloud_radial_offset = multiphase_params[
        "cloud_radial_offset"
    ]  ### don't start integration exactly at r_sonic
    irstart = np.argmin(np.abs(r_hot_only - r_sonic * (1.0 + cloud_radial_offset)))
    r_init = r_hot_only[irstart]
    v_init = v_wind_hot_only[irstart]
    rho_init = rho_wind_hot_only[irstart]
    P_init = P_wind_hot_only[irstart]
    M_cloud_init = 10**log_M_cloud_init * constants.M_SUN
    Z_cloud_init = multiphase_params["Z_cloud_init"] * constants.Z_SOLAR
    v_cloud_init = multiphase_params["v_cloud_init"] * constants.KM / constants.SEC
    Mdot_cold_init = eta_M_cold * SFR  ## mass flux in cold clouds
    Ndot_cloud_init = Mdot_cold_init / M_cloud_init  ## number flux in cold clouds

    #### ICs
    supersonic_initial_conditions = np.r_[
        v_init,
        rho_init,
        P_init,
        Z_wind_init * rho_init,
        M_cloud_init,
        v_cloud_init,
        Z_cloud_init,
    ]

    ### integrate!
    sol = solve_ivp(
        Multiphase_Wind_Evo,
        [r_init, 1e2 * r_sonic],
        supersonic_initial_conditions,
        events=[supersonic, cloud_stop, cold_wind],
        dense_output=True,
        rtol=1e-10,
    )
    # print(sol.message)
    # print(sol.t_events)

    ## gather solution and manipulate into useful form
    r = sol.t
    v_wind = sol.y[0]
    rho_wind = sol.y[1]
    P_wind = sol.y[2]
    rhoZ_wind = sol.y[3]
    M_cloud = sol.y[4]
    v_cloud = sol.y[5]
    Z_cloud = sol.y[6]
    ksi, r_cloud, v_turb = cloud_ksi(r, sol.y)

    cloud_Mdots = (
        np.outer(
            Ndot_cloud_init,
            np.where(
                r < cold_cloud_injection_radial_extent,
                (r / cold_cloud_injection_radial_extent)
                ** cold_cloud_injection_radial_power,
                1.0,
            ),
        )
        * M_cloud
        / (constants.M_SUN / constants.YR)
    )
    Mdot_wind = Omwind * r**2 * rho_wind * v_wind / (constants.M_SUN / constants.YR)
    cs_wind = np.sqrt(constants.GAMMA * P_wind / rho_wind)
    T_wind = P_wind / constants.K_B / (rho_wind / (constants.MU * constants.M_P))
    K_wind = (P_wind / constants.K_B) / (
        rho_wind / (constants.MU * constants.M_P)
    ) ** constants.GAMMA

    Pdot_wind = (
        Omwind * r**2 * rho_wind * v_wind**2 / (1e5 * constants.M_SUN / constants.YR)
    )
    cloud_Pdots = (
        np.outer(
            Ndot_cloud_init,
            np.where(
                r < cold_cloud_injection_radial_extent,
                (r / cold_cloud_injection_radial_extent)
                ** cold_cloud_injection_radial_power,
                1.0,
            ),
        )
        * M_cloud
        * v_cloud
        / (1e5 * constants.M_SUN / constants.YR)
    )

    Edot_wind = (
        Omwind
        * r**2
        * rho_wind
        * v_wind
        * (0.5 * v_wind**2 + 1.5 * cs_wind**2)
        / (1e5**2 * constants.M_SUN / constants.YR)
    )
    cloud_Edots = (
        np.outer(
            Ndot_cloud_init,
            np.where(
                r < cold_cloud_injection_radial_extent,
                (r / cold_cloud_injection_radial_extent)
                ** cold_cloud_injection_radial_power,
                1.0,
            ),
        )
        * M_cloud
        * (
            0.5 * v_cloud**2
            + 2.5 * constants.K_B * T_cloud / (constants.MU * constants.M_P)
        )
        / (1e5**2 * constants.M_SUN / constants.YR)
    )

    sol_dict = {
        "r_hot_only": r_hot_only,
        "v_wind_hot_only": v_wind_hot_only,
        "rho_wind_hot_only": rho_wind_hot_only,
        "P_wind_hot_only": P_wind_hot_only,
        "Mdot_wind_hot_only": Mdot_wind_hot_only,
        "cs_wind_hot_only": cs_wind_hot_only,
        "T_wind_hot_only": T_wind_hot_only,
        "K_wind_hot_only": K_wind_hot_only,
        "Pdot_wind_hot_only": Pdot_wind_hot_only,
        "Edot_wind_hot_only": Edot_wind_hot_only,
        "r": r,
        "v_wind": v_wind,
        "rho_wind": rho_wind,
        "P_wind": P_wind,
        "rhoZ_wind": rhoZ_wind,
        "M_cloud": M_cloud,
        "T_cloud": np.ones(T_wind.shape) * T_cloud,
        "v_cloud": v_cloud,
        "Z_cloud": Z_cloud,
        "cloud_Mdots": cloud_Mdots,
        "Mdot_wind": Mdot_wind,
        "cs_wind": cs_wind,
        "T_wind": T_wind,
        "K_wind": K_wind,
        "Pdot_wind": Pdot_wind,
        "cloud_Pdots": cloud_Pdots,
        "Edot_wind": Edot_wind,
        "cloud_Edots": cloud_Edots,
        "ksi": ksi,
        "r_cloud": r_cloud,
        "v_turb": v_turb,
        "r_sonic": r_sonic,
        "r_cloud_start": cold_cloud_injection_radial_extent,
    }

    return utils.Wind(sol_dict)
