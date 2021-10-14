import os
import sys
import numpy as np
import vulcan_cfg_template
import scipy
from astropy import constants as c
from astropy import units as u
import multiprocessing as mp
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
stellar_dir = '../stellar_spectra/'
sys.path.append(os.path.join(script_dir, stellar_dir))

from CreateSpecGrid import create_specs


def analytic_MR(M):
    """
    M in Mjup, R in R_jup
    """
    M_earth = u.Mjup.to(u.Mearth, M)

    if M_earth.value < 120:
        # Eq. 1 https://www.aanda.org/articles/aa/pdf/2020/02/aa36482-19.pdf
        R_earth = 0.70 * M_earth ** 0.63
    else:
        # Table 1 https://arxiv.org/pdf/1603.08614.pdf
        R_earth = 17.78 * M_earth ** (-0.044)

    return u.Rearth.to(u.Rjup, R_earth)


def irradiation_temperature(T_star, R_star, a):
    """
    Calculate the irradiation temperature of a planet with Bond Albedo = 0.

    Args:
        T_star: (float) effective stellar temperature [K]
        R_star: (float) stellar radius
        a: (float) semi-major axis

    Returns:
        T_irr: (float) irradiation temperature [K]
    """

    return T_star * np.sqrt(R_star / a)


# T(P) profile in Heng et al. 2014 (126)
# modified from VULCAN/build_atm.py line 449
def TP_H14(pco, g, *args_analytical):
    # convert args_analytical tuple to a list so we can modify it
    T_int, T_irr, ka_0, ka_s, beta_s, beta_l = list(args_analytical)

    P_b = vulcan_cfg_template.P_b

    # albedo(beta_s) also affects T_irr
    albedo = (1.0 - beta_s) / (1.0 + beta_s)
    T_irr *= (1 - albedo) ** 0.25
    eps_L = 3. / 8
    eps_L3 = 1. / 3
    ka_CIA = 0
    m = pco / g
    m_0 = P_b / g
    ka_l = ka_0 + ka_CIA * m / m_0
    term1 = T_int ** 4 / 4 * (1 / eps_L + m / (eps_L3 * beta_l ** 2) * (ka_0 + ka_CIA * m / (2 * m_0)))
    term2 = (1 / (2 * eps_L) + scipy.special.expn(2, ka_s * m / beta_s) * (
            ka_s / (ka_l * beta_s) - (ka_CIA) * m * beta_s / (eps_L3 * ka_s * m_0 * beta_l ** 2)))
    term3 = ka_0 * beta_s / (eps_L3 * ka_s * beta_l ** 2) * (1. / 3 - scipy.special.expn(4, ka_s * m / beta_s))
    term4 = 0.  # related to CIA
    T = (term1 + T_irr ** 4 / 8 * (term2 + term3 + term4)) ** 0.25

    return T


def calculate_TP(gs, para_anaTP):
    Pco = np.logspace(np.log10(vulcan_cfg_template.P_b), np.log10(vulcan_cfg_template.P_t),
                      vulcan_cfg_template.nz)  # pressure grids
    Tco = TP_H14(Pco, gs, *para_anaTP)

    return (Tco, Pco)


def surface_gravity(M, R):
    """
    M in g, R in cm.
    """
    return c.G.cgs * M / R ** 2


# parameters from fitted data in plot_hot_jupiters.py
# TODO: change to fit when initializing?
def effective_temperature(R_s):
    """
    R_S in Rsun.
    """
    return 5.58049869e+03 * R_s ** 3.48859155e-01


def make_valid_parameters(params):
    # calcualte effective temperature
    R_star = params['r_star']  # Rsun
    T_eff = effective_temperature(R_star)  # K
    T_eff = round(T_eff.value, 2)

    # calculate planet radius
    M_p = params['planet_mass']  # Mjup
    R_p = analytic_MR(M_p)  # Rjup

    R_p = R_p.cgs
    M_p = M_p.cgs

    # calculate surface gravity
    gs = surface_gravity(M_p, R_p)

    # calculate TP
    a = params["orbit_radius"]  # AU
    T_irr = irradiation_temperature(T_eff, R_star.to(u.AU), a)

    TP_params = [120., T_irr.value, 0.1, 0.02, 1., 1.]

    (Tco, Pco) = calculate_TP(gs.value, TP_params)

    # check if valid
    if np.max(Tco) > 3000 or np.min(Tco) < 500:
        return None

    # create spectra
    sflux_file = create_specs([T_eff], save_to_txt=True)  # in list because that's what Amy's function uses

    # append to valid parameters
    valid_params = dict(
        T_eff=T_eff,
        T_irr=T_irr.value,
        sflux_file=str(sflux_file),
        r_star=R_star.value,
        Rp=R_p.value,
        orbit_radius=a.value,
        gs=gs.value
    )

    return valid_params


def make_valid_parameter_grid(parameter_grid, num_workers):
    print('making valid parameter grid...')
    with mp.Pool(num_workers) as p:
        results = list(tqdm(p.imap(make_valid_parameters, parameter_grid),  # return results otherwise it doesn't work properly
                            total=len(parameter_grid)))

    print(np.shape(results))
    valid_parameter_grid = np.array(results).flatten()
    # remove None values
    valid_parameter_grid = valid_parameter_grid[valid_parameter_grid != np.array(None)]
    print(np.shape(valid_parameter_grid))
    return valid_parameter_grid