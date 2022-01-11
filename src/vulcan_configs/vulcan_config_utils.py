import os
import sys
import numpy as np
import scipy
from astropy import constants as c
from astropy import units as u
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
import shutil

# own modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = str(Path(script_dir).parents[1])
sys.path.append(src_dir)

import src.vulcan_configs.vulcan_cfg_template as vulcan_cfg_template


def analytic_MR(M):
    """
    M in Mjup, R in R_jup
    """
    M_earth = u.Mjup.to(u.Mearth, M)

    if M_earth < 120:
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


def make_valid_parameters(mp_params):
    from src.stellar_spectra.CreateSpecGrid import create_specs

    (params, sflux_dir) = mp_params

    # calcualte effective temperature
    R_star = params['r_star']  # Rsun
    T_eff = effective_temperature(R_star)  # K

    # calculate planet radius
    M_p = params['planet_mass']  # Mjup
    R_p = analytic_MR(M_p.value) * u.Rjup  # Rjup

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
    if np.max(Tco) > 2500 or np.min(Tco) < 500:
        return None

    # create spectra
    sflux_file = create_specs([T_eff.value], output_dir=sflux_dir, save_to_txt=True)  # in list because that's what Amy's function uses

    # append to valid parameters
    valid_params = dict(
        T_eff=T_eff.value,
        T_irr=T_irr.value,
        sflux_file=str(sflux_file),
        r_star=R_star.value,
        Rp=R_p.value,
        planet_mass=params['planet_mass'].value,
        orbit_radius=a.value,
        gs=gs.value
    )

    return valid_params


def make_valid_parameter_grid(parameter_grid, num_workers, sflux_dir):
    print('making valid parameter grid...')

    mp_params = [(params, sflux_dir) for params in parameter_grid]

    with mp.Pool(num_workers) as p:
        results = list(tqdm(p.imap(make_valid_parameters, mp_params),  # return results otherwise it doesn't work properly
                            total=len(mp_params)))

    valid_parameter_grid = np.array(results).flatten()
    # remove None values
    valid_parameter_grid = valid_parameter_grid[valid_parameter_grid != np.array(None)]
    return valid_parameter_grid


class CopyManager:
    """
    Manage available VULCAN copies for multiprocessing.
    """

    def __init__(self, num_workers, VULCAN_dir):
        self.VULCAN_dir = VULCAN_dir

        git_dir = str(Path(VULCAN_dir).parents[0])

        self.copies_base_dir = os.path.join(git_dir, 'vulcans')

        # create VULCAN copies and save their directories
        self.available_copies = self.make_initial_copies(num_workers)

    def make_initial_copies(self, num_workers):
        """
        Make num_workers copies of the VULCAN directory.
        """
        print(f'making {num_workers} copies of VULCAN...')

        # remake the folder
        if os.path.isdir(self.copies_base_dir):
            shutil.rmtree(self.copies_base_dir)
        os.mkdir(self.copies_base_dir)

        # make list of all available dirs
        copy_dir_list = []

        # make copies
        for i in tqdm(range(num_workers)):
            copy_dir = os.path.join(self.copies_base_dir, f'VULCAN_{i}')
            shutil.copytree(self.VULCAN_dir, copy_dir)
            copy_dir_list.append(copy_dir)

        return copy_dir_list

    def get_available_copy(self):
        """
        Get a copy directory from the list of available directories
        """

        if len(self.available_copies) > 0:
            available_copy = self.available_copies[0]
            self.available_copies.remove(available_copy)
        else:
            raise ValueError('No copies available!')

        return available_copy

    def add_used_copy(self, used_copy):
        """
        Add a directory to the list of available directories
        """

        if used_copy not in self.available_copies:
            self.available_copies.append(used_copy)
        else:
            raise ValueError(f'{used_copy} already in available_copies!')