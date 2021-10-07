import numpy as np
import glob
import os
import shutil
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import multiprocessing as mp

# properties to be appended
# -------------------------------
# output_dir = 'output/'
# plot_dir = 'plot/'
# movie_dir = 'plot/movie/'
# out_name = 'HD189.vul' # output file name
#
# # setting the parameters for the analytical T-P from (126)in Heng et al. 2014. Only reads when atm_type = 'analytical'
# # T_int, T_irr, ka_L, ka_S, beta_S, beta_L
# para_warm = [120., 1500., 0.1, 0.02, 1., 1.]
# para_anaTP = para_warm
#
# sflux_file = 'atm/stellar_flux/sflux-HD189_Moses11.txt'
# sflux-HD189_B2020.txt This is the flux density at the stellar surface
#
# r_star = 0.805 # stellar radius in solar radius
# Rp = 1.138*7.1492E9 # Planetary radius (cm) (for computing gravity)
# orbit_radius = 0.03142 # planet-star distance in A.U.

solar_radius = 6.957e11    # cm


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


def make_config(mp_params):
    """
    Make and save a config file for a given set of parameters. Copies and appends to the "vulcan_cfg_template.py" file
    which should be in the same directory as this script.

    Args:
        mp_params: (tuple) (params, configs_dir, output_dir)

    Returns:

    """
    (params, configs_dir, output_dir) = mp_params

    # extract parameters
    orbit_radius = params['orbit_radius']
    r_star = params['r_star']
    sflux_file = params['sflux_file']
    T_star = float(os.path.basename(sflux_file)[:-6])  # remove '_K.txt' and convert to float
    T_irr = irradiation_temperature(T_star, r_star * solar_radius, orbit_radius)

    # copy template file
    config_filename = f'{configs_dir}/vulcan_cfg_{orbit_radius}_{r_star}_{T_star}.py'

    shutil.copyfile('vulcan_cfg_template.py', config_filename)

    # append to template file
    with open(config_filename, 'a') as file:
        output_name = f'output_{orbit_radius}_{r_star}_{T_star}.vul'

        text_to_append = f"output_dir = '{output_dir}'\n" \
                         "plot_dir = 'plot/'\n" \
                         "movie_dir = 'plot/movie/'\n" \
                         f"out_name = '{output_name}' # output file name\n" \
                         f"para_warm = [120., {T_irr}, 0.1, 0.02, 1., 1.]\n" \
                         "para_anaTP = para_warm\n" \
                         f"sflux_file = '{sflux_file}'\n" \
                         f"r_star = {r_star}\n" \
                         "Rp = 1.138*7.1492E9\n" \
                         f"orbit_radius = {orbit_radius}"

        file.write(text_to_append)
    return 1


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(script_dir, 'configs')
    output_dir = os.path.join(script_dir, 'vulcan_output')

    stellar_spectra = glob.glob('../stellar_spectra/output/txtfiles/*.txt')
    stellar_spectra = [os.path.abspath(path) for path in stellar_spectra]

    # TODO: define ranges
    # parameters to be sampled:
    parameter_ranges = dict(
        orbit_radius=np.linspace(0.01, 10, 5),  # AU, circular orbit
        r_star=np.linspace(0.5, 1.5, 5),  # R_sun
        sflux_file=stellar_spectra
    )

    parameter_grid = ParameterGrid(parameter_ranges)

    if os.path.isdir(configs_dir):
        shutil.rmtree(configs_dir)
    os.mkdir(configs_dir)

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    num_workers = mp.cpu_count() - 1
    mp_params = [(params, configs_dir, output_dir) for params in parameter_grid]

    print('generating vulcan_cfg files...')
    with mp.Pool(num_workers) as p:
        results = list(tqdm(p.imap(make_config, mp_params),    # return results otherwise it doesn't work properly
                            total=len(mp_params)))


if __name__ == "__main__":
    main()
