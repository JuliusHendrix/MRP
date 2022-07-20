import numpy as np
import glob
import os
import shutil
from sklearn.model_selection import ParameterGrid
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from astropy import units as u

from vulcan_config_utils import make_valid_parameter_grid


def make_config(mp_params):
    """
    Make and save a config file for a given set of parameters. Copies and appends to the "vulcan_cfg_template.py" file
    which should be in the same directory as this script.

    Args:
        mp_params: (tuple) (params, configs_dir, output_dir)
data
    Returns:

    """
    (params, configs_dir, output_dir, script_dir) = mp_params

    # extract parameters
    orbit_radius = params['orbit_radius']
    r_star = params['r_star']
    sflux_file = params['sflux_file']
    T_eff = params["T_eff"]
    T_irr = params["T_irr"]
    Rp = params["Rp"]
    gs = params["gs"]
    planet_mass = params["planet_mass"]

    # give unique name
    config_name = f'{orbit_radius}_{r_star}_{planet_mass}'
    config_filename = f'{configs_dir}/vulcan_cfg_{config_name}.py'
    output_name = f'output_{config_name}.vul'

    # copy template file
    shutil.copyfile(os.path.join(script_dir, 'vulcan_cfg_template.py'), config_filename)

    # append to template file
    with open(config_filename, 'a') as file:
        text_to_append = f"output_dir = '{output_dir}'\n" \
                         "plot_dir = 'plot/'\n" \
                         "movie_dir = 'plot/movie/'\n" \
                         f"out_name = '{output_name}'\n" \
                         f"para_warm = [120., {T_irr}, 0.1, 0.02, 1., 1.]\n" \
                         "para_anaTP = para_warm\n" \
                         f"sflux_file = '{sflux_file}'\n" \
                         f"r_star = {r_star}\n" \
                         f"Rp = {Rp}\n" \
                         f"orbit_radius = {orbit_radius}\n" \
                         f"gs = {gs}\n" \
                         f"planet_mass = {planet_mass}"

        file.write(text_to_append)
    return 0


def main():
    # setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MRP_dir = str(Path(script_dir).parents[1])
    configs_dir = os.path.join(MRP_dir, 'data/configs')
    output_dir_vulcan = '../../MRP/data/vulcan_output/'    # vulcan needs a relative dir...
    sflux_dir = os.path.join(MRP_dir, 'src/stellar_spectra/output')
    num_workers = mp.cpu_count() - 1

    # remake the config directory
    if os.path.isdir(configs_dir):
        shutil.rmtree(configs_dir)
    os.mkdir(configs_dir)

    # remake the sflux directory
    if os.path.isdir(sflux_dir):
        shutil.rmtree(sflux_dir)
    os.mkdir(sflux_dir)

    # setup parameter ranges and intervals
    # parameter_ranges = dict(
    #     orbit_radius=np.linspace(0.01, 0.5, 20) * u.AU,    # AU, circular orbit
    #     planet_mass=np.linspace(0.5, 5, 20) * u.Mjup,    # Mjup
    #     r_star=np.linspace(1, 1.5, 20) * u.Rsun,   # Rsun   # values same as fit
    # )

    parameter_ranges = dict(
        orbit_radius=np.linspace(0.01, 0.5, 20) * u.AU,    # AU, circular orbit
        planet_mass=np.linspace(0.5, 5, 20) * u.Mjup,    # Mjup
        r_star=np.linspace(1, 1.5, 20) * u.Rsun,   # Rsun   # values same as fit
    )

    # create parameter grid of valid configurations
    parameter_grid = ParameterGrid(parameter_ranges)
    valid_parameter_grid = make_valid_parameter_grid(parameter_grid, num_workers, sflux_dir)

    # make the mp parameters
    mp_params = [(params, configs_dir, output_dir_vulcan, script_dir) for params in valid_parameter_grid]

    # run mp Pool
    print('generating vulcan_cfg files...')
    with mp.Pool(num_workers) as p:
        results = list(tqdm(p.imap(make_config, mp_params),    # return results otherwise it doesn't work properly
                            total=len(mp_params)))


if __name__ == "__main__":
    main()

