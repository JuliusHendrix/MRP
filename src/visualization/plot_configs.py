import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp
from pathlib import Path
from astropy import units as u
from sklearn.model_selection import ParameterGrid
import importlib.util

# own modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = str(Path(script_dir).parents[1])
sys.path.append(src_dir)

from src.vulcan_configs.vulcan_config_utils import make_valid_parameter_grid, calculate_TP

colors = [
'#065e60', '#0a9396', '#99d5c9', '#ee9b00', '#ffb833', '#99d6ff', '#0093f5', '#00568f'
]

def plot_config_ranges(MRP_dir):
    sflux_dir = os.path.join(MRP_dir, 'src/stellar_spectra/output')
    num_workers = mp.cpu_count() - 1

    steps = 10

    # setup parameter ranges and intervals
    parameter_ranges = dict(
        orbit_radius=np.linspace(0.01, 0.5, steps) * u.AU,  # AU, circular orbit
        planet_mass=np.linspace(0.5, 5, steps) * u.Mjup,  # Mjup
        r_star=np.linspace(1, 1.5, steps) * u.Rsun,  # Rsun   # values same as fit
    )

    # create parameter grid of valid configurations
    parameter_grid = ParameterGrid(parameter_ranges)
    valid_parameter_grid = make_valid_parameter_grid(parameter_grid, num_workers, sflux_dir)

    print(f'{len(valid_parameter_grid)} valid parameters.')

    orbit_radii = np.linspace(0.01, 0.5, steps)  # AU, circular orbit
    planet_masses = np.linspace(0.5, 5, steps)  # Mjup
    radii_star = np.linspace(1, 1.5, steps)  # Rsun   # values same as fit

    ma = np.zeros(shape=(
        len(orbit_radii),
        len(radii_star),
        len(planet_masses)
    ))

    for params in valid_parameter_grid:
        orbit_idx = np.where(orbit_radii == params['orbit_radius'])[0][0]
        mass_idx = np.where(planet_masses == params['planet_mass'])[0][0]
        r_star_idx = np.where(radii_star == params['r_star'])[0][0]

        ma[orbit_idx, r_star_idx, mass_idx] = 1

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.set_aspect('equal')

    corners_x = np.zeros(len(orbit_radii) + 1)
    corners_x[0] = orbit_radii[0] - 0.5 * np.diff(orbit_radii)[0]
    corners_x[1:] = orbit_radii + 0.5 * np.diff(orbit_radii)[0]

    corners_y = np.zeros(len(radii_star) + 1)
    corners_y[1:] = radii_star + 0.5 * np.diff(radii_star)[0]
    corners_y[0] = radii_star[0] - 0.5 * np.diff(radii_star)[0]

    corners_z = np.zeros(len(planet_masses) + 1)
    corners_z[1:] = planet_masses + 0.5 * np.diff(planet_masses)[0]
    corners_z[0] = planet_masses[-1] - 0.5 * np.diff(planet_masses)[0]

    x, y, z = np.indices(np.array(ma.shape) + 1)

    x = x / ma.shape[0]  # scale from 0-1
    y = y / ma.shape[0]
    z = z / ma.shape[0]

    x = x * (max(corners_x) - min(corners_x)) + min(corners_x)  # scale to range
    y = y * (max(corners_y) - min(corners_y)) + min(corners_y)
    z = z * (max(corners_z) - min(corners_z)) + min(corners_z)

    ax.voxels(x, y, z, ma, facecolors='lightskyblue', edgecolor="k")

    ax.set_xlabel("a [AU]")
    # ax.set_xticks(orbit_radii)

    ax.set_zlabel("$M_p$ [$M_{jup}$]")
    # ax.set_zticks(planet_masses)

    ax.set_ylabel("$R_*$ [$R_{\odot}$]")
    # ax.set_yticks(radii_star)

    # ax.invert_xaxis()

    plt.savefig('configs.png', dpi=900)
    plt.show()


def plot_TP(configs_dir):
    # Set general font size
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 22

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    configs = glob.glob(os.path.join(configs_dir, '*.py'))
    random_config = np.random.choice(configs)

    spec = importlib.util.spec_from_file_location("random_config", random_config)
    random_config_m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(random_config_m)

    (Tco, Pco) = calculate_TP(random_config_m.gs, random_config_m.para_warm)

    plt.figure(figsize=(4,5))

    plt.plot(Tco, Pco*1e-6, color=colors[3])
    plt.xlabel('Temperature [K]')
    plt.ylabel('Pressure [bar]')
    plt.gca().invert_yaxis()
    # plt.xscale('log')
    plt.yscale('log')

    plt.title("Analytical TP-profile\n"
              rf"$R_* = {random_config_m.r_star:.1f}$ R$_{{Sun}}$, "
              rf"$r = {random_config_m.orbit_radius:.1f}$ AU, "
              rf"$M_P = {random_config_m.planet_mass:.1f}$ M$_{{Jup}}$"
              )

        # rf'r_star = {random_config_m.r_star:.2} $R_{Sun}$'
        #       f'orbit radius = {random_config_m.orbit_radius:.2} au\n'
        #       f'planet_mass={random_config_m.planet_mass:.2} M$_{{Jup}}$')

    plt.savefig(
        'random_images/TP_profile.pdf',
        bbox_inches='tight',)
    plt.tight_layout()
    plt.show()


def main():
    # setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MRP_dir = str(Path(script_dir).parents[1])
    configs_dir = os.path.join(MRP_dir, 'data/configs')
    output_dir_vulcan = '../../MRP/data/vulcan_output/'    # vulcan needs a relative dir...

    # plot_config_ranges(MRP_dir)
    plot_TP(configs_dir)


if __name__ == "__main__":
    main()
