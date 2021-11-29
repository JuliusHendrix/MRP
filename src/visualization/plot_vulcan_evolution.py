import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.legend as lg

import os
import glob
import pickle
import shutil
import multiprocessing as mp
from tqdm import tqdm

import pickle

# Setting the list of species to plot
plot_spec = ['CH4', 'CO', 'H2O', 'H']
# plot_spec = ['CH4', 'H2O', 'OH', 'O2', 'O3']

# plot_spec = ('H', 'O', 'C', 'N')
colors = ['k', 'y', 'b', 'pink', 'grey']

# tex labels for plotting
tex_labels = {'H':'H','H2':'H$_2$','O':'O','OH':'OH','H2O':'H$_2$O','CH':'CH','C':'C','CH2':'CH$_2$','CH3':'CH$_3$','CH4':'CH$_4$','HCO':'HCO','H2CO':'H$_2$CO', 'C4H2':'C$_4$H$_2$',\
'C2':'C$_2$','C2H2':'C$_2$H$_2$','C2H3':'C$_2$H$_3$','C2H':'C$_2$H','CO':'CO','CO2':'CO$_2$','He':'He','O2':'O$_2$','CH3OH':'CH$_3$OH','C2H4':'C$_2$H$_4$','C2H5':'C$_2$H$_5$','C2H6':'C$_2$H$_6$','CH3O': 'CH$_3$O'\
,'CH2OH':'CH$_2$OH','N2':'N$_2$','NH3':'NH$_3$','HCN':'HCN','NO':'NO', 'NO2':'NO$_2$' }


def plot_evolution(params):
    # extract params
    (plot_dir, vulcan_file) = params

    # extract filename
    filename = os.path.basename(vulcan_file)
    plot_filename = os.path.join(plot_dir, f'{filename[:-4]}.png')

    # extract data
    with open(vulcan_file, 'rb') as handle:
        data = pickle.load(handle)
    species = data['variable']['species']

    # Setting the pressure level (cgs) to plot
    plot_p = 1e5

    # Find the index of pco closest to p_ana
    p_indx1 = min(range(len(data['atm']['pco'])), key=lambda i: abs(data['atm']['pco'][i] - plot_p))


    for color_index, sp in enumerate(plot_spec):
        if sp in tex_labels:
            sp_lab = tex_labels[sp]
        else:
            sp_lab = sp
        plt.plot(data['variable']['t_time'],
                 np.array(data['variable']['y_time'])[:, p_indx1, species.index(sp)] / float(data['atm']['n_0'][p_indx1]),
                 color=colors[color_index], label=sp_lab)

    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')

    plt.ylim((1e-22, 2.))
    plt.legend(frameon=0, prop={'size': 13}, loc='best')

    plt.xlabel("Time(s)", fontsize=12)
    # plt.ylabel("Pressure (bar)")
    plt.ylabel("Mixing Ratio", fontsize=12)
    # plt.title('Earth (CIRA equator in January 1986)', fontsize=14)

    plt.title(f'Mixing ratio vs time\nat {plot_p / 1.e6} bar')

    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()

    return 0


def main():
    # setup paths
    # output_dir = os.path.expanduser('/data/vulcan_output_parallel/')
    output_dir = os.path.expanduser('/data/vulcan_output_parallel/')
    plot_base_dir = os.path.join(output_dir, 'plots/')
    plot_dir = os.path.join(plot_base_dir, 'evolution/')

    # create if it doesn't exist
    if not os.path.isdir(plot_base_dir):
        os.mkdir(plot_base_dir)

    # remake plot directory
    if os.path.isdir(plot_dir):
        shutil.rmtree(plot_dir)
    os.mkdir(plot_dir)
    # list vulcan output files
    vulcan_files = glob.glob(os.path.join(output_dir, '*.vul'))

    # setup_mp_params
    mp_params = [(plot_dir, vulcan_file) for vulcan_file in vulcan_files]
    num_workers = mp.cpu_count() - 1

    print("plotting abundances...")
    with mp.get_context("spawn").Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(plot_evolution, mp_params),  # return results otherwise it doesn't work properly
                            total=len(mp_params)))


if __name__ == "__main__":
    main()

