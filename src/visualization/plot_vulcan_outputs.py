import matplotlib.pyplot as plt
import os
import glob
import pickle
import shutil
import multiprocessing as mp
from tqdm import tqdm

use_height = False

# plot_spec = ('H', 'O', 'C', 'N')
plot_spec = ('H2', 'H', 'CO', 'H2O')
colors = ['k', 'y', 'b', 'pink']

# tex labels for plotting
tex_labels = {'H':'H','H2':'H$_2$','O':'O','OH':'OH','H2O':'H$_2$O','CH':'CH','C':'C','CH2':'CH$_2$','CH3':'CH$_3$','CH4':'CH$_4$','HCO':'HCO','H2CO':'H$_2$CO', 'C4H2':'C$_4$H$_2$',\
'C2':'C$_2$','C2H2':'C$_2$H$_2$','C2H3':'C$_2$H$_3$','C2H':'C$_2$H','CO':'CO','CO2':'CO$_2$','He':'He','O2':'O$_2$','CH3OH':'CH$_3$OH','C2H4':'C$_2$H$_4$','C2H5':'C$_2$H$_5$','C2H6':'C$_2$H$_6$','CH3O': 'CH$_3$O'\
,'CH2OH':'CH$_2$OH','N2':'N$_2$','NH3':'NH$_3$','HCN':'HCN','NO':'NO', 'NO2':'NO$_2$' }


def plot_vulcan_file(params):
    # extract params
    (plot_dir, vulcan_file) = params

    # extract filename
    filename = os.path.basename(vulcan_file)
    plot_filename = os.path.join(plot_dir, f'{filename[:-4]}.png')

    # extract data
    with open(vulcan_file, 'rb') as handle:
        data = pickle.load(handle)

    # plotting takes from plot_vulcan.py
    vulcan_spec = data['variable']['species']
    for color_index, sp in enumerate(plot_spec):

        if sp in tex_labels:
            sp_lab = tex_labels[sp]
        else:
            sp_lab = sp

        # plt.plot(data['variable']['ymix'][:,vulcan_spec.index(sp)], data['atm']['zco'][:-1]/1.e5, color=tableau20[color_index], label=sp_lab, lw=1.5)
        if use_height == False:
            plt.plot(data['variable']['ymix'][:, vulcan_spec.index(sp)], data['atm']['pco'] / 1.e6,
                     color=colors[color_index], label=sp_lab, lw=1.5)
        else:
            plt.plot(data['variable']['ymix'][:, vulcan_spec.index(sp)], data['atm']['zco'][1:] / 1.e5,
                     color=colors[color_index], label=sp_lab, lw=1.5)
        # plt.plot(data['variable']['y_ini'][:,vulcan_spec.index(sp)]/data['atm']['n_0'], data['atm']['pco']/1.e6, color=tableau20[color_index], ls=':', lw=1.5) # plotting the initial (equilibrium) abundances

    if use_height == False:
        plt.gca().set_yscale('log')
        plt.gca().invert_yaxis()
        plt.ylim((data['atm']['pco'][0] / 1e6, data['atm']['pco'][-1] / 1e6))
        plt.ylabel("Pressure (bar)")
    else:
        plt.ylim((data['atm']['zmco'][0] / 1e5, data['atm']['zmco'][-1] / 1e5))

    plt.xlabel("Mixing Ratio")

        # plt.title('T1400')

    plt.gca().set_xscale('log')
    # plt.xlim((1.E-12, 1.e-2))
    plt.legend(frameon=0, prop={'size': 12}, loc='best')
    # handles, labels = plt.gca().get_legend_handles_labels()
    # display = range(len(sp_list))
    # #Create custom artists
    # art0 = plt.Line2D((0,0),(0,0), ls='None')
    # Artist1 = plt.Line2D(range(10),range(10), color='black')
    # Artist2 = plt.Line2D((0,1),(0,0), color='black', ls='--',lw=1.5)
    # plt.legend([Artist1,Artist2],['Equilibrium','Kinetics'], frameon=False, prop={'size':12}, loc='best')

    # plt.title(f'{filename}')
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()

    return 0


def main():
    # setup paths
    # output_dir = os.path.expanduser('/data/vulcan_output_parallel/')
    output_dir = os.path.expanduser('/data/vulcan_output_parallel/')
    plot_base_dir = os.path.join(output_dir, 'plots/')
    plot_dir = os.path.join(plot_base_dir, 'mixing_ratios/')

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
        results = list(tqdm(pool.imap(plot_vulcan_file, mp_params),  # return results otherwise it doesn't work properly
                            total=len(mp_params)))


if __name__ == "__main__":
    main()
