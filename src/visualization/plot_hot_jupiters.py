import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from astropy import units as u
from scipy.optimize import curve_fit
from pathlib import Path

# own modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = str(Path(script_dir).parents[1])
sys.path.append(src_dir)

from src.vulcan_configs.vulcan_config_utils import analytic_MR

csv_path = '../../data/hotjupiters.csv'
HJ_df = pd.read_csv(os.path.join(script_dir, csv_path))

names = HJ_df.columns
units = HJ_df.values[0, :]
data = HJ_df.values[1:, :]

print(names)
print(units)

colors = [
'#065e60', '#0a9396', '#99d5c9', '#ee9b00', '#ffb833', '#99d6ff', '#0093f5', '#00568f'
]

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def read_prop_data(properties):
    props_data = []
    props_units = []
    for prop in properties:
        prop_index = np.where(names == prop)[0][0]
        prop_unit = units[prop_index]
        if type(prop_unit) == str:
            prop_data = data[:, prop_index].astype(np.float64)
        else:
            prop_data = data[:, prop_index]

        props_data.append(prop_data)
        props_units.append(prop_unit)

    return props_data, props_units


def plot_props_data(propx, propy, log=False, show=True, ):
    # read data
    properties = [propx, propy]
    props_data, props_units = read_prop_data(properties)
    x, y = props_data
    u_x, u_y = props_units

    if show:
        fig = plt.figure(figsize=(6, 4))
        plt.title(f'{propy} vs {propx} hot Jupiters')
        plt.scatter(x, y, s=1, color='k')
        plt.xlabel(f'{propx} [{u_x}]')
        plt.ylabel(f'{propy} [{u_y}]')
        if log:
            plt.xscale('log')
            plt.yscale('log')
        plt.tight_layout()

        plt.show()

    return x, y


def fancy_plot_fit(T, R, popt):
    # fit power func
    def fit_func(x, a, b):
        return a * x ** b

    sorted_R = np.sort(R)

    f, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(R, T, s=10, color='k', alpha=0.3, marker='.')
    ax.plot(sorted_R, fit_func(sorted_R, *popt), color=colors[6],
            label=rf'$y(x) = {{{popt[0]:.1f}}} \cdot x^{{{popt[1]:.2f}}}$')
    ax.set_ylabel('Effective Temperature [K]')
    ax.set_xlabel(r'Stellar Radius [R$_{\mathrm{Sun}}$]')
    ax.set_title('Effective Temperature vs Stellar Radius')
    # ax.set_xscale('log')
    # ax.set_yscale('log')

    ax.legend(loc='lower right')

    plt.savefig(
        'random_images/stellar_parameters_fit.pdf',
        bbox_inches='tight')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    M_p, R_p = plot_props_data('MASS', 'R', log=True, show=False)
    # add predicted relation

    R = [analytic_MR(m) for m in M_p]
    # plt.plot(M_p, R, c='y')

    R_star, T_eff = plot_props_data('RSTAR', 'TEFF', log=True, show=False)

    # slice to remove outliers
    T_eff = T_eff[R_star < 1.5]
    R_star = R_star[R_star < 1.5]

    # remove nans
    T_eff = T_eff[~np.isnan(R_star)]
    R_star = R_star[~np.isnan(R_star)]

    R_star = R_star[~np.isnan(T_eff)]
    T_eff = T_eff[~np.isnan(T_eff)]

    # fit power func
    def fit_func(x, a, b):
        return a * x**b

    popt, pconv = curve_fit(fit_func, R_star, T_eff)
    print(popt)

    fancy_plot_fit(T_eff, R_star, popt)

    # plt.plot(R_star, fit_func(R_star, *popt), c='y')
    #
    # plot_props_data('MSTAR', 'RSTAR', log=True)
    # a, per = plot_props_data('A', 'PER', log=True)
    # print(np.min(a), np.max(a))
    # plot_props_data('A', 'MASS', log=True)
    # plot_props_data('MSTAR', 'A', log=True)
