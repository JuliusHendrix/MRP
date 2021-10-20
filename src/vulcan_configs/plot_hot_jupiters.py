import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from astropy import units as u
from scipy.optimize import curve_fit

from vulcan_config_utils import analytic_MR

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = '../../data/hotjupiters.csv'
HJ_df = pd.read_csv(os.path.join(script_dir, csv_path))

names = HJ_df.columns
units = HJ_df.values[0, :]
data = HJ_df.values[1:, :]

print(names)
print(units)





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

    fig = plt.figure(figsize=(6, 4))
    plt.title(f'{propy} vs {propx} hot Jupiters')
    plt.scatter(x, y, s=1, color='k')
    plt.xlabel(f'{propx} [{u_x}]')
    plt.ylabel(f'{propy} [{u_y}]')
    if log:
        plt.xscale('log')
        plt.yscale('log')
    plt.tight_layout()

    if show:
        plt.show()

    return x, y


if __name__ == "__main__":
    M_p, R_p = plot_props_data('MASS', 'R', log=True, show=False)
    # add predicted relation

    R = [analytic_MR(m) for m in M_p]
    plt.plot(M_p, R, c='y')

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

    plt.plot(R_star, fit_func(R_star, *popt), c='y')

    plot_props_data('MSTAR', 'RSTAR', log=True)
    a, per = plot_props_data('A', 'PER', log=True)
    print(np.min(a), np.max(a))
    plot_props_data('A', 'MASS', log=True)
    plot_props_data('MSTAR', 'A', log=True)
