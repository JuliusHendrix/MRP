import os
import glob
import sys
from pathlib import Path
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import shutil
import psutil
from tqdm import tqdm
import numpy as np
import torch
import pickle
import importlib
import argparse

# import public modules
import matplotlib.pyplot as plt
import matplotlib.legend as lg
import scipy
from scipy.interpolate import interp1d
import scipy.optimize as sop
import time, timeit, os, sys
import ast

# own modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = str(Path(script_dir).parents[1])
sys.path.append(src_dir)

from src.vulcan_configs.vulcan_config_utils import CopyManager

# TODO: don't know if this is nescessary
# Limiting the number of threads
os.environ["OMP_NUM_THREADS"] = "1"


def ini_vulcan():
    """
    Initial steps of a VULCAN simulation, taken from vulcan.py.
    """

    # import the configuration inputs
    # reload vulcan_cfg per worker
    import vulcan_cfg
    importlib.reload(sys.modules['vulcan_cfg'])

    # import VULCAN modules
    import store, build_atm, op
    try:
        import chem_funs
    except:
        raise IOError('\nThe module "chem_funs" does not exist.\nPlease run prepipe.py first to create the module...')

    from phy_const import kb, Navo, au, r_sun

    from chem_funs import ni, nr  # number of species and reactions in the network

    species = chem_funs.spec_list

    ### read in the basic chemistry data
    with open(vulcan_cfg.com_file, 'r') as f:
        columns = f.readline()  # reading in the first line
        num_ele = len(columns.split()) - 2  # number of elements (-2 for removing "species" and "mass")
    type_list = ['int' for i in range(num_ele)]
    type_list.insert(0, 'U20');
    type_list.append('float')
    compo = np.genfromtxt(vulcan_cfg.com_file, names=True, dtype=type_list)
    # dtype=None in python 2.X but Sx -> Ux in python3
    compo_row = list(compo['species'])
    ### read in the basic chemistry data

    ### creat the instances for storing the variables and parameters
    data_var = store.Variables()
    data_atm = store.AtmData()
    data_para = store.Parameters()

    # record starting CPU time
    data_para.start_time = time.time()

    make_atm = build_atm.Atm()

    # construct pico
    data_atm = make_atm.f_pico(data_atm)
    # construct Tco and Kzz
    data_atm = make_atm.load_TPK(data_atm)
    # construct Dzz (molecular diffusion)

    # Only setting up ms (the species molecular weight) if vulcan_cfg.use_moldiff == False
    make_atm.mol_diff(data_atm)

    # calculating the saturation pressure
    if vulcan_cfg.use_condense == True: make_atm.sp_sat(data_atm)

    # for reading rates
    rate = op.ReadRate()

    # read-in network and calculating forward rates
    data_var = rate.read_rate(data_var, data_atm)

    # for low-T rates e.g. Jupiter
    if vulcan_cfg.use_lowT_limit_rates == True: data_var = rate.lim_lowT_rates(data_var, data_atm)

    # reversing rates
    data_var = rate.rev_rate(data_var, data_atm)
    # removing rates
    data_var = rate.remove_rate(data_var)

    ini_abun = build_atm.InitialAbun()
    # initialing y and ymix (the number density and the mixing ratio of every species)
    data_var = ini_abun.ini_y(data_var, data_atm)

    # storing the initial total number of atmos
    data_var = ini_abun.ele_sum(data_var)

    # calculating mean molecular weight, dz, and dzi and plotting TP
    data_atm = make_atm.f_mu_dz(data_var, data_atm, output=None)

    # specify the BC
    make_atm.BC_flux(data_atm)


    # Setting up for photo chemistry
    if vulcan_cfg.use_photo == True:
        rate.make_bins_read_cross(data_var, data_atm)
        # rate.read_cross(data_var)
        make_atm.read_sflux(data_var, data_atm)

    # modified vulcan code but with 2500 points so all spectra are uniform
    bins = np.linspace(data_var.bins[0], data_var.bins[-1], 2500)

    sflux_top = np.zeros(len(bins))

    inter_sflux = interp1d(data_atm.sflux_raw['lambda'], data_atm.sflux_raw['flux'] * (
                vulcan_cfg.r_star * r_sun / (au * vulcan_cfg.orbit_radius)) ** 2, bounds_error=False, fill_value=0)

    for n, ld in enumerate(bins):
        sflux_top[n] = inter_sflux(ld)

    data_var.sflux_top = sflux_top

    return data_atm, data_var


def distribution_standardization(prop, prop_mean=None, prop_std=None):
    if not prop_mean:
        prop_mean = np.mean(prop)
    if not prop_std:
        prop_std = np.std(prop)

    scaled_prop = (prop - prop_mean) / prop_std

    return scaled_prop, prop_mean, prop_std


def generate_inputs():
    # generate simulation state
    data_atm, data_var = ini_vulcan()

    # flux
    top_flux = data_var.sflux_top    # (2500,)

    # TP-profile
    Pco = data_atm.pco  # (150,)
    Tco = data_atm.Tco  # (150,)

    # initial abundances
    y_ini = data_var.y_ini  # (150, 69)

    # mixing  ratios
    total_abundances = np.sum(y_ini, axis=-1)
    y_mix_ini = y_ini / np.tile(total_abundances[..., None], y_ini.shape[-1])

    # mean molecular mass
    g = data_atm.g  # (150,)    # TODO: waarom ook deze?

    # surface gravity
    gs = data_atm.gs  # ()

    # scaling
    y_mix_ini_scaled, y_mix_ini_mean, y_mix_ini_std = distribution_standardization(np.log10(y_mix_ini))
    Tco_scaled, Tco_mean, Tco_std = distribution_standardization(np.log10(Tco))
    g_scaled, g_mean, g_std = distribution_standardization(np.log10(g))
    Pco_scaled = np.log10(Pco)
    top_flux_scaled = top_flux / 1e5
    gs_scaled = np.log10(gs)

    scaling_parameters = {
        "y_mix_ini": (y_mix_ini_mean, y_mix_ini_std),
        "Tco": (Tco_mean, Tco_std),
        "g": (g_mean, g_std)
    }

    # to tensors
    inputs = {
        "y_mix_ini": torch.from_numpy(y_mix_ini_scaled),    # (150, 69)
        "Tco": torch.from_numpy(Tco_scaled),    # (150, )
        "Pco": torch.from_numpy(Pco_scaled),    # (150, )
        "g": torch.from_numpy(g_scaled),    # (150, )
        "top_flux": torch.from_numpy(top_flux_scaled),    # (2500,)
        "gravity": torch.tensor(gs_scaled),    # ()
        "scaling_parameters": scaling_parameters
    }

    return inputs


def generate_output(vul_file, scaling_parameters):
    # extract data
    with open(vul_file, 'rb') as handle:
        data = pickle.load(handle)

    y = data['variable']['y']

    # mixing  ratios
    total_abundances = np.sum(y, axis=-1)
    y_mix = y / np.tile(total_abundances[..., None], y.shape[-1])

    # same scaling as input
    y_scaling = scaling_parameters["y_mix_ini"]
    y_mix_scaled, _, _ = distribution_standardization(np.log10(y_mix),
                                                      prop_mean=y_scaling[0],
                                                      prop_std=y_scaling[1])

    outputs = {
        "y_mix": torch.from_numpy(y_mix_scaled)    # (150, 69)
    }

    return outputs


def generate_input_output_pair(params):
    """
    Generate simulation input and output pair.
    """

    # extract params
    (i, config_file, copy_manager, output_dir, dataset_dir) = params

    # get available VULCAN dir copy
    available_dir = copy_manager.get_available_copy()

    # change working directory of this process
    os.chdir(available_dir)
    sys.path.append(available_dir)

    # copy config file to VULCAN directory
    shutil.copyfile(config_file, os.path.join(available_dir, 'vulcan_cfg.py'))

    # make std_output redirect file
    cf_name = os.path.basename(config_file)

    # print info
    print(
        f'\n{mp.current_process()}'
        f'\non cpu {psutil.Process().cpu_num()}'
        f'\nin {os.getcwd()}'
        f'\nwith {os.path.basename(config_file)}\n'
    )

    # generate input tensors
    inputs = generate_inputs()

    # generate output tensor
    vul_file = os.path.join(output_dir, f'output_{cf_name[11:-3]}.vul')
    outputs = generate_output(vul_file, inputs["scaling_parameters"])

    # save example
    example = {
        'inputs': inputs,
        'outputs': outputs
    }

    filename = f'{i:04}.pt'
    torch.save(example, os.path.join(dataset_dir, filename))

    # add VULCAN dir copy back to list
    copy_manager.add_used_copy(available_dir)

    # print info
    print(
        f'exiting'
        f'\n{mp.current_process()}\n'
    )

    # make dict entry
    entry = {
        str(i): cf_name
    }

    return entry


def main(num_workers):
    # setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    git_dir = str(Path(script_dir).parents[2])
    output_dir = os.path.join(git_dir, 'MRP/data/christmas_dataset/vulcan_output')
    config_dir = os.path.join(git_dir, 'MRP/data/christmas_dataset/configs')
    VULCAN_dir = os.path.join(git_dir, 'VULCAN')
    dataset_dir = os.path.join(git_dir, 'MRP/data/christmas_dataset/dataset')

    # create dataset dir
    if os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.mkdir(dataset_dir)

    # extract saved config files, but in .txt format for some reason?
    config_files = glob.glob(os.path.join(config_dir, '*.py'))

    # setup copy manager
    BaseManager.register('CopyManager', CopyManager)
    manager = BaseManager()
    manager.start()
    mp_copy_manager = manager.CopyManager(num_workers, VULCAN_dir)

    # setup_mp_params
    mp_params = [(i, config_file, mp_copy_manager, output_dir, dataset_dir) for i, config_file in enumerate(config_files)]

    # run parallel
    print(f'running with {num_workers} workers...')
    with mp.get_context("spawn").Pool(processes=num_workers) as pool:
        entries = list(tqdm(pool.imap(generate_input_output_pair, mp_params),  # return results otherwise it doesn't work properly
                            total=len(mp_params)))

    # save index dict
    index_dict = {}
    for entry in entries:
        index_dict.update(entry)

    index_dict_file = os.path.join(dataset_dir, 'index_dict.pkl')
    with open(index_dict_file, 'wb') as f:
        pickle.dump(index_dict, f)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Run the vulcan configurations')
    parser.add_argument('-w', '--workers', help='Number of multiprocessing-subprocesses', type=int, default=mp.cpu_count() - 1,
                        required=False)
    # parser.add_argument('-b', '--batch', help='Number of pairs per .pt file', type=int, default=64,
    #                     required=False)
    args = vars(parser.parse_args())

    # run main
    main(num_workers=args['workers'])
