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
from torch.utils.data import DataLoader
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
from src.neural_nets.dataset_utils import unscale_example, create_scaling_dict, scale_dataset
from src.neural_nets.dataloaders import SingleVulcanDataset
from src.neural_nets.interpolate_dataset import interpolate_dataset

# TODO: don't know if this is nescessary
# Limiting the number of threads
os.environ["OMP_NUM_THREADS"] = "1"


def cut_values(dataset_dir, threshold, spec_list):
    # dataset loader
    vulcan_dataset = SingleVulcanDataset(dataset_dir)
    dataloader = DataLoader(vulcan_dataset, batch_size=1,
                            shuffle=True,
                            num_workers=0)
    # get scaling parameters
    scaling_file = os.path.join(dataset_dir, 'scaling_dict.pkl')
    with open(scaling_file, 'rb') as f:
        scaling_params = pickle.load(f)

    # create tot dict
    for i, dummy_example in enumerate(dataloader):
        unscaled_dummy_example = unscale_example(dummy_example, scaling_params)
        tot_dict = unscaled_dummy_example.copy()
        for top_key, top_value in tot_dict.items():
            for key, value in top_value.items():
                zero_value = np.zeros_like(value)
                tot_dict[top_key][key] = np.tile(zero_value[..., None], len(dataloader))
        break

    # loop through examples
    with tqdm(dataloader, unit='example', desc=f'Summing values') as dataloader:
        for i, example in enumerate(dataloader):
            # unscale dict
            unscaled_dict = unscale_example(example, scaling_params)

            # add
            for top_key, top_value in tot_dict.items():
                for key, value in top_value.items():
                    tot_dict[top_key][key][..., i] = unscaled_dict[top_key][key]

    # calculate means
    agg_dict = tot_dict.copy()
    for top_key, top_value in tot_dict.items():
        for key, value in top_value.items():
            agg_dict[top_key][key] = np.median(value, axis=-1)

    y_mix_ini = agg_dict['inputs']['y_mix_ini'].swapaxes(0, 1)
    y_mix_ini_median_height = np.median(y_mix_ini, axis=1)

    inds = np.where(y_mix_ini_median_height > threshold)[0]
    print(f'cutting to {len(inds)} species...')
    spec_list = spec_list[inds]

    torch_files = glob.glob(os.path.join(dataset_dir, '*.pt'))

    for torch_file in tqdm(torch_files, desc='cutting torch files'):
        example = torch.load(torch_file)

        cut_example = example.copy()
        cut_example['inputs']['y_mix_ini'] = example['inputs']['y_mix_ini'][:, inds]
        cut_example['outputs']['y_mix'] = example['outputs']['y_mix'][:, inds]

        torch.save(cut_example, torch_file)

    return spec_list


def ini_vulcan():
    """
    Initial steps of a VULCAN simulation, taken from vulcan.py.
    """

    # import the configuration inputs
    # reload vulcan_cfg per worker
    import vulcan_cfg
    importlib.reload(sys.modules['vulcan_cfg'])

    # import chem_funs
    import chem_funs

    # import VULCAN modules
    import store, build_atm, op

    from phy_const import kb, Navo, au, r_sun

    from chem_funs import ni, nr  # number of species and reactions in the network

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
    data_var.bins = bins

    return data_atm, data_var


def generate_inputs(mode):
    # generate simulation state
    data_atm, data_var = ini_vulcan()

    # flux
    top_flux = data_var.sflux_top    # (2500,)
    wavelengths = data_var.bins    # (2500,)

    # TP-profile
    Pco = data_atm.pco  # (150,)
    Tco = data_atm.Tco  # (150,)

    # initial abundances
    y_ini = data_var.y_ini  # (150, 69)

    # mixing  ratios
    total_abundances = np.sum(y_ini, axis=-1)
    y_mix_ini = y_ini / np.tile(total_abundances[..., None], y_ini.shape[-1])

    print(f'\n------------')
    print(f'{y_mix_ini.min() = }')
    print(f'{top_flux.min() = }')

    if mode == 'clipped':
        # clipping of values
        y_mix_ini = np.where(y_mix_ini < 1e-14, 1e-14, y_mix_ini)

    top_flux = np.where(top_flux < 1e-10, 1e-10, top_flux)

    # gravity
    g = data_atm.g  # (150,)    # TODO: waarom ook deze?

    # surface gravity
    gs = data_atm.gs  # ()

    # to tensors
    inputs = {
        "y_mix_ini": torch.from_numpy(y_mix_ini),    # (150, 69)
        "Tco": torch.from_numpy(Tco),    # (150, )
        "Pco": torch.from_numpy(Pco),    # (150, )
        "g": torch.from_numpy(g),    # (150, )
        "top_flux": torch.from_numpy(top_flux),    # (2500,)
        "wavelengths": torch.from_numpy(wavelengths),    # (2500,)
        "gravity": torch.tensor(gs),    # ()
    }

    return inputs


def generate_output(vul_file, mode):
    # extract data
    with open(vul_file, 'rb') as handle:
        data = pickle.load(handle)

    y = data['variable']['y']

    # mixing  ratios
    total_abundances = np.sum(y, axis=-1)
    y_mix = y / np.tile(total_abundances[..., None], y.shape[-1])

    if mode == 'clipped':
        # clipping of values
        y_mix = np.where(y_mix < 1e-14, 1e-14, y_mix)

    outputs = {
        "y_mix": torch.from_numpy(y_mix)    # (150, 69)
    }

    return outputs


def generate_output_time(vul_file, mode):
    # extract data
    with open(vul_file, 'rb') as handle:
        data = pickle.load(handle)

    y_time = data['variable']['y_time']    # (256, 150, 69)

    # 10 evenly spaced integers including first and last, but ignore first time as it is practically t=0
    idx = np.round(np.linspace(0, y_time.shape[0] - 1, 11)[1:]).astype(int)    # (10,)
    y_t = y_time[idx, :, :]  # (10, 150, 69)
    total_abundances = np.sum(y_t, axis=-1)  # (10, 150)
    y_mixs = y_t / np.tile(total_abundances[..., None], y_t.shape[-1])  # (10, 150, 69)

    if mode == 'clipped':
        # clipping of values
        y_mix = np.where(y_mixs < 1e-14, 1e-14, y_mixs)

    outputs = {
        "y_mixs": torch.from_numpy(y_mixs)    # (150, 69)
    }

    return outputs


def generate_input_output_pair(params):
    """
    Generate simulation input and output pair.
    """

    # extract params
    (i, config_file, copy_manager, output_dir, dataset_dir, mode, time_series) = params

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
    inputs = generate_inputs(mode)

    # generate output tensor
    vul_file = os.path.join(output_dir, f'output_{cf_name[11:-3]}.vul')

    if time_series:
        outputs = generate_output_time(vul_file, mode)
    else:
        outputs = generate_output(vul_file, mode)

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


def main(num_workers, generate=True):
    # setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    git_dir = str(Path(script_dir).parents[2])
    # output_dir = os.path.join(git_dir, 'MRP/data/christmas_dataset/vulcan_output')
    output_dir = os.path.join(git_dir, 'MRP/data/bday_dataset/vulcan_output')
    # config_dir = os.path.join(git_dir, 'MRP/data/christmas_dataset/configs')
    config_dir = os.path.join(git_dir, 'MRP/data/bday_dataset/configs')
    VULCAN_dir = os.path.join(git_dir, 'VULCAN')

    mode = ''    # '', 'clipped', 'cut'
    time_series = True

    if mode == '':
        dataset_dir = os.path.join(git_dir, 'MRP/data/bday_dataset/time_series_dataset')
    else:
        dataset_dir = os.path.join(git_dir, f'MRP/data/bday_dataset/{mode}_dataset')

    if generate:
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
        mp_params = [(i, config_file, mp_copy_manager, output_dir, dataset_dir, mode, time_series)
                     for i, config_file in enumerate(config_files)]

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

    # save scaling dict
    create_scaling_dict(dataset_dir)

    # scale dataset
    scale_dataset(dataset_dir)

    # save species list
    os.chdir(VULCAN_dir)
    sys.path.append(VULCAN_dir)

    from chem_funs import spec_list

    if mode == 'cut':
        spec_list = np.array(spec_list)
        spec_list = cut_values(dataset_dir, 1e-30, spec_list)
        spec_list = spec_list.tolist()

    species_list_file = os.path.join(dataset_dir, 'species_list.pkl')
    with open(species_list_file, 'wb') as f:
        pickle.dump(spec_list, f)

    interpolate_dataset(dataset_dir, num_workers=num_workers, time_series=time_series)


if __name__ == "__main__":
    # run main
    main(num_workers=mp.cpu_count() - 1, generate=True)
