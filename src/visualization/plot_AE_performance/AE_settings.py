import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np
from tqdm import tqdm
import copy
from functools import partial

# own modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = str(Path(script_dir).parents[1])
sys.path.append(src_dir)

from src.neural_nets.dataset_utils import unscale, scale
from src.neural_nets.NN_utils import move_to

from src.neural_nets.individualAEs.MRAE.MixingRatioAE import MixingRatioAE
from src.neural_nets.individualAEs.MRAE.train_MRAE import model_step as MRAE_model_step

from src.neural_nets.individualAEs.FAE.FluxAE import FluxAE
from src.neural_nets.individualAEs.FAE.train_FAE import model_step as FAE_model_step

from src.neural_nets.individualAEs.WAE.train_WAE import model_step as WAE_model_step

from src.neural_nets.individualAEs.HAE.HAE_routine import model_step as HAE_model_step

from src.neural_nets.dataloaders import MixingRatioVulcanDataset
from src.neural_nets.dataloaders import SingleVulcanDataset

AE_settings = dict(

    MRAE = dict(
        model=MixingRatioAE,

        name='MRAE',

        model_params={
            'latent_dim': 30,
            'layer_size': 256,
            'activation_function': 'tanh',
        },

        optimizer_params={
            'lr': 1e-5
        },

        model_step=MRAE_model_step,

        variable_name='y_mix_ini',

        dataloader=MixingRatioVulcanDataset,

        plot_params={
            'dim_names': ['Height layer', 'Mixing Ratio'],
            'xlog': False,
            'ylog': True,
            'label': 'Mixing Ratio',
            'title': 'Mixing Ratio',
        }
    ),

    FAE = dict(
        model=FluxAE,

        name='FAE',

        model_params={
            'latent_dim': 256,
            'layer_size': 1024,
            'activation_function': 'tanh',
        },

        optimizer_params={
            'lr': 1e-5
        },

        model_step=FAE_model_step,

        variable_name='top_flux',

        dataloader=SingleVulcanDataset,

        plot_params={
            'dim_names': ['index', 'Flux'],
            'xlog': False,
            'ylog': True,
            'label': r'Flux [erg nm$^{-1}$ cm$^{-2}$ s$^{-1}$]',
            'title': 'Flux',
        }
    ),

    WAE = dict(
        model=FluxAE,

        name='WAE',

        model_params={
            'latent_dim': 2,
            'layer_size': 1024,
            'activation_function': 'tanh',
        },

        optimizer_params={
            'lr': 5e-7
        },

        model_step=WAE_model_step,

        variable_name='wavelengths',

        dataloader=SingleVulcanDataset,

        plot_params={
            'dim_names': ['index', 'Wavelength'],
            'xlog': False,
            'ylog': False,
            'label': 'Wavelength [nm]',
            'title': 'Wavelength',
        }
    ),

    PAE = dict(
        model=MixingRatioAE,

        name='PAE',

        model_params={
            'latent_dim': 2,
            'layer_size': 256,
            'activation_function': 'tanh',
        },

        optimizer_params={
            'lr': 1e-6
        },

        model_step=partial(HAE_model_step, variable_key='Pco'),

        variable_name='Pco',

        dataloader=SingleVulcanDataset,

        plot_params={
            'dim_names': ['Height layer', 'Pressure gradient'],
            'xlog': False,
            'ylog': True,
            'label': r'Pressure [10$^{6}$ bar]',
            'title': 'Pressure gradient',
        }
    ),

    GAE=dict(
        model=MixingRatioAE,

        name='gAE',

        model_params={
            'latent_dim': 75,
            'layer_size': 256,
            'activation_function': 'tanh',
        },

        optimizer_params={
            'lr': 1e-5
        },

        model_step=partial(HAE_model_step, variable_key='g'),

        variable_name='g',

        dataloader=SingleVulcanDataset,

        plot_params={
            'dim_names': ['Height layer', 'Gravity gradient'],
            'xlog': False,
            'ylog': False,
            'label': r'Gravity [g cm s$^{-2}$]',
            'title': 'Gravity gradient',
        }
    ),

    TAE=dict(
        model=MixingRatioAE,

        name='TAE',

        model_params={
            'latent_dim': 75,
            'layer_size': 256,
            'activation_function': 'tanh',
        },

        optimizer_params={
            'lr': 1e-5
        },

        model_step=partial(HAE_model_step, variable_key='Tco'),

        variable_name='Tco',

        dataloader=SingleVulcanDataset,

        plot_params={
            'dim_names': ['Height layer', 'Temperature gradient'],
            'xlog': False,
            'ylog': False,
            'label': 'Temperature [K]',
            'title': 'Temperature gradient',
        }
    ),

)

def get_params(AE_name):
    return AE_settings[AE_name]
