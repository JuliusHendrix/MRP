import os
import sys
from pathlib import Path

# own modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = str(Path(script_dir).parents[2])
sys.path.append(src_dir)

from src.neural_nets.individualAEs.MRAE.MixingRatioAE import MixingRatioAE
from src.neural_nets.individualAEs.FAE.FluxAE import FluxAE
from src.neural_nets.individualAEs.CopyAE.CopyAE import CopyAE

ae_params = dict(
    models={
        'mrae': MixingRatioAE,
        'wae': FluxAE,
        'fae': FluxAE,
        'pae': MixingRatioAE,
        'tae': CopyAE,  # copy bc MRAE cannot encode well for some reason
        'gae': CopyAE,
    },

    state_dicts={
        'mrae': "MRAE_d_interp,hparams={'latent_dim': 30, 'layer_size': 256, 'activation_function': 'tanh', 'lr': 1e-05}_state_dict",
        'wae': "WAE,hparams={'latent_dim': 2, 'layer_size': 1024, 'lr': 1e-06}_state_dict",
        'fae': "FAE,hparams={'latent_dim': 256, 'layer_size': 1024, 'activation_function': 'tanh', 'lr': 1e-05}_state_dict",
        'pae': "PAE,hparams={'latent_dim': 2, 'layer_size': 256, 'lr': 1e-06}_state_dict",
        'tae': None,  # "TAE,hparams={'latent_dim': 75, 'layer_size': 256, 'lr': 1e-05}_state_dict",
        'gae': None,  # "gAE,hparams={'latent_dim': 75, 'layer_size': 256, 'lr': 1e-05}_state_dict",
    },

    model_params={
        'mrae': {
            'latent_dim': 30,
            'layer_size': 256,
            'activation_function': 'tanh',
        },
        'wae': {
            'latent_dim': 2,
            'layer_size': 1024,
            'activation_function': 'leaky_relu',
        },
        'fae': {
            'latent_dim': 256,
            'layer_size': 1024,
            'activation_function': 'tanh',
        },
        'pae': {
            'latent_dim': 2,
            'layer_size': 256,
            'activation_function': 'leaky_relu',
        },
        'tae': {
            # 'latent_dim': 75,
            # 'layer_size': 256,
            # 'activation_function': 'leaky_relu',
        },
        'gae': {
            # 'latent_dim': 75,
            # 'layer_size': 256,
            # 'activation_function': 'leaky_relu',
        },
    },
)
