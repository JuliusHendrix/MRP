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

from src.neural_nets.core.lstm_core import LSTMCore
from src.neural_nets.core.train_lstm_core import model_step as LSTM_model_step

settings = dict(

    LSTM = dict(
        model=LSTMCore,

        name='lstm_core',

        core_model_params={
            'input_size': (69 * 30 + 256 + 2 * 2 + 2 * 150),
            'hidden_size': 4096,
            'output_size': 69 * 30,
            'time_series': True,
            'sigma': 0,
            'weight_decay_norm': 0,
        },

        core_model_extra_params={  # because the filename became too long...
            'steps': 10,
            'activation_function': 'tanh',
        },

        optimizer_params={
            'lr': 1e-4
        },

        model_step=LSTM_model_step,

        plot_params={
        }
    ),
)

def get_params(name):
    return settings[name]
