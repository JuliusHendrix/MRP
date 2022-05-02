import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# own modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = str(Path(script_dir).parents[2])
sys.path.append(src_dir)


class MlpCore(nn.Module):
    def __init__(self, latent_dim, layer_size, num_hidden, activation_function, y_mix_latent_dim=None):
        super().__init__()

        self.latent_dim = latent_dim
        self.latent_dim2 = y_mix_latent_dim if y_mix_latent_dim is not None else latent_dim
        self.layer_size = layer_size

        # set activation function
        if activation_function == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        elif activation_function == 'tanh':
            self.activation_function = nn.Tanh()
        else:
            raise ValueError('Activation function not supported')

        # TODO: fix forward
        # TODO: activation function every layer?
        # create core
        self.core = nn.ModuleList()
        self.core.append(nn.Linear(latent_dim, layer_size))
        for k in range(num_hidden):
            self.core.append(nn.Linear(layer_size, layer_size))
        self.core.append(nn.Linear(layer_size, self.latent_dim2))

    def forward(self, latent_input):
        latent_output = latent_input.clone()
        for layer in self.core:
            latent_output = self.activation_function(layer(latent_output))
        return latent_output
