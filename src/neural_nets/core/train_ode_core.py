import os
import sys
from pathlib import Path

import torch

from mlp_core import MlpCore
from core_training_routine import train_core

# from torchdiffeq import odeint_adjoint as odeint    # VERY SLOW
from torchdiffeq import odeint


def loss_fn(y_pred, y_true):
    step_losses = torch.mean((y_pred - y_true) ** 2, dim=(0, 2))
    return torch.sum(step_losses)


# wrapper class to fit neural ode structure
class OdeCore(MlpCore):
    def __init__(self, device, steps, *args, **kwargs):
        self.output_dim = kwargs['y_mix_latent_dim']
        kwargs['y_mix_latent_dim'] = None

        super().__init__(*args, **kwargs)
        # self.t = torch.Tensor([steps]).to(device)
        self.t = torch.arange(0, steps, 1, dtype=torch.float64).to(device)
        self.ode_solver = 'dopri5'
        # self.solver_options = dict(step_size=1.0)
        self.solver_options = dict(max_num_steps=100*steps)
        # self.solver_options = None

    def _forward(self, latent_input):
        return MlpCore.forward(self, latent_input)

    def forward(self, t, latent_input):
        return self._forward(latent_input)


def model_step(latent_input, core_model, **kwargs):

    latent_model_outputs = odeint(
        core_model, latent_input, core_model.t,    # [1, batch, latent_dim]
        method=core_model.ode_solver, options=core_model.solver_options
    )
    return latent_model_outputs[-1, :, :core_model.output_dim]   # [batch, y_mix_latent_dim]


def model_step_time_series(latent_input, y_mixs_latent_outputs, core_model, loss_fn, **kwargs):

    latent_model_outputs = odeint(
        core_model, latent_input, core_model.t,    # [1, batch, latent_dim]
        method=core_model.ode_solver, options=core_model.solver_options
    )

    latent_model_outputs = latent_model_outputs.swapaxes(0, 1)  # [batch, steps, latent_dim]

    loss = loss_fn(latent_model_outputs[:, :, :core_model.output_dim], y_mixs_latent_outputs)

    return loss, latent_model_outputs[:, -1, :core_model.output_dim]   # [batch, y_mix_latent_dim]


def main():
    # setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MRP_dir = str(Path(script_dir).parents[2])
    dataset_dir = os.path.join(MRP_dir, 'data/bday_dataset/time_series_dataset')
    # dataset_dir = os.path.join(MRP_dir, 'data/christmas_dataset/clipped_dataset')
    save_model_dir = os.path.join(script_dir, '../saved_models')
    log_dir = os.path.join(script_dir, '../runs')

    # remake the config directory
    if not os.path.isdir(save_model_dir):
        os.mkdir(save_model_dir)

    params = dict(
        name='ode_core',

        gpu=0,

        ds_params={
            'batch_size': 8,
            'shuffle': True,
            'num_workers': 4,
            'train_test_validation_ratios': [0.7, 0.2, 0.1]
        },

        core_model=OdeCore,

        core_model_params={
            'latent_dim': (69 * 30 + 256 + 2 * 2 + 2 * 150),
            'layer_size': 2048,
            'y_mix_latent_dim': 69 * 30,
            'num_hidden': 2,
            'dropout': 0,
            'sigma': 0,
            'weight_decay_norm': 0,
            'batch_norm': False,
            'time_series': True,
        },

        core_model_extra_params={    # because the filename became too long...
            'steps': 10,
            'activation_function': 'tanh',
        },

        core_model_step=model_step_time_series,

        loss_function=loss_fn,

        optimizer_params={
            'lr': 1e-4
        },

        train_params={
            'epochs': 100,
            'writer_interval': 10,
        },
    )

    train_core(dataset_dir, save_model_dir, log_dir, params)


if __name__ == "__main__":
    main()
