import os
import sys
from pathlib import Path
import torch

# own modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = str(Path(script_dir).parents[2])
sys.path.append(src_dir)

from src.neural_nets.core.lstm_core import LSTMCore
from src.neural_nets.core.core_training_routine import train_core


def model_step(latent_input, core_model, device):
    output = latent_input    # (b, input_size)
    hidden, cell = core_model.init_hidden_cell(latent_input.shape[0], device)   # (1, b, hidden_size)
    for step in range(core_model.steps):
        output, hidden, cell = core_model(
            output.unsqueeze(dim=1),    # (b, 1, input_size)
            hidden,    # (1, b, hidden_size)
            cell,    # (1, b, hidden_size)
        )
    return output[:, :core_model.output_size]    # (b, y_mix_latent_dim)


def model_step_time_series(latent_input, y_mixs_latent_outputs, core_model, loss_fn, device):
    output = latent_input  # (b, input_size)
    hidden, cell = core_model.init_hidden_cell(latent_input.shape[0], device)  # (1, b, hidden_size)

    loss = 0

    # loop over time steps
    for step in range(core_model.steps):
        output, hidden, cell = core_model(
            output.unsqueeze(dim=1),  # (b, 1, input_size)
            hidden,  # (1, b, hidden_size)
            cell,  # (1, b, hidden_size)
        )

        # calculate loss
        t_loss = loss_fn(output[:, :core_model.output_size], y_mixs_latent_outputs[:, step, :])
        loss += t_loss

        # replace output with next input when training, except for last step
        if core_model.training and step < core_model.steps - 1:
            new_output = output.clone()
            new_output[:, :core_model.output_size] = y_mixs_latent_outputs[:, step, :]
            output = new_output

    return loss, output[:, :core_model.output_size]  # (b, y_mix_latent_dim)


def main():
    # setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MRP_dir = str(Path(script_dir).parents[2])
    dataset_dir = os.path.join(MRP_dir, 'data/bday_dataset/time_series_dataset')
    # dataset_dir = os.path.join(MRP_dir, 'data/christmas_dataset/clipped_dataset')
    save_model_dir = os.path.join(script_dir, '../saved_models_final')
    log_dir = os.path.join(script_dir, '../runs_final')

    # remake the config directory
    if not os.path.isdir(save_model_dir):
        os.mkdir(save_model_dir)

    params = dict(
        name='lstm_core',

        gpu=1,

        ds_params={
            'batch_size': 8,
            'shuffle': True,
            'num_workers': 4,
            'train_test_validation_ratios': [0.7, 0.2, 0.1]
        },

        core_model=LSTMCore,

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

        core_model_step=model_step_time_series,

        loss_function=torch.nn.MSELoss(),

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
