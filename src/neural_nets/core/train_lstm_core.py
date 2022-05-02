import os
import sys
from pathlib import Path

from lstm_core import LSTMCore
from core_training_routine import train_core


def model_step(latent_input, core_model, device):
    output = latent_input    # (b, input_size)
    hidden, cell = core_model.init_hidden_cell(latent_input.shape[0], device)   # (1, b, hidden_size)
    for step in range(core_model.steps):
        output, hidden = core_model(
            output.unsqueeze(dim=1),    # (b, 1, input_size)
            hidden,    # (1, b, hidden_size)
            cell,    # (1, b, hidden_size)
        )
    return output[:, :core_model.output_size]    # (b, y_mix_latent_dim)


def main():
    # setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MRP_dir = str(Path(script_dir).parents[2])
    dataset_dir = os.path.join(MRP_dir, 'data/bday_dataset/dataset')
    # dataset_dir = os.path.join(MRP_dir, 'data/christmas_dataset/clipped_dataset')
    save_model_dir = os.path.join(script_dir, '../saved_models')
    log_dir = os.path.join(script_dir, '../runs')

    # remake the config directory
    if not os.path.isdir(save_model_dir):
        os.mkdir(save_model_dir)

    params = dict(
        name='lstm_core',

        gpu=0,

        ds_params={
            'batch_size': 4,
            'shuffle': True,
            'num_workers': 4,
            'train_test_validation_ratios': [0.7, 0.2, 0.1]
        },

        core_model=LSTMCore,

        core_model_params={
            'input_size': (69 * 30 + 256 + 2 * 2 + 2 * 150),
            'hidden_size': (69 * 30 + 256 + 2 * 2 + 2 * 150),
            'output_size': 69 * 30,
            'steps': 10,
            'activation_function': 'tanh',
        },

        core_model_step=model_step,

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
