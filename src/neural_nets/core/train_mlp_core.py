import os
import sys
from pathlib import Path

from mlp_core import MlpCore
from core_training_routine import train_core


def model_step(latent_input, core_model, **kwargs):
    latent_model_output = core_model(latent_input)
    return latent_model_output


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
        name='mlp_core',

        gpu=1,

        ds_params={
            'batch_size': 4,
            'shuffle': True,
            'num_workers': 4,
            'train_test_validation_ratios': [0.7, 0.2, 0.1]
        },

        core_model=MlpCore,

        core_model_params={
            'latent_dim': (69 * 30 + 256 + 2 * 2 + 2 * 150),
            'layer_size': 4096,
            'y_mix_latent_dim': 69 * 30,
            'num_hidden': 6,
            'activation_function': 'tanh',
        },

        core_model_step=model_step,

        optimizer_params={
            'lr': 1e-5
        },

        train_params={
            'epochs': 200,
            'writer_interval': 10,
        },
    )

    train_core(dataset_dir, save_model_dir, log_dir, params)


if __name__ == "__main__":
    main()
