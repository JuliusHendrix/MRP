import os
import sys
from pathlib import Path

# own modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = str(Path(script_dir).parents[3])
sys.path.append(src_dir)

from src.neural_nets.NN_utils import LossWeightScheduler
from HAE_routine import train_autoencoder


def main():
    # setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MRP_dir = str(Path(script_dir).parents[3])    # TODO: same as src_dir?
    dataset_dir = os.path.join(MRP_dir, 'data/bday_dataset/dataset')
    # dataset_dir = os.path.join(MRP_dir, 'data/christmas_dataset/clipped_dataset')
    # dataset_dir = os.path.join(MRP_dir, 'data/christmas_dataset/cut_dataset')
    save_model_dir = os.path.join(MRP_dir, 'src/neural_nets/saved_models')
    log_dir = os.path.join(MRP_dir, 'src/neural_nets/runs')

    # make save directory if not present
    if not os.path.isdir(save_model_dir):
        os.mkdir(save_model_dir)

    params = dict(
        name='gAE',

        gpu=1,

        ds_params={
            'batch_size': 4,
            'shuffle': True,
            'num_workers': 4,
            'train_test_validation_ratios': [0.7, 0.2, 0.1]
        },

        model_params={
            'latent_dim': 75,
            'layer_size': 256
        },

        optimizer_params={
            'lr': 1e-5
        },

        loss_params={
            'LossWeightScheduler_d': LossWeightScheduler(
                start_epoch=100,
                end_epoch=200,
                start_weight=0.1,
                end_weight=1e3
            ),
        },

        train_params={
            'epochs': 300,
            'writer_interval': 10,
            'variable_key': 'g'
        },

        plot_params={
            'ylabel': 'Gravity (g cm / s2)',
            'ylog': False
        }
    )

    train_autoencoder(dataset_dir, save_model_dir, log_dir, params)


if __name__ == "__main__":
    main()
