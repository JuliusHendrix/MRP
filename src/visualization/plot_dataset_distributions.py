import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# own modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = str(Path(script_dir).parents[1])
sys.path.append(src_dir)

from src.neural_nets.dataset_utils import VulcanDataset


def scale_prop(prop):
    prop_mean = np.mean(prop)
    prop_std = np.std(prop)
    scaled_prop = (prop - prop_mean) / prop_std

    return scaled_prop, prop_mean, prop_std


def main():
    # setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MRP_dir = str(Path(script_dir).parents[1])
    dataset_dir = os.path.join(MRP_dir, 'data/dataset')

    # dataset loader
    vulcan_dataset = VulcanDataset(dataset_dir)
    dataloader = DataLoader(vulcan_dataset, batch_size=1,
                            shuffle=True,
                            num_workers=0)

    # loop through examples
    for i, example in enumerate(dataloader):
        # extract inputs
        y_mix = example['inputs']['y_mix']
        Tco = example['inputs']['Tco']
        Pco = example['inputs']['Pco']
        g = example['inputs']['g']
        top_flux = example['inputs']['top_flux']
        gravity = example['inputs']['gravity']

        scaled_height = np.log10(y_mix.numpy().flatten())
        # scaled_height = (scaled_height - np.min(scaled_height)) / (np.max(scaled_height) - np.min(scaled_height))
        scaled_height, _, _ = scale_prop(scaled_height)

        scaled_Tco = np.log10(Tco.numpy().flatten())
        scaled_Tco, _, _ = scale_prop(scaled_Tco)

        # TODO: constant for all examples?
        scaled_Pco = np.log10(Pco.numpy().flatten())
        # scaled_Pco, _, _ = scale_prop(scaled_Pco)

        scaled_g = np.log10(g.numpy().flatten())
        scaled_g, _, _ = scale_prop(scaled_g)

        scaled_top_flux = top_flux.numpy().flatten() / 1e5

        scaled_const = np.log10(gravity.numpy().flatten())

        prop = scaled_top_flux

        print(f'{max(prop) = }')
        print(f'{min(prop) = }')

        plt.figure()
        plt.hist(prop, bins=200)
        plt.show()

        if i == 9:
            break


if __name__ == "__main__":
    main()
