import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# own modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = str(Path(script_dir).parents[1])
sys.path.append(src_dir)

from src.neural_nets.dataset_utils import SingleVulcanDataset, unscale_example


def scale_prop(prop):
    prop_mean = np.mean(prop)
    prop_std = np.std(prop)
    scaled_prop = (prop - prop_mean) / prop_std

    return scaled_prop, prop_mean, prop_std


def test_scales(dataset_dir):
    # dataset loader
    vulcan_dataset = SingleVulcanDataset(dataset_dir)
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
        wavelengths = example['inpuits']['wavelength']
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


def save_dataset_distributions(dataset_dir, mode='mean'):
    # dataset loader
    vulcan_dataset = 0(dataset_dir)
    dataloader = DataLoader(vulcan_dataset, batch_size=1,
                            shuffle=True,
                            num_workers=0)
    # get scaling parameters
    scaling_file = os.path.join(dataset_dir, 'scaling_dict.pkl')
    with open(scaling_file, 'rb') as f:
        scaling_params = pickle.load(f)

    # create tot dict
    for i, dummy_example in enumerate(dataloader):
        unscaled_dummy_example = unscale_example(dummy_example, scaling_params)
        tot_dict = unscaled_dummy_example.copy()
        for top_key, top_value in tot_dict.items():
            for key, value in top_value.items():
                zero_value = np.zeros_like(value)
                tot_dict[top_key][key] = np.tile(zero_value[..., None], len(dataloader))
        break

    # loop through examples
    with tqdm(dataloader, unit='example', desc=f'Summing values') as dataloader:
        for i, example in enumerate(dataloader):
            # unscale dict
            unscaled_dict = unscale_example(example, scaling_params)

            # add
            for top_key, top_value in tot_dict.items():
                for key, value in top_value.items():
                    tot_dict[top_key][key][..., i] = unscaled_dict[top_key][key]

    # calculate means
    agg_dict = tot_dict.copy()
    for top_key, top_value in tot_dict.items():
        for key, value in top_value.items():
            if mode == 'mean':
                agg_dict[top_key][key] = np.mean(value, axis=-1)
            elif mode == 'median':
                agg_dict[top_key][key] = np.median(value, axis=-1)

    # save dict
    agg_dict_file = f'agg_dicts/{mode}_dict.pkl'
    with open(agg_dict_file, 'wb') as f:
        pickle.dump(agg_dict, f)


def plot_dataset_distributions(dataset_dir, mode='mean'):
    # extra imports
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.gridspec as gridspec

    # get agg dict
    agg_dict_file = f'agg_dicts/{mode}_dict.pkl'
    with open(agg_dict_file, 'rb') as f:
        agg_dict = pickle.load(f)

    # get species list
    scaling_file = os.path.join(dataset_dir, 'species_list.pkl')
    with open(scaling_file, 'rb') as f:
        spec_list = np.array(pickle.load(f))

    tex_labels = {'H': 'H', 'H2': 'H$_2$', 'O': 'O', 'OH': 'OH', 'H2O': 'H$_2$O', 'CH': 'CH', 'C': 'C', 'CH2': 'CH$_2$',
                  'CH3': 'CH$_3$', 'CH4': 'CH$_4$', 'HCO': 'HCO', 'H2CO': 'H$_2$CO', 'C4H2': 'C$_4$H$_2$',
                  'C2': 'C$_2$', 'C2H2': 'C$_2$H$_2$', 'C2H3': 'C$_2$H$_3$', 'C2H': 'C$_2$H', 'CO': 'CO',
                  'CO2': 'CO$_2$', 'He': 'He', 'O2': 'O$_2$', 'CH3OH': 'CH$_3$OH', 'C2H4': 'C$_2$H$_4$',
                  'C2H5': 'C$_2$H$_5$', 'C2H6': 'C$_2$H$_6$', 'CH3O': 'CH$_3$O', 'CH2OH': 'CH$_2$OH', 'N2': 'N$_2$',
                  'NH3': 'NH$_3$', 'HCN': 'HCN', 'NO': 'NO', 'NO2': 'NO$_2$'}

    # spec_list = [tex_labels[sp] if sp in tex_labels else sp for sp in spec_list]

    # get y_ini_mix
    y_mix_ini = agg_dict['inputs']['y_mix_ini'].swapaxes(0, 1)
    if mode == 'mean':
        y_mix_ini_mean_height = np.mean(y_mix_ini, axis=1)
    if mode == 'median':
        y_mix_ini_mean_height = np.median(y_mix_ini, axis=1)

    y_inds = y_mix_ini_mean_height.argsort()
    y_mix_ini = y_mix_ini[y_inds[::-1], :]
    y_mix_ini_mean_height = y_mix_ini_mean_height[y_inds[::-1]]
    spec_list = spec_list[y_inds[::-1]]

    fig = plt.figure(constrained_layout=True, figsize=(10, 12))
    spec = gridspec.GridSpec(ncols=10, nrows=20, figure=fig)

    # plot 2d

    ax1 = fig.add_subplot(spec[1:, 3:9])
    y_mix_ini = np.log10(y_mix_ini)
    print(f'{y_mix_ini.min() = }')
    print(f'{y_mix_ini.max() = }')

    # cmap
    cmap = 'RdYlBu_r'
    cmap_func = plt.get_cmap(cmap)

    # extract all colors from the map
    cmaplist = [cmap_func(i) for i in range(cmap_func.N)]

    # create the new map
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap_func.N)

    # define the bins and normalize
    bounds = np.linspace(np.min(y_mix_ini), np.max(y_mix_ini), 21)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    im = ax1.imshow(y_mix_ini, aspect='auto', interpolation='nearest',
                    cmap=cmap, norm=norm
                    )

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    cbar_ticks = np.arange(-60, 0, 5)
    # cbar_ticks = [b - 0.5*np.diff(bounds)[0] for b in bounds]
    cbar = plt.colorbar(im, cax=cax, label='mixing ratio',
                        # ticks=cbar_ticks,
                        )
    # cbar.ax.set_yticklabels(
    #     # [f'$10^{{{t:.3}}}$' for t in cbar_ticks[:-1]]
    #     [f'$10^{{{t}}}$' for t in cbar_ticks]
    # )

    im_xticks = np.arange(0, 175, 25)
    ax1.set_xticks(im_xticks)
    ax1.set_xlabel('height layer')

    # ax1.set_ylabel('species')
    ax1.set_yticks(np.arange(len(spec_list)))
    ax1.set_yticklabels(spec_list)

    ax1.set_title('as a function of height')
    # fig.suptitle('mean initial mixing ratio\'s', fontsize=18, y=0.99)

    # plt.tight_layout()
    # plt.show()

    # plot 1d
    # fig, ax = plt.subplots(figsize=(6, 12))

    ax2 = fig.add_subplot(spec[1:, :3])

    # y_mix_ini_mean_height = np.log10(y_mix_ini_mean_height)
    print(f'{y_mix_ini_mean_height.min() = }')
    print(f'{y_mix_ini_mean_height.max() = }')

    cmap_func = plt.get_cmap(cmap)
    rescale_color = lambda y: (y - np.min(y_mix_ini)) / (np.max(y_mix_ini) - np.min(y_mix_ini))

    ax2.barh(np.arange(len(spec_list)),
             y_mix_ini_mean_height[::-1],
             color=cmap_func(rescale_color(np.log10(y_mix_ini_mean_height[::-1]))),
             )

    ax2.set_xlabel('mixing ratio')
    ax2.set_xscale('log')
    ax2.invert_xaxis()

    ax2.set_ylabel('species')
    ax2.set_yticks(np.arange(len(spec_list)))
    ax2.set_yticklabels(spec_list[::-1])
    ax2.set_ylim(-0.5, len(spec_list) - 0.5)

    ax2.set_title(f'{mode} of height')
    fig.suptitle(f'{mode} initial mixing ratio\'s', fontsize=18, y=0.99)

    # plt.tight_layout()
    # plt.savefig('y_mix_dist.png', dpi=300)
    plt.show()


def main():
    # setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MRP_dir = str(Path(script_dir).parents[1])
    dataset_dir = os.path.join(MRP_dir, 'data/bday_dataset/dataset')
    # dataset_dir = os.path.join(MRP_dir, 'data/christmas_dataset/wavelength_dataset')
    # dataset_dir = os.path.join(MRP_dir, 'data/christmas_dataset/cut_dataset')
    # dataset_dir = os.path.join(MRP_dir, 'data/christmas_dataset/clipped_dataset')

    # test_scales(dataset_dir)
    mode = 'median'
    save_dataset_distributions(dataset_dir, mode=mode)
    plot_dataset_distributions(dataset_dir, mode=mode)


if __name__ == "__main__":
    main()
