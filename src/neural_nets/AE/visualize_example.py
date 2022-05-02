import glob
import os
import sys
from pathlib import Path

import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


# own modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = str(Path(script_dir).parents[1])
sys.path.append(src_dir)

from src.neural_nets.dataset_utils import unscale_inputs_outputs
from src.neural_nets.dataloaders import DoubleVulcanDataset
from src.visualization.plot_vulcan_outputs import tex_labels

# plot_spec = ('H', 'O', 'C', 'N')
# plot_spec = ('H2O', 'CO2', 'CO', 'NH3', 'HCN', 'H2')
plot_spec = ('H2O', 'CO2', 'HCN')
# plot_spec = ('N2H2', 'N2H3', 'C2H3CN')
# colors = ['k', 'y', 'b', 'pink', 'r', 'mediumspringgreen']
colors = [
'#065e60', '#ee9b00', '#0093f5', '#00568f'
]

ref_color = 'b'
rec_color = 'y'
rec_linestyle = (0, (1, 1))
# rec_linestyle_2 = (0, (5, 1))
rec_linestyle_2 = '-'


def plot_single_variable(x, y, y_o, model_name, xlabel, ylabel, xlog=False, ylog=False):
    f, ax = plt.subplots()

    ax.plot(
        x, y,
        c='mediumspringgreen', linestyle='-'
    )

    ax.plot(
        x, y_o,
        c='mediumspringgreen', linestyle=rec_linestyle, alpha=0.5, zorder=-10
    )

    ax.set_xlabel(xlabel)
    if xlog:
        ax.set_xscale('log')

    ax.set_ylabel(ylabel)
    if ylog:
        ax.set_yscale('log')

    handles, labels = ax.get_legend_handles_labels()

    output_line = Line2D([0], [0], linestyle='-', color='k', label='VULCAN output')
    decoded_output_line = Line2D([0], [0], linestyle=rec_linestyle, color='k', label='decoded output')
    handles.extend([output_line, decoded_output_line])

    ax.legend(handles=handles, frameon=0, prop={'size': 10}, loc='best')
    # plt.tight_layout()

    ax.set_title(f'{model_name}')

    return f


def plot_individual_y_mix(y_mix, y_mix_decoded, sp_idx, spec_list, model_name):
    f, ax = plt.subplots()

    height_layers = np.arange(150)

    # set label
    sp = spec_list[sp_idx]

    if sp in tex_labels:
        sp_lab = tex_labels[sp]
    else:
        sp_lab = sp

    ax.plot(
        y_mix, height_layers,
        c='mediumspringgreen', linestyle='-', label=sp_lab)

    ax.plot(
        y_mix_decoded, height_layers,
        c='mediumspringgreen', linestyle=rec_linestyle, alpha=0.5, zorder=-10)

    ax.set_xlabel("Mixing Ratio")
    ax.set_xscale('log')

    # ax.set_yscale('log')
    # ax.invert_yaxis()
    ax.set_ylabel("Height layer")

    handles, labels = ax.get_legend_handles_labels()

    output_line = Line2D([0], [0], linestyle='-', color='k', label='VULCAN output')
    decoded_output_line = Line2D([0], [0], linestyle=rec_linestyle, color='k', label='decoded output')
    handles.extend([output_line, decoded_output_line])

    ax.legend(handles=handles, frameon=0, prop={'size': 10}, loc='best')
    # plt.tight_layout()

    ax.set_title(f'{model_name}\nMixing ratio')

    return f


def plot_y_mix(unscaled_dict, vulcan_spec, g_ax=None, core=False, Pco=True):
    if not g_ax:
        f, ax = plt.subplots()
    else:
        ax = g_ax

    rand_plot_spec = np.random.choice(vulcan_spec, size=3, replace=False)

    for i, sp in enumerate(rand_plot_spec):
        # set label
        if sp in tex_labels:
            sp_lab = tex_labels[sp]
        else:
            sp_lab = sp

        if core:
            # ax.plot(
            #     unscaled_dict['outputs']['y_mix'][:, vulcan_spec.index(sp)],
            #     unscaled_dict['inputs']['Pco'] / 1.e6 if Pco else torch.arange(150),
            #     c=colors[i], linestyle='-', label=sp_lab)

            ax.plot(
                unscaled_dict['decoded_model_outputs']['y_mix_ini'][:, vulcan_spec.index(sp)],
                unscaled_dict['decoded_model_outputs']['Pco'] / 1.e6 if Pco else torch.arange(150),
                c=colors[i], linestyle=rec_linestyle, alpha=1, zorder=-10)

            ax.plot(
                unscaled_dict['decoded_outputs']['y_mix_ini'][:, vulcan_spec.index(sp)],
                unscaled_dict['decoded_outputs']['Pco'] / 1.e6 if Pco else torch.arange(150),
                c=colors[i], linestyle=rec_linestyle_2, alpha=1, zorder=-15, label=sp_lab)
        else:
            ax.plot(
                unscaled_dict['inputs']['y_mix_ini'][:, vulcan_spec.index(sp)],
                unscaled_dict['inputs']['Pco'] / 1.e6 if Pco else torch.arange(150),
                c=colors[i], linestyle='-', label=sp_lab)

            ax.plot(
                unscaled_dict['outputs']['y_mix_ini'][:, vulcan_spec.index(sp)],
                unscaled_dict['outputs']['Pco'] / 1.e6 if Pco else torch.arange(150),
                c=colors[i], linestyle=rec_linestyle, alpha=0.5, zorder=-10)

    ax.set_xlabel("Mixing Ratio")
    ax.set_xscale('log')

    if Pco:
        ax.set_yscale('log')
        ax.invert_yaxis()
        ax.set_ylabel("Pressure (bar)")
    else:
        ax.set_ylabel("Height layer")

    handles, labels = ax.get_legend_handles_labels()

    if core:
        # output_line = Line2D([0], [0], linestyle='-', color='k', label='VULCAN output')
        decoded_output_line = Line2D([0], [0], linestyle=rec_linestyle_2, color='k', label='decoded output')
        decoded_model_output_line = Line2D([0], [0], linestyle=rec_linestyle, color='k', label='decoded core output')
        handles.extend([
            # output_line,
            decoded_output_line, decoded_model_output_line])
    else:
        output_line = Line2D([0], [0], linestyle='-', color='k', label='VULCAN output')
        decoded_output_line = Line2D([0], [0], linestyle=rec_linestyle, color='k', label='decoded output')
        handles.extend([output_line, decoded_output_line])

    ax.legend(handles=handles, frameon=0, prop={'size': 10}, loc='best')
    # plt.tight_layout()

    ax.set_title('Mixing ratios')

    if not g_ax:
        plt.show()


def plot_TP(unscaled_dict, g_ax=None):
    if not g_ax:
        f, ax = plt.subplots()
    else:
        ax = g_ax

    ax.plot(unscaled_dict['inputs']['Tco'], unscaled_dict['inputs']['Pco'] / 1e6,
            linestyle='-', color=ref_color)

    ax.plot(unscaled_dict['outputs']['Tco'], unscaled_dict['outputs']['Pco'] / 1e6,
            linestyle=rec_linestyle, color=rec_color)

    ax.set_yscale('log')
    ax.invert_yaxis()

    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel("Pressure (bar)")

    ax.set_title('TP-profile')

    if not g_ax:
        plt.show()


def plot_spectrum(unscaled_dict, g_ax=None):
    if not g_ax:
        f, ax = plt.subplots()
    else:
        ax = g_ax

    ax.plot(unscaled_dict['inputs']['wavelengths'], unscaled_dict['inputs']['top_flux'], linestyle='-', color=ref_color)
    ax.plot(unscaled_dict['outputs']['wavelengths'], unscaled_dict['outputs']['top_flux'],
            linestyle=rec_linestyle, color=rec_color)

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Flux (erg / (nm cm2 s))')

    ax.set_yscale('log')
    plt.xscale('log')

    ax.set_title('Top spectrum')

    if not g_ax:
        plt.show()


def plot_g(unscaled_dict, g_ax=None):
    if not g_ax:
        f, ax = plt.subplots()
    else:
        ax = g_ax

    ax.plot(unscaled_dict['inputs']['g'], unscaled_dict['inputs']['Pco'] / 1e6,
            linestyle='-', color=ref_color)

    ax.plot(unscaled_dict['outputs']['g'], unscaled_dict['outputs']['Pco'] / 1e6,
            linestyle=rec_linestyle, color=rec_color)

    ax.set_yscale('log')
    ax.invert_yaxis()

    ax.set_xlabel('Gravity (g cm / s2)')
    ax.set_ylabel("Pressure (bar)")

    ax.set_title('g')

    if not g_ax:
        plt.show()


def plot_all(unscaled_dict, vulcan_spec, model_name, show=False, save=False, example_num=0):
    # PLOTTING
    fig = plt.figure(constrained_layout=True, figsize=(10, 7))
    spec = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)

    plot_y_mix(unscaled_dict, vulcan_spec, fig.add_subplot(spec[:, 0:2]))
    plot_TP(unscaled_dict, fig.add_subplot(spec[0, 2]))
    plot_g(unscaled_dict, fig.add_subplot(spec[0, 3]))
    plot_spectrum(unscaled_dict, fig.add_subplot(spec[1, 2:]))

    plt.suptitle(model_name, fontsize=17)
    if save:
        plt.savefig(f'{example_num}.png', dpi=600)
    if show:
        plt.show()

    return fig


def plot_y_mix_core(unscaled_dict, vulcan_spec, model_name, show=False, save=False, Pco=True, example_num=0):
    # PLOTTING
    fig, ax = plt.subplots(figsize=(10, 7))

    plot_y_mix(unscaled_dict, vulcan_spec, ax, core=True, Pco=Pco)

    plt.suptitle(model_name)
    if save:
        plt.savefig(f'{example_num}.png', dpi=600)
    if show:
        plt.show()

    return fig


def main():
    # setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MRP_dir = str(Path(script_dir).parents[2])
    dataset_dir = os.path.join(MRP_dir, 'data/bday_dataset/dataset')
    save_model_dir = os.path.join(script_dir, '../saved_models')
    # vulcan_output_dir = os.path.join(MRP_dir, 'data/christmas_dataset/vulcan_output')

    # from autoencoder2 import AutoEncoder
    # model_name = "AE2,hparams={'latent_dim': 2048, 'lr': 0.0001}_state_dict"

    from autoencoder_large_ls import AutoEncoder
    model_name = "AE2_large,hparams={'latent_dim': 4096, 'lr': 0.0001}_state_dict"

    # from src.neural_nets.VAE.VAE_large import VariationalAutoEncoder as AutoEncoder
    # model_name = "VAE_large,hparams={'latent_dim': 4096, 'lr': 0.0001}_state_dict"

    # dataset loader
    vulcan_dataset = DoubleVulcanDataset(os.path.join(dataset_dir, 'interpolated_dataset'))

    dataloader = DataLoader(vulcan_dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=0)

    device = torch.device("cpu")
    print(f'running on device: {device}')

    # # load model
    # model = AutoEncoder(
    #     # device=device,
    #     latent_dim=4096
    # ).double()

    # model.load_state_dict(torch.load(os.path.join(save_model_dir, f'{model_name}')))
    #
    # model.eval()
    example_num = 0
    for example in dataloader:
        inputs = example['inputs']

        # output of autoencoder
        # outputs = model(inputs)

        # get scaling parameters
        scaling_file = os.path.join(dataset_dir, 'scaling_dict.pkl')
        with open(scaling_file, 'rb') as f:
            scaling_params = pickle.load(f)

        # get species list
        scaling_file = os.path.join(dataset_dir, 'species_list.pkl')
        with open(scaling_file, 'rb') as f:
            spec_list = pickle.load(f)

        unscaled_dict = unscale_inputs_outputs(inputs, inputs, scaling_params)

        # PLOTTING
        fig = plot_all(unscaled_dict, spec_list, model_name, show=True, save=True, example_num=example_num)

        example_num += 1
        if example_num > 4:
            break


if __name__ == "__main__":
    main()
