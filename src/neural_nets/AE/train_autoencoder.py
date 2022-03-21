import os
import sys
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# own modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = str(Path(script_dir).parents[2])
sys.path.append(src_dir)

from src.neural_nets.dataset_utils import make_data_loaders
from src.neural_nets.NN_utils import multiple_MSELoss_dict, move_to, double_derivative
from autoencoder2 import AutoEncoder


def loss_fn(device, inputs, outputs, loss_weights):
    # Calculating the MSE loss
    mse_loss, mse_loss_arr = multiple_MSELoss_dict(
        inputs=inputs,
        outputs=outputs,
        weights=loss_weights,
        device=device
    )

    # penalty for jittery behaviour
    sum_double_diff = double_derivative(
        x=outputs['Pco'][:, None, :].tile(1, 69, 1),
        y=outputs['y_mix_ini'].permute(0, 2, 1),
    ).sum()

    # TODO: only add after initial training?
    loss = mse_loss + sum_double_diff

    return loss, mse_loss_arr


def model_step(device, model, example, loss_weights=None):
    # extract inputs
    inputs = move_to(example['inputs'], device)

    # output of autoencoder
    outputs = model(inputs)

    # Calculating the loss function
    loss, loss_arr = loss_fn(device, inputs, outputs, loss_weights)

    return loss, loss_arr


def train_autoencoder(dataset_dir, save_model_dir, log_dir, params):
    # setup pytorch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'running on device: {device}')

    # move params to gpu/cpu
    params['loss_params'] = move_to(params['loss_params'], device)

    # Initialize model with double precision
    model = AutoEncoder(
        **params['model_params']
    ).double().to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), **params['optimizer_params'])

    # Tensorboard logging
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    hparams = {}
    hparams.update(params['model_params'])
    hparams.update(params['optimizer_params'])
    # hparams.update(params['loss_params'])

    model_name = f'{params["name"]},{hparams=}'
    summary_file = dt_string + f' | {model_name}'
    writer = SummaryWriter(
        log_dir=os.path.join(log_dir, summary_file)
    )

    # load datasets
    train_loader, test_loader, validation_loader = make_data_loaders(dataset_dir, **params['ds_params'])

    print('created dataloaders:')
    print(f'{len(train_loader) = }')
    print(f'{len(test_loader) = }')
    print(f'{len(validation_loader) = }')

    # extract parameters
    epochs = params['train_params']['epochs']
    writer_interval = params['train_params']['writer_interval']
    num_elements_in_example = params['train_params']['num_elements_in_example']

    # save best model params
    best_loss = torch.inf
    best_model_params = {}

    for epoch in range(epochs):
        # TRAINING
        with tqdm(train_loader, unit='batch', desc=f'Train epoch {epoch}') as train_epoch:
            model.train()

            # keep track of total loss
            tot_loss = 0

            for n_iter, example in enumerate(train_epoch):
                loss, loss_arr = model_step(device, model, example, **params['loss_params'])

                # update gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tot_loss += loss.detach()

                # update pbar
                train_epoch.set_postfix(loss=loss.item())

                # visualize steps with Tensorboard
                if n_iter % writer_interval == 0:
                    writer.add_scalar('Batch/loss', loss, n_iter + epoch * len(train_loader))

        # visualize epochs with Tensorboard
        avg_train_loss = tot_loss / len(train_loader)
        writer.add_scalar('Epoch loss/train', avg_train_loss, epoch)

        # TESTING
        with tqdm(test_loader, unit='batch', desc=f'Test epoch {epoch}') as test_epoch:
            model.eval()

            # keep track of total losses
            tot_loss = 0
            tot_ind_losses = torch.zeros(num_elements_in_example, device=device)

            for n_iter, example in enumerate(test_epoch):
                loss, loss_arr = model_step(device, model, example, **params['loss_params'])

                tot_loss += loss.detach()

                # update pbar
                test_epoch.set_postfix(loss=loss.item())

                # visualize steps with Tensorboard
                if n_iter % writer_interval == 0:
                    for i, el_loss in enumerate(loss_arr):
                        tot_ind_losses[i] += el_loss

        # visualize epochs with Tensorboard
        avg_test_loss = tot_loss / len(test_loader)
        writer.add_scalar('Epoch loss/test', avg_test_loss, epoch)

        avg_test_ind_losses = tot_ind_losses / len(test_loader)
        for i, avg_ind_loss in enumerate(avg_test_ind_losses):
            writer.add_scalar(f'Epoch individual loss/test/{i}', avg_ind_loss, epoch)

        # save best model params
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_model_params = model.state_dict().copy()

    # load best model params
    model.load_state_dict(best_model_params)

    # VALIDATION
    with tqdm(validation_loader, unit='batch', desc='Validation') as validation:
        model.eval()

        # keep track of total losses
        tot_loss = 0
        tot_ind_losses = torch.zeros(num_elements_in_example, device=device)

        for n_iter, example in enumerate(validation):
            loss, loss_arr = model_step(device, model, example, **params['loss_params'])

            tot_loss += loss.detach()

            # update pbar
            validation.set_postfix(loss=loss.item())

            # visualize steps with Tensorboard
            if n_iter % writer_interval == 0:
                # writer.add_scalar('Epoch loss/validation', loss, n_iter)

                for i, el_loss in enumerate(loss_arr):
                    # writer.add_scalar(f'Batch individual loss/validation/{i}', el_loss, n_iter)
                    tot_ind_losses[i] += el_loss

    # visualize epochs with Tensorboard
    validation_loss = tot_loss / len(validation_loader)
    validation_ind_losses = tot_ind_losses / len(validation_loader)

    metric_dict = {"Validation/loss": validation_loss}

    for i, avg_ind_loss in enumerate(validation_ind_losses):
        metric_dict.update(
            {f'Validation/individual loss {i}': avg_ind_loss}
        )

    # add hyperparameters
    writer.add_hparams(
        hparams,
        metric_dict
    )

    # make sure to write everything
    writer.flush()

    # close Tensorboard
    writer.close()

    # save the model
    torch.save(best_model_params, os.path.join(save_model_dir, f'{model_name}_state_dict'))


def main():
    # setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MRP_dir = str(Path(script_dir).parents[2])
    dataset_dir = os.path.join(MRP_dir, 'data/christmas_dataset/wavelength_dataset')
    save_model_dir = os.path.join(script_dir, '../saved_models')
    log_dir = os.path.join(script_dir, '../runs')

    # remake the config directory
    if not os.path.isdir(save_model_dir):
        os.mkdir(save_model_dir)

    params = dict(
        name='AE2',

        ds_params={
            'batch_size': 4,
            'double_mode': True,
            'shuffle': True,
            'num_workers': 4,
            'train_test_validation_ratios': [0.7, 0.2, 0.1]
        },

        model_params={
            'latent_dim': 2048
        },

        optimizer_params={
            'lr': 1e-4
        },

        loss_params={
            'loss_weights': {
                'y_mix_ini': 1.,
                'top_flux': 1.,
                'wavelengths': 1.,
                'Tco': 1.,
                'Pco': 1.,
                'g': 1.,
                'gravity': 1.
            }
        },

        train_params={
            'epochs': 200,
            'writer_interval': 10,
            'num_elements_in_example': 7
        }
    )

    train_autoencoder(dataset_dir, save_model_dir, log_dir, params)


if __name__ == "__main__":
    main()
