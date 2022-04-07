import os
import sys
from pathlib import Path
import shutil
import numpy as np

import matplotlib
matplotlib.use('Agg')

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import pickle

# own modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = str(Path(script_dir).parents[3])
sys.path.append(src_dir)

from src.neural_nets.dataloaders import SingleVulcanDataset
from src.neural_nets.dataset_utils import make_data_loaders
from src.neural_nets.NN_utils import move_to, plot_variable, derivative_MSE
from src.neural_nets.individualAEs.MRAE.MixingRatioAE import MixingRatioAE

height_values = torch.arange(150)


def loss_fn(device, variable, variable_decoded, diff_weight):
    loss = torch.mean(
        ((variable - variable_decoded) / variable) ** 2
    )

    # soms is de batch size < 32 als er nog maar een paar over zijn
    global height_values
    height_values_batch = torch.tile(height_values[None, ...], dims=(variable.size()[0], 1))
    height_values_batch = height_values_batch.to(device)

    io_pair = (
        height_values_batch,
        variable,
        height_values_batch,
        variable_decoded
    )

    diff_loss = diff_weight * derivative_MSE(*io_pair)

    loss += diff_loss

    return loss, diff_loss


def model_step(device, model, variable, diff_weight):
    # extract inputs

    # output of autoencoder
    variable_decoded = model(variable)

    # Calculating the loss function
    loss, diff_loss = loss_fn(device, variable, variable_decoded, diff_weight)

    return loss, diff_loss


def train_autoencoder(dataset_dir, save_model_dir, log_dir, params):
    # setup pytorch
    device = torch.device(f"cuda:{params['gpu']}" if torch.cuda.is_available() else "cpu")
    print(f'running on device: {device}')

    # Initialize model with double precision
    model = MixingRatioAE(
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

    model_name = f'{params["name"]},{hparams=}'
    summary_file = dt_string + f' | {model_name}'
    writer = SummaryWriter(
        log_dir=os.path.join(log_dir, summary_file)
    )

    # load datasets
    train_loader, test_loader, validation_loader = make_data_loaders(SingleVulcanDataset,
                                                                     os.path.join(dataset_dir, 'scaled_dataset/'),
                                                                     **params['ds_params'])
    # get scaling parameters
    scaling_file = os.path.join(dataset_dir, 'scaling_dict.pkl')
    with open(scaling_file, 'rb') as f:
        scaling_params = pickle.load(f)

    # get species list
    spec_file = os.path.join(dataset_dir, 'species_list.pkl')
    with open(spec_file, 'rb') as f:
        spec_list = pickle.load(f)

    print('created dataloaders:')
    print(f'{len(train_loader) = }')
    print(f'{len(test_loader) = }')
    print(f'{len(validation_loader) = }')

    # extract parameters
    epochs = params['train_params']['epochs']
    writer_interval = params['train_params']['writer_interval']

    # # change height_values array for diff loss
    # global height_values
    # height_values = torch.tile(height_values[None, ...], dims=(params['ds_params']['batch_size'], 1))
    # height_values = height_values.to(device)

    # save best model params
    best_loss = torch.inf
    best_model_params = {}

    for epoch in range(epochs):
        diff_weight = params['loss_params']['LossWeightScheduler_d'].get_weight(epoch)

        # TRAINING
        with tqdm(train_loader, unit='batch', desc=f'Train epoch {epoch}') as train_epoch:
            model.train()

            # keep track of total loss
            tot_loss = 0

            # loop through examples
            for n_iter, example in enumerate(train_epoch):
                variable = move_to(example['inputs'][params['train_params']['variable_key']], device)

                loss, diff_loss = model_step(device, model, variable, diff_weight)

                # update gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tot_loss += loss.detach()

                # update pbar
                # train_epoch.set_postfix(loss=loss.item())

                # visualize steps with Tensorboard
                if n_iter % writer_interval == 0:
                    writer.add_scalar('Batch/loss', loss, n_iter + epoch * len(train_loader))
                    writer.add_scalar('Batch/diff_loss', diff_loss, n_iter + epoch * len(train_loader))

        # visualize epochs with Tensorboard
        avg_train_loss = tot_loss / len(train_loader)
        writer.add_scalar('Epoch loss/train', avg_train_loss, epoch)

        # TESTING
        with tqdm(test_loader, unit='batch', desc=f'Test epoch {epoch}') as test_epoch:
            model.eval()

            # keep track of total losses
            tot_loss = 0
            tot_diff_loss = 0

            # loop through examples
            for n_iter, example in enumerate(test_epoch):
                variable = move_to(example['inputs'][params['train_params']['variable_key']], device)

                loss, diff_loss = model_step(device, model, variable, diff_weight)

                tot_loss += loss.detach()
                tot_diff_loss += diff_loss.detach()

                # update pbar
                # test_epoch.set_postfix(loss=loss.item())

        # show matplotlib graph every 10 epochs
        if epoch % 5 == 0 or epoch == epochs - 1:
            # extract inputs
            variable = move_to(example['inputs'][params['train_params']['variable_key']], device)

            # output of autoencoder
            variable_decoded = model(variable)

            # scales
            scales = scaling_params['inputs'][params['train_params']['variable_key']]

            fig = plot_variable(
                x=move_to(height_values, device=torch.device('cpu')),
                y=move_to(variable, device=torch.device('cpu')),
                y_o=move_to(variable_decoded, device=torch.device('cpu')),
                scales=scales,
                model_name=model_name,
                xlabel='height layer',
                ylabel=params['plot_params']['ylabel'],
                xlog=False,
                ylog=params['plot_params']['ylog']
            )

            writer.add_figure('Plot', fig, epoch)

        # visualize epochs with Tensorboard
        avg_test_loss = tot_loss / len(test_loader)
        writer.add_scalar('Epoch loss/test', avg_test_loss, epoch)

        avg_diff_loss = tot_diff_loss / len(test_loader)
        writer.add_scalar('Epoch diff loss/test', avg_diff_loss, epoch)

        writer.add_scalar('Epoch diff weight', diff_weight, epoch)

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
        tot_diff_loss = 0

        # loop through examples
        for n_iter, example in enumerate(validation):
            variable = move_to(example['inputs'][params['train_params']['variable_key']], device)

            loss, diff_loss = model_step(device, model, variable, diff_weight)

            tot_loss += loss.detach()
            tot_diff_loss += diff_loss.detach()

            # update pbar
            # validation.set_postfix(loss=loss.item())

    # visualize epochs with Tensorboard
    validation_loss = tot_loss / len(validation_loader)
    validation_diff_loss = tot_diff_loss / len(validation_loader)

    metric_dict = {
        "Validation/loss": validation_loss,
        "Validation/diff loss": validation_diff_loss,
    }

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
