import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# own modules
from dataset_utils import VulcanDataset
from autoencoder import AutoEncoder


def multiple_MSELoss(device, inputs, outputs, weights=None):
    """
    Calculates the weighted mean of the MSE losses of the inputs and outputs.

    Args:
        device: str, pytorch device (gpu or cpu)
        inputs: torch.Tensor, inputs
        outputs: torch.Tensor, outputs
        weights: torch.Tensor, weights

    Returns:
        loss: torch.Tensor, the weighted mean loss
        loss_arr: torch.Tensor, the individual MSE losses
    """

    if weights is not None:
        normalized_weights = weights / torch.sum(weights)
    else:
        normalized_weights = torch.zeros(len(inputs), device=device)

    loss = torch.zeros(len(inputs), device=device)
    loss_arr = torch.zeros(len(inputs), device=device)

    for i, (input, output, weight) in enumerate(zip(inputs, outputs, normalized_weights)):
        i_loss = torch.mean((output - input) ** 2)
        loss[i] = i_loss * weight
        loss_arr[i] = i_loss

    return torch.mean(loss_arr), loss_arr


def model_step(device, model, example, loss_weights=None):
    # extract inputs
    y_mix_ini = example['inputs']['y_mix_ini'].to(device)
    top_flux = example['inputs']['top_flux'].to(device)
    Tco = example['inputs']['Tco'].to(device)
    Pco = example['inputs']['Pco'].to(device)
    g = example['inputs']['g'].to(device)
    gravity = example['inputs']['gravity'].to(device)

    # output of autoencoder
    decoded_y_mix_ini, decoded_top_flux, decoded_Tco, decoded_Pco, decoded_g, decoded_gravity = \
        model(y_mix_ini, top_flux, Tco, Pco, g, gravity)

    # Calculating the loss function
    loss, loss_arr = multiple_MSELoss(
        inputs=(
            y_mix_ini, top_flux, Tco, Pco, g, gravity
        ),
        outputs=(
            decoded_y_mix_ini, decoded_top_flux, decoded_Tco, decoded_Pco, decoded_g, decoded_gravity
        ),
        weights=loss_weights,
        device=device
    )

    return loss, loss_arr


def train_autoencoder(dataset_dir):
    # TODO: import parameters in dictionary, maybe nested dict for model params?
    lr = 1e-3
    batch_size = 4
    epochs = 50
    shuffle = True
    num_workers = 0
    train_test_validation_ratios = [0.7, 0.2, 0.1]
    loss_weights = [3., 1., 1., 1., 1., 1.]    # y_mix_ini, top_flux, Tco, Pco, g, gravity

    # setup pytorch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'running on device: {device}')

    # dataset loader
    vulcan_dataset = VulcanDataset(dataset_dir)

    # split like this to make sure len(subsets) = len(dataset)
    train_size = int(train_test_validation_ratios[0] * len(vulcan_dataset))
    test_size = int(train_test_validation_ratios[1] * len(vulcan_dataset))
    validation_size = len(vulcan_dataset) - train_size - test_size

    train_dataset, test_dataset, validation_dataset = torch.utils.data.random_split(vulcan_dataset,
                                                                                    [train_size, test_size,
                                                                                     validation_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                                   shuffle=shuffle,
                                   num_workers=num_workers)

    # Initialize model with double precision
    model = AutoEncoder().double().to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Make loss weights
    loss_weights = torch.Tensor(loss_weights).to(device)

    # Tensorboard logging
    writer = SummaryWriter(comment=f', {lr = }, {batch_size = }, {shuffle = }')

    for epoch in range(epochs):
        # TRAINING
        with tqdm(train_loader, unit='batch', desc=f'Train epoch {epoch}') as train_epoch:
            model.train()
            tot_loss = 0
            for n_iter, example in enumerate(train_epoch):
                loss, loss_arr = model_step(device, model, example, loss_weights)

                # The gradients are set to zero,
                # the the gradient is computed and stored.
                # .step() performs parameter update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tot_loss += loss

                # update pbar
                train_epoch.set_postfix(loss=loss.item())

                # visualize steps with Tensorboard
                if n_iter % 10 == 0:
                    writer.add_scalar('Loss/train', loss, n_iter + epoch * len(train_loader))

        # visualize epochs with Tensorboard
        train_loss = tot_loss / len(train_loader)
        writer.add_scalar('Epoch loss/train', train_loss, epoch)

        # TESTING
        with tqdm(test_loader, unit='batch', desc=f'Test epoch {epoch}') as test_epoch:
            model.eval()
            tot_loss = 0
            for n_iter, example in enumerate(test_epoch):
                loss, loss_arr = model_step(device, model, example, loss_weights)

                tot_loss += loss

                # update pbar
                test_epoch.set_postfix(loss=loss.item())

                # visualize steps with Tensorboard
                if n_iter % 10 == 0:
                    writer.add_scalar('Loss/test', loss, n_iter + epoch * len(test_loader))

                    for i, el_loss in enumerate(loss_arr):
                        writer.add_scalar(f'Individual losses/{i}', el_loss, n_iter + epoch * len(test_loader))

        # visualize epochs with Tensorboard
        test_loss = tot_loss / len(test_loader)
        writer.add_scalar('Epoch loss/test', test_loss, epoch)

    # VALIDATION
    with tqdm(validation_loader, unit='batch', desc='Validation') as validation:
        model.eval()
        tot_loss = 0
        for n_iter, example in enumerate(validation):
            loss, loss_arr = model_step(device, model, example, loss_weights)

            tot_loss += loss

            # update pbar
            validation.set_postfix(loss=loss.item())

            # visualize steps with Tensorboard
            if n_iter % 10 == 0:
                writer.add_scalar('Loss/validation', loss, n_iter)

    # visualize epochs with Tensorboard
    validation_loss = tot_loss / len(validation_loader)

    # add hyperparameters
    writer.add_hparams(
        {"lr": lr,
         "batch_size": batch_size,
         "shuffle": shuffle},
        {"Validation loss": validation_loss}
    )

    # make sure to write everything
    writer.flush()

    # close Tensorboard
    writer.close()


def main():
    # setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MRP_dir = str(Path(script_dir).parents[1])
    dataset_dir = os.path.join(MRP_dir, 'data/christmas_dataset/dataset')

    train_autoencoder(dataset_dir)


if __name__ == "__main__":
    main()
