import os
import sys
from pathlib import Path
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

from src.neural_nets.dataloaders import MixingRatioVulcanDataset
from src.neural_nets.dataset_utils import make_data_loaders
from src.neural_nets.NN_utils import move_to, plot_single_y_mix, derivative_MSE, LossWeightScheduler, plot_variable
from src.neural_nets.individualAEs.MRAE.MixingRatioAE import MixingRatioAE

height_values = torch.arange(150)


def loss_fn(device, y_mix, y_mix_decoded):
    loss = torch.mean(
        ((y_mix - y_mix_decoded) / y_mix) ** 2
    )

    return loss


def model_step(device, model, spec_example):
    # extract inputs
    y_mix = move_to(spec_example['species_mr'], device)

    # output of autoencoder
    y_mix_decoded = model(y_mix)

    return y_mix, y_mix_decoded


def train_autoencoder(dataset_dir, save_model_dir, log_dir, params):
    # headless plotting
    import matplotlib
    matplotlib.use('Agg')

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
    train_loader, test_loader, validation_loader = make_data_loaders(MixingRatioVulcanDataset,
                                                                     os.path.join(dataset_dir, 'interpolated_dataset/'),
                                                                     **params['ds_params'])
    # save validation indices
    torch.save(validation_loader.dataset.indices, os.path.join(save_model_dir, f'{model_name}_validation_indices.pt'))

    # get scaling parameters
    scaling_file = os.path.join(dataset_dir, 'scaling_dict.pkl')
    with open(scaling_file, 'rb') as f:
        scaling_params = pickle.load(f)
        print(f'{scaling_params = }')

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
        # TRAINING
        with tqdm(train_loader, unit='batch', desc=f'Train epoch {epoch}') as train_epoch:
            model.train()

            # keep track of total loss
            tot_loss = 0

            # loop through examples
            for n_iter, spec_example in enumerate(train_epoch):
                y_mix, y_mix_decoded = model_step(device, model, spec_example)
                loss = loss_fn(device, y_mix, y_mix_decoded)

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

        # visualize epochs with Tensorboard
        avg_train_loss = tot_loss / len(train_loader)
        writer.add_scalar('Epoch loss/train', avg_train_loss, epoch)

        # TESTING
        with tqdm(test_loader, unit='batch', desc=f'Test epoch {epoch}') as test_epoch:
            model.eval()

            # keep track of total losses
            tot_loss = 0

            # loop through examples
            for n_iter, spec_example in enumerate(test_epoch):
                y_mix, y_mix_decoded = model_step(device, model, spec_example)
                loss = loss_fn(device, y_mix, y_mix_decoded)

                tot_loss += loss.detach()

                # update pbar
                # test_epoch.set_postfix(loss=loss.item())

        # show matplotlib graph every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            # extract inputs
            y_mix = move_to(spec_example['species_mr'], device)

            sp_idx = spec_example['sp_idx'][0]

            # output of autoencoder
            y_mix_decoded = model(y_mix)

            # scales
            scales = scaling_params['inputs']['y_mix_ini']

            fig = plot_variable(
                x=move_to(height_values, device=torch.device('cpu')),
                y=move_to(y_mix, device=torch.device('cpu')),
                y_o=move_to(y_mix_decoded, device=torch.device('cpu')),
                scales=scales,
                model_name=model_name + '\n' + spec_list[sp_idx],
                xlabel='height layer',
                ylabel='Mixing ratio',
                xlog=False,
                ylog=True
            )

            writer.add_figure('Plot', fig, epoch)

        # visualize epochs with Tensorboard
        avg_test_loss = tot_loss / len(test_loader)
        writer.add_scalar('Epoch loss/test', avg_test_loss, epoch)

        # save best model params
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_model_params = model.state_dict().copy()

            # save the model
            torch.save(best_model_params, os.path.join(save_model_dir, f'{model_name}_state_dict'))

    # load best model params
    model.load_state_dict(best_model_params)

    # VALIDATION
    with tqdm(validation_loader, unit='batch', desc='Validation') as validation:
        model.eval()

        # keep track of total losses
        tot_loss = 0

        # loop through examples
        for n_iter, spec_example in enumerate(validation):
            y_mix, y_mix_decoded = model_step(device, model, spec_example)
            loss = loss_fn(device, y_mix, y_mix_decoded)

            tot_loss += loss.detach()

            # update pbar
            # validation.set_postfix(loss=loss.item())

    # visualize epochs with Tensorboard
    validation_loss = tot_loss / len(validation_loader)

    metric_dict = {
        "Validation/loss": validation_loss,
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


def main():
    # setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MRP_dir = str(Path(script_dir).parents[3])    # TODO: same as src_dir?
    dataset_dir = os.path.join(MRP_dir, 'data/bday_dataset/dataset')
    save_model_dir = os.path.join(MRP_dir, 'src/neural_nets/saved_models_final')
    log_dir = os.path.join(MRP_dir, 'src/neural_nets/runs_final')

    # make save directory if not present
    if not os.path.isdir(save_model_dir):
        os.mkdir(save_model_dir)

    params = dict(
        name='MRAE',

        gpu=0,

        ds_params={
            'batch_size': 32,
            'shuffle': True,
            'num_workers': 4,
            'train_test_validation_ratios': [0.7, 0.2, 0.1]
        },

        model_params={
            'latent_dim': 30,
            'layer_size': 256,
            'activation_function': 'tanh',
        },

        optimizer_params={
            'lr': 1e-5
        },

        loss_params={
            'LossWeightScheduler_d': LossWeightScheduler(
                start_epoch=0,
                end_epoch=1,
                start_weight=0,
                end_weight=0
            ),
        },

        train_params={
            'epochs': 200,
            'writer_interval': 10,
        }
    )

    train_autoencoder(dataset_dir, save_model_dir, log_dir, params)


if __name__ == "__main__":
    main()
