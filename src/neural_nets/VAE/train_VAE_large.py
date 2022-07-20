import os
import sys
from pathlib import Path

from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import pickle

# own modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = str(Path(script_dir).parents[2])
sys.path.append(src_dir)

from src.neural_nets.dataset_utils import make_data_loaders
from src.neural_nets.NN_utils import multiple_MSELoss_dict, move_to, derivative_MSE, double_derivative_MSE, plot_vars, \
    LossWeightScheduler
# from VAE_large import VariationalAutoEncoder
from VAE_large_cut import VariationalAutoEncoder

num_species = 45


def loss_fn(device, inputs, outputs, diff_weight, ddiff_weight, loss_weights):
    # Calculating the MSE loss
    loss, loss_arr = multiple_MSELoss_dict(
        inputs=inputs,
        outputs=outputs,
        weights=loss_weights,
        device=device
    )

    # input-outputs pairs
    io_pairs = [
        (  # y_mix_ini
            inputs['Pco'][:, None, :].tile(1, num_species, 1),
            inputs['y_mix_ini'].permute(0, 2, 1),
            outputs['Pco'][:, None, :].tile(1, num_species, 1),
            outputs['y_mix_ini'].permute(0, 2, 1)
        ),
        (  # TP
            inputs['Pco'],
            inputs['Tco'],
            outputs['Pco'],
            outputs['Tco']
        ),
        (  # g
            inputs['Pco'],
            inputs['g'],
            outputs['Pco'],
            outputs['g']
        ),
        (  # flux
            inputs['wavelengths'],
            inputs['top_flux'],
            outputs['wavelengths'],
            outputs['top_flux']
        )
    ]

    # first derivatives
    diff_loss = 0
    for io_pair in io_pairs:
        diff_loss += diff_weight * derivative_MSE(*io_pair)

    # second derivatives
    ddiff_loss = 0
    for io_pair in io_pairs:
        ddiff_loss += ddiff_weight * double_derivative_MSE(*io_pair)

    loss += diff_loss + ddiff_loss

    return loss, loss_arr, diff_loss, ddiff_loss


def model_step(device, model, example, diff_weight, ddiff_weight, loss_weights):
    # extract inputs
    inputs = move_to(example['inputs'], device)

    # output of autoencoder
    outputs, metrics = model(inputs)

    # Calculating the loss function
    mse_loss, loss_arr, diff_loss, ddiff_loss = loss_fn(device, inputs, outputs, diff_weight, ddiff_weight,
                                                        loss_weights)
    kl_div = metrics['kl_div']
    loss = mse_loss + kl_div

    return loss, loss_arr, kl_div, diff_loss, ddiff_loss


def train_VAE(dataset_dir, save_model_dir, log_dir, params):
    # setup pytorch
    device = torch.device(f"cuda:{params['gpu']}" if torch.cuda.is_available() else "cpu")
    print(f'running on device: {device}')

    # move params to gpu/cpu
    params['loss_params']['loss_weights'] = move_to(params['loss_params']['loss_weights'], device)

    # Initialize model with double precision
    model = VariationalAutoEncoder(
        device=device,
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

    # get scaling parameters
    scaling_file = os.path.join(dataset_dir, 'scaling_dict.pkl')
    with open(scaling_file, 'rb') as f:
        scaling_params = pickle.load(f)

    # get species list
    scaling_file = os.path.join(dataset_dir, 'species_list.pkl')
    with open(scaling_file, 'rb') as f:
        spec_list = pickle.load(f)

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

        diff_weight = params['loss_params']['LossWeightScheduler_d'].get_weight(epoch)
        ddiff_weight = params['loss_params']['LossWeightScheduler_dd'].get_weight(epoch)

        # TRAINING
        with tqdm(train_loader, unit='batch', desc=f'Train epoch {epoch}') as train_epoch:
            model.train()

            # keep track of total loss
            tot_loss = 0
            tot_kl_div = 0
            for n_iter, example in enumerate(train_epoch):
                loss, loss_arr, kl_div, diff_loss, ddiff_loss = model_step(device, model, example,
                                                                           loss_weights=params['loss_params'][
                                                                               'loss_weights'],
                                                                           diff_weight=diff_weight,
                                                                           ddiff_weight=ddiff_weight)

                # update gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tot_loss += loss.detach()
                tot_kl_div += kl_div.detach()

                # update pbar
                train_epoch.set_postfix(loss=loss.item())

                # visualize steps with Tensorboard
                if n_iter % writer_interval == 0:
                    writer.add_scalar('Batch/loss', loss, n_iter + epoch * len(train_loader))
                    writer.add_scalar('Batch/diff_loss', diff_loss, n_iter + epoch * len(train_loader))
                    writer.add_scalar('Batch/ddiff_loss', ddiff_loss, n_iter + epoch * len(train_loader))
                    writer.add_scalar('Batch/KL', kl_div, n_iter + epoch * len(train_loader))

        # visualize epochs with Tensorboard
        avg_train_loss = tot_loss / len(train_loader)
        writer.add_scalar('Epoch loss/train', avg_train_loss, epoch)

        avg_train_kl_div = tot_kl_div / len(train_loader)
        writer.add_scalar('Epoch KL/train', avg_train_kl_div, epoch)

        # TESTING
        with tqdm(test_loader, unit='batch', desc=f'Test epoch {epoch}') as test_epoch:
            model.eval()

            # keep track of total losses
            tot_loss = 0
            tot_diff_loss = 0
            tot_ddiff_loss = 0
            tot_kl_div = 0
            tot_ind_losses = torch.zeros(num_elements_in_example, device=device)

            for n_iter, example in enumerate(test_epoch):
                loss, loss_arr, kl_div, diff_loss, ddiff_loss = model_step(device, model, example,
                                                                           loss_weights=params['loss_params'][
                                                                               'loss_weights'],
                                                                           diff_weight=diff_weight,
                                                                           ddiff_weight=ddiff_weight)

                tot_loss += loss.detach()
                tot_diff_loss += diff_loss.detach()
                tot_ddiff_loss += ddiff_loss.detach()
                tot_kl_div += kl_div.detach()

                # update pbar
                test_epoch.set_postfix(loss=loss.item())

                # visualize steps with Tensorboard
                if n_iter % writer_interval == 0:
                    for i, el_loss in enumerate(loss_arr):
                        tot_ind_losses[i] += el_loss

        # show matplotlib graph every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            # extract inputs
            inputs = move_to(example['inputs'], device)

            # output of autoencoder
            outputs, metrics = model(inputs)

            fig = plot_vars(
                inputs=move_to(inputs, device=torch.device('cpu')),
                outputs=move_to(outputs, device=torch.device('cpu')),
                scaling_params=scaling_params,
                spec_list=spec_list,
                model_name=model_name
            )

            writer.add_figure('Plot', fig, epoch)

        # visualize epochs with Tensorboard
        avg_test_loss = tot_loss / len(test_loader)
        writer.add_scalar('Epoch loss/test', avg_test_loss, epoch)

        avg_diff_loss = tot_diff_loss / len(test_loader)
        writer.add_scalar('Epoch diff loss/test', avg_diff_loss, epoch)

        avg_ddiff_loss = tot_ddiff_loss / len(test_loader)
        writer.add_scalar('Epoch ddiff loss/test', avg_ddiff_loss, epoch)

        writer.add_scalar('Epoch diff weight', diff_weight, epoch)
        writer.add_scalar('Epoch ddiff weight', ddiff_weight, epoch)

        avg_test_kl_div = tot_kl_div / len(test_loader)
        writer.add_scalar('Epoch KL/test', avg_test_kl_div, epoch)

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
        tot_diff_loss += diff_loss.detach()
        tot_ddiff_loss += ddiff_loss.detach()
        tot_kl_div = 0
        tot_ind_losses = torch.zeros(num_elements_in_example, device=device)

        for n_iter, example in enumerate(validation):
            loss, loss_arr, kl_div, diff_loss, ddiff_loss = model_step(device, model, example,
                                                                       loss_weights=params['loss_params'][
                                                                           'loss_weights'],
                                                                       diff_weight=diff_weight,
                                                                       ddiff_weight=ddiff_weight)

            tot_loss += loss.detach()
            tot_kl_div += kl_div.detach()

            # update pbar
            validation.set_postfix(loss=loss.item())

            # visualize steps with Tensorboard
            if n_iter % writer_interval == 0:
                for i, el_loss in enumerate(loss_arr):
                    tot_ind_losses[i] += el_loss

    # visualize epochs with Tensorboard
    validation_loss = tot_loss / len(validation_loader)
    validation_diff_loss = tot_diff_loss / len(validation_loader)
    validation_ddiff_loss = tot_ddiff_loss / len(validation_loader)
    validation_kl_div = tot_kl_div / len(validation_loader)

    metric_dict = {
        "Validation/loss": validation_loss,
        "Validation/diff loss": validation_diff_loss,
        "Validation/ddiff loss": validation_ddiff_loss,
        "Validation/KL": validation_kl_div
    }

    validation_ind_losses = tot_ind_losses / len(validation_loader)
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
    # dataset_dir = os.path.join(MRP_dir, 'data/christmas_dataset/wavelength_dataset')
    dataset_dir = os.path.join(MRP_dir, 'data/christmas_dataset/cut_dataset')
    save_model_dir = os.path.join(script_dir, '../saved_models')
    log_dir = os.path.join(script_dir, '../runs')

    # remake the config directory
    if not os.path.isdir(save_model_dir):
        os.mkdir(save_model_dir)

    params = dict(
        name='VAE_large_cut',

        gpu=0,

        ds_params={
            'batch_size': 4,
            'double_mode': True,
            'shuffle': True,
            'num_workers': 4,
            'train_test_validation_ratios': [0.7, 0.2, 0.1]
        },

        model_params={
            'latent_dim': 4096
        },

        optimizer_params={
            'lr': 1e-4
        },

        loss_params={
            'loss_weights': {
                'y_mix_ini': 2.,
                'top_flux': 3.,
                'wavelengths': 3.,
                'Tco': 1.,
                'Pco': 1.,
                'g': 1.,
                'gravity': 1.
            },
            'LossWeightScheduler_d': LossWeightScheduler(
                start_epoch=50,
                end_epoch=150,
                start_weight=0.1,
                end_weight=1e3
            ),
            'LossWeightScheduler_dd': LossWeightScheduler(
                start_epoch=0,
                end_epoch=1,
                start_weight=0,
                end_weight=0
            )
        },

        train_params={
            'epochs': 300,
            'writer_interval': 10,
            'num_elements_in_example': 7
        }
    )

    train_VAE(dataset_dir, save_model_dir, log_dir, params)


if __name__ == "__main__":
    main()
