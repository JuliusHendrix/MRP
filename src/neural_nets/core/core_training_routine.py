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

from src.neural_nets.dataloaders import SingleVulcanDataset
from src.neural_nets.dataset_utils import make_data_loaders
from src.neural_nets.NN_utils import move_to, plot_core_y_mixs, weight_decay

from src.neural_nets.core.ae_params import ae_params
from src.neural_nets.core.gaussian_noise import GaussianNoise


def loss_fn(x, y):
    '''
    MSE error between x and y
    '''

    loss = torch.mean((x - y) ** 2)
    return loss


def encode_y_mixs(device, y_mixs, mrae_model):
    # mixing ratio's
    y_mixs_latent = torch.zeros(y_mixs.shape[0], mrae_model.latent_dim, y_mixs.shape[2]).double().to(
        device)  # [b, mrae_latent_dim, num_species]
    for i_y in range(y_mixs.shape[-1]):
        y_mix = y_mixs[:, :, i_y]  # [b, height_layers]
        y_mixs_latent[:, :, i_y] = mrae_model.encode(y_mix)
    y_mixs_latent = y_mixs_latent.flatten(start_dim=1)  # [b, num_species * mrae_latent_dim]
    return y_mixs_latent


def decode_y_mixs(device, y_mixs_latent, mrae_model, num_species):
    # mixing ratio's
    y_mixs = torch.zeros(y_mixs_latent.shape[0], 150, num_species).double().to(device)  # [b, 150, num_species]
    y_mixs_latent = y_mixs_latent.reshape(y_mixs_latent.shape[0], mrae_model.latent_dim,
                                          num_species)  # [b, mrae_latent_dim, num_species]
    for i_y in range(y_mixs_latent.shape[-1]):
        y_mix_latent = y_mixs_latent[:, :, i_y]  # [b, mrae_latent_dim]
        y_mixs[:, :, i_y] = mrae_model.decode(y_mix_latent)
    return y_mixs


def initialize_models(device, models, state_dicts, model_params, save_model_dir):
    initialized_models = {}

    for key in models.keys():
        model = models[key](**model_params[key]).double().to(device)
        if state_dicts[key] is not None:
            model.load_state_dict(
                torch.load(os.path.join(save_model_dir, state_dicts[key]), map_location='cuda:0')
            )
        model.eval()
        initialized_models.update({key: model})

    return initialized_models


def encode_inputs_outputs(device, ae_models, example, time_series=False):
    # extract inputs
    inputs = move_to(example['inputs'], device)
    outputs = move_to(example['outputs'], device)

    # encode individual parts for input and output example

    # mixing ratio's
    y_mixs_latent_inputs = encode_y_mixs(device, inputs['y_mix_ini'], ae_models['mrae'])

    if time_series:
        y_mixs = outputs['y_mixs']
        y_mixs_latent_outputs = torch.zeros(y_mixs.shape[0], y_mixs.shape[1],
                                            # [b, time_steps, mrae_latent_dim*num_species]
                                            ae_models['mrae'].latent_dim * y_mixs.shape[-1]).double().to(device)
        for i_y_mix in range(y_mixs.shape[1]):
            y_mixs_latent = encode_y_mixs(device, y_mixs[:, i_y_mix, :, :], ae_models['mrae'])
            y_mixs_latent_outputs[:, i_y_mix, :] = y_mixs_latent
    else:
        y_mixs_latent_outputs = encode_y_mixs(device, outputs['y_mix'], ae_models['mrae'])

    # wavelengths
    wls_latent_inputs = ae_models['wae'].encode(inputs['wavelengths'])

    # flux
    flux_latent_inputs = ae_models['fae'].encode(inputs['top_flux'])

    # pressure
    pressure_latent_inputs = ae_models['pae'].encode(inputs['Pco'])

    # temperature
    temperature_latent_inputs = ae_models['tae'].encode(inputs['Tco'])

    # gravity
    gravity_latent_inputs = ae_models['gae'].encode(inputs['g'])

    # to latent representations
    latent_input = torch.cat((
        y_mixs_latent_inputs,
        wls_latent_inputs,
        flux_latent_inputs,
        pressure_latent_inputs,
        temperature_latent_inputs,
        gravity_latent_inputs),
        dim=1)  # [b, latent_dim]

    return latent_input, y_mixs_latent_outputs


def train_core(dataset_dir, save_model_dir, log_dir, params):
    # headless plotting
    import matplotlib
    matplotlib.use('Agg')

    # setup pytorch
    # device = torch.device(f"cuda:{params['gpu']}" if torch.cuda.is_available() else "cpu")

    os.environ['CUDA_VISIBLE_DEVICES'] = f"{params['gpu']}"
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{os.environ['CUDA_VISIBLE_DEVICES'] =}")
    print(f'running on device: {device}')

    # load datasets
    train_loader, test_loader, validation_loader = make_data_loaders(SingleVulcanDataset,
                                                                     os.path.join(dataset_dir, 'interpolated_dataset/'),
                                                                     **params['ds_params'])

    # initialize core model
    core_model = params['core_model'](
        **params['core_model_params'],
        **params['core_model_extra_params'],
        device=device
    ).double().to(device)

    # Initialize models with double precision
    ae_models = initialize_models(device, ae_params['models'], ae_params['state_dicts'], ae_params['model_params'],
                                  save_model_dir)

    # Create optimizer and add weight decay if applicable
    if params['core_model_params']['weight_decay_norm'] > 0:
        optimizer = torch.optim.Adam(core_model.parameters(), **params['optimizer_params'],
                                     weight_decay=weight_decay(
                                         lam_norm=params['core_model_params']['weight_decay_norm'],
                                         batch_size=params['ds_params']['batch_size'],
                                         num_training_points=len(train_loader) * params['ds_params']['batch_size'],
                                         num_epochs=params['train_params']['epochs']
                                     )
                                     )
    else:
        optimizer = torch.optim.Adam(core_model.parameters(), **params['optimizer_params'])

    # Loss function
    # loss_fn = torch.nn.MSELoss()
    loss_fn = params['loss_function']

    # add noise if sigma
    if params['core_model_params']['sigma'] > 0:
        noise = GaussianNoise(device, params['core_model_params']['sigma'])
    else:
        noise = None

    # Tensorboard logging
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    hparams = {}
    hparams.update(params['core_model_params'])
    hparams.update(params['optimizer_params'])

    model_name = f'{params["name"]},{hparams=}'
    summary_file = dt_string + f' | {model_name}'
    writer = SummaryWriter(
        log_dir=os.path.join(log_dir, summary_file)
    )

    # save validation indices
    torch.save(validation_loader.dataset.indices, os.path.join(save_model_dir, f'{model_name}_validation_indices.pt'))

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

    time_series = params['core_model_params']['time_series']

    # save best model params
    best_loss = torch.inf
    best_model_params = {}

    for epoch in range(epochs):
        # TRAINING
        with tqdm(train_loader, unit='batch', desc=f'Train epoch {epoch}') as train_epoch:
            core_model.train()

            # keep track of total loss
            tot_loss = 0

            for n_iter, example in enumerate(train_epoch):
                latent_input, y_mixs_latent_outputs = encode_inputs_outputs(device, ae_models, example,
                                                                            time_series=time_series)

                # add noise
                if noise is not None:
                    latent_input = noise(latent_input)

                if time_series:
                    loss, latent_model_output = params['core_model_step'](
                        latent_input, y_mixs_latent_outputs, core_model, loss_fn, device=device)
                else:
                    latent_model_output = params['core_model_step'](latent_input, core_model, device=device)
                    loss = loss_fn(latent_model_output, y_mixs_latent_outputs)

                # update gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tot_loss += loss.detach()

                # visualize steps with Tensorboard
                if n_iter % writer_interval == 0:
                    writer.add_scalar('Batch/loss', loss, n_iter + epoch * len(train_loader))

        # visualize epochs with Tensorboard
        avg_train_loss = tot_loss / len(train_loader)
        writer.add_scalar('Epoch loss/train', avg_train_loss, epoch)

        # TESTING
        with tqdm(test_loader, unit='batch', desc=f'Test epoch {epoch}') as test_epoch:
            core_model.eval()

            # keep track of total losses
            tot_loss = 0

            for n_iter, example in enumerate(test_epoch):
                latent_input, y_mixs_latent_outputs = encode_inputs_outputs(device, ae_models, example,
                                                                            time_series=time_series)

                # add noise
                if noise is not None:
                    latent_input = noise(latent_input)

                if time_series:
                    loss, latent_model_output = params['core_model_step'](
                        latent_input, y_mixs_latent_outputs, core_model, loss_fn, device=device)
                else:
                    latent_model_output = params['core_model_step'](latent_input, core_model, device=device)
                    loss = loss_fn(latent_model_output, y_mixs_latent_outputs)

                tot_loss += loss.detach()

        # show matplotlib graph every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            latent_input, y_mixs_latent_outputs = encode_inputs_outputs(device, ae_models, example,
                                                                        time_series=time_series)

            if time_series:
                loss, latent_model_output = params['core_model_step'](
                    latent_input, y_mixs_latent_outputs, core_model, loss_fn, device=device)
            else:
                latent_model_output = params['core_model_step'](latent_input, core_model, device=device)

            # decode latent model output
            decoded_model_outputs = decode_y_mixs(device, latent_model_output, ae_models['mrae'], len(spec_list))

            # decode latent output
            if time_series:
                decoded_outputs = decode_y_mixs(device, y_mixs_latent_outputs[:, -1, :], ae_models['mrae'], len(spec_list))
            else:
                decoded_outputs = decode_y_mixs(device, y_mixs_latent_outputs, ae_models['mrae'], len(spec_list))

            # plot
            scales = scaling_params['inputs']['y_mix_ini']
            fig = plot_core_y_mixs(
                y_mix_decoded_outputs=move_to(decoded_outputs, device=torch.device('cpu')),
                y_mix_decoded_model_outputs=move_to(decoded_model_outputs, device=torch.device('cpu')),
                scales=scales,
                spec_list=spec_list,
                model_name=model_name
            )

            writer.add_figure('Plot', fig, epoch)

        # visualize epochs with Tensorboard
        avg_test_loss = tot_loss / len(test_loader)
        writer.add_scalar('Epoch loss/test', avg_test_loss, epoch)

        # save best model params
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_model_params = core_model.state_dict().copy()

            # save the model
            torch.save(best_model_params, os.path.join(save_model_dir, f'{model_name}_state_dict'))

    # load best model params
    core_model.load_state_dict(best_model_params)

    core_model.load_state_dict(best_model_params)

    # VALIDATION
    with tqdm(validation_loader, unit='batch', desc='Validation') as validation:
        core_model.eval()

        # keep track of total losses
        tot_loss = 0

        for n_iter, example in enumerate(validation):
            latent_input, y_mixs_latent_outputs = encode_inputs_outputs(device, ae_models, example,
                                                                        time_series=time_series)

            # add noise
            if noise is not None:
                latent_input = noise(latent_input)

            if time_series:
                loss, latent_model_output = params['core_model_step'](
                    latent_input, y_mixs_latent_outputs, core_model, loss_fn, device=device)
            else:
                latent_model_output = params['core_model_step'](latent_input, core_model, device=device)
                loss = loss_fn(latent_model_output, y_mixs_latent_outputs)

            tot_loss += loss.detach()

    # visualize epochs with Tensorboard
    validation_loss = tot_loss / len(validation_loader)

    metric_dict = {"Validation/loss": validation_loss}

    # add hyperparameters
    writer.add_hparams(
        hparams,
        metric_dict
    )

    # make sure to write everything
    writer.flush()

    # close Tensorboard
    writer.close()
