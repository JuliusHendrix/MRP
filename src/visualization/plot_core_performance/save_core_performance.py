import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
import pickle
import numpy as np
from tqdm import tqdm
import copy
from functools import partial
import timeit

# own modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = str(Path(script_dir).parents[2])
sys.path.append(src_dir)

from src.neural_nets.dataset_utils import unscale, scale
from src.neural_nets.NN_utils import move_to

from src.visualization.plot_core_performance.core_settings import get_params

from src.neural_nets.core.ae_params import ae_params
from src.neural_nets.core.core_training_routine import initialize_models, encode_inputs_outputs, decode_y_mixs

from src.neural_nets.dataloaders import SingleVulcanDataset

def save_core_performance(device, params, dataset_dir, save_model_dir, time_only=False):

    # initialize core model
    core_model = params['model'](
        **params['core_model_params'],
        **params['core_model_extra_params'],
        device=device
    ).double().to(device)

    # Initialize models with double precision
    ae_models = initialize_models(device, ae_params['models'], ae_params['state_dicts'], ae_params['model_params'],
                                  save_model_dir)

    # get model name
    hparams = {}
    hparams.update(params['core_model_params'])
    hparams.update(params['optimizer_params'])
    model_name = f'{params["name"]},{hparams=}'

    # load previous model
    print('loading state dict...')
    core_model.load_state_dict(torch.load(os.path.join(save_model_dir, f'{model_name}_state_dict'), map_location='cpu'))

    # get validation indices
    validation_indices = torch.load(os.path.join(save_model_dir, f'{model_name}_validation_indices.pt'))

    # dataset loader
    vulcan_dataset = SingleVulcanDataset(os.path.join(dataset_dir, 'interpolated_dataset'))
    if time_only:
        validation_dataset = vulcan_dataset
    else:
        validation_dataset = Subset(vulcan_dataset, validation_indices)

    dataloader = DataLoader(validation_dataset, batch_size=1,
                            shuffle=True,
                            num_workers=0)

    print(f'{len(dataloader)} validation examples')

    # get scaling parameters
    scaling_file = os.path.join(dataset_dir, 'scaling_dict.pkl')
    with open(scaling_file, 'rb') as f:
        scaling_params = pickle.load(f)

    # get species list
    scaling_file = os.path.join(dataset_dir, 'species_list.pkl')
    with open(scaling_file, 'rb') as f:
        spec_list = pickle.load(f)

    # evaluation mode
    core_model.eval()

    def core_model_step(example):
        latent_input, y_mixs_latent_outputs = encode_inputs_outputs(device, ae_models, example, time_series=True)
        latent_model_output = params['model_step'](latent_input, core_model, device=device)

        decoded_model_outputs = decode_y_mixs(device, latent_model_output, ae_models['mrae'], len(spec_list))
        decoded_outputs = decode_y_mixs(device, y_mixs_latent_outputs[:, -1, :], ae_models['mrae'], len(spec_list))

        return decoded_model_outputs, decoded_outputs


    # save actual and reconstructed values
    for i, dummy_example in enumerate(dataloader):
        input, output = core_model_step(dummy_example)

        # create empty array to save the actual and predicted values
        zero_value = np.zeros_like(input[0, ...].detach().numpy())    # 0 to remove batch dimension
        actual = np.tile(zero_value[..., None], len(dataloader))
        predictions = copy.deepcopy(actual)

        print(f'{actual.shape = }')

        break

    # create dict to hold actual and predicted values
    if time_only:
        perf_dict = {'time': np.zeros((actual.shape[-1]))}
    else:
        perf_dict = {
            'actual': actual,
            'predictions': predictions,
            'time': np.zeros((actual.shape[-1]))
        }

    # loop through examples
    with tqdm(dataloader, unit='example', desc=f'Predicting values') as dataloader:
        for i, example in enumerate(dataloader):
            time_start = timeit.default_timer()

            # model step
            input, output = core_model_step(example)

            input = move_to(input, device=torch.device('cpu'))
            output = move_to(output, device=torch.device('cpu'))

            # scales
            scales = scaling_params['inputs']['y_mix_ini']

            # unscale
            input_unscale = unscale(input, *scales).detach().numpy()[0]
            output_unscale = unscale(output, *scales).detach().numpy()[0]

            elapsed_time = timeit.default_timer() - time_start

            # add to dict
            if not time_only:
                perf_dict['actual'][..., i] = input_unscale
                perf_dict['predictions'][..., i] = output_unscale
            perf_dict['time'][i] = elapsed_time

    # save to disk
    if time_only:
        perf_dict_file = f'performance_dicts/{model_name}_perf_dict_time_only.pkl'
    else:
        perf_dict_file = f'performance_dicts/{model_name}_perf_dict.pkl'
    with open(perf_dict_file, 'wb') as f:
        pickle.dump(perf_dict, f)


def main():
    # setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MRP_dir = str(Path(script_dir).parents[2])
    dataset_dir = os.path.join(MRP_dir, 'data/bday_dataset/time_series_dataset')
    # dataset_dir = os.path.join(MRP_dir, 'data/bday_dataset/time_series_dataset')
    save_model_dir = os.path.join(MRP_dir, 'src/neural_nets/saved_models_final')

    # setup pytorch
    device = torch.device("cpu")
    print(f'running on device: {device}')

    core_name = 'LSTM'

    params = get_params(core_name)

    save_core_performance(device, params, dataset_dir, save_model_dir, time_only=True)


if __name__ == "__main__":
    main()
