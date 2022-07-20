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

# own modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = str(Path(script_dir).parents[2])
sys.path.append(src_dir)

from src.neural_nets.dataset_utils import unscale, scale
from src.neural_nets.NN_utils import move_to

from src.visualization.plot_AE_performance.AE_settings import get_params


def save_AE_performance(device, params, dataset_dir, save_model_dir):

    # initialize model
    model = params['model'](
        **params['model_params']
    ).double().to(device)

    # get model name
    hparams = {}
    hparams.update(params['model_params'])
    hparams.update(params['optimizer_params'])
    model_name = f'{params["name"]},{hparams=}'

    # load previous model
    print('loading state dict...')
    model.load_state_dict(torch.load(os.path.join(save_model_dir, f'{model_name}_state_dict'), map_location='cpu'))

    # get validation indices
    validation_indices = torch.load(os.path.join(save_model_dir, f'{model_name}_validation_indices.pt'))

    # dataset loader
    vulcan_dataset = params["dataloader"](os.path.join(dataset_dir, 'interpolated_dataset'))
    validation_dataset = Subset(vulcan_dataset, validation_indices)

    dataloader = DataLoader(validation_dataset, batch_size=1,
                            shuffle=True,
                            num_workers=0)

    print(f'{len(dataloader)} validation examples')

    # get scaling parameters
    scaling_file = os.path.join(dataset_dir, 'scaling_dict.pkl')
    with open(scaling_file, 'rb') as f:
        scaling_params = pickle.load(f)

    # evaluation mode
    model.eval()

    # save actual and reconstructed values
    for i, dummy_example in enumerate(dataloader):
        input, output = params['model_step'](device, model, dummy_example)

        # create empty array to save the actual and predicted values
        input = move_to(input, device=torch.device('cpu'))
        zero_value = np.zeros_like(input[0, ...])
        actual = np.tile(zero_value[..., None], len(dataloader))
        predictions = copy.deepcopy(actual)

        print(f'{actual.shape = }')

        break

    # create dict to hold actual and predicted values
    perf_dict = {
        'actual': actual,
        'predictions': predictions
    }

    # loop through examples
    with tqdm(dataloader, unit='example', desc=f'Predicting values') as dataloader:
        for i, example in enumerate(dataloader):
            # model step
            input, output = params['model_step'](device, model, example)

            input = move_to(input, device=torch.device('cpu'))
            output = move_to(output, device=torch.device('cpu'))

            # scales
            scales = scaling_params['inputs'][params["variable_name"]]

            # unscale
            input_unscale = unscale(input, *scales).detach().numpy()[0]
            output_unscale = unscale(output, *scales).detach().numpy()[0]

            # add to dict
            perf_dict['actual'][:, i] = input_unscale
            perf_dict['predictions'][:, i] = output_unscale

    # save to disk
    perf_dict_file = f'performance_dicts/{model_name}_perf_dict.pkl'
    with open(perf_dict_file, 'wb') as f:
        pickle.dump(perf_dict, f)


def main():
    # setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MRP_dir = str(Path(script_dir).parents[2])
    dataset_dir = os.path.join(MRP_dir, 'data/bday_dataset/dataset')
    # dataset_dir = os.path.join(MRP_dir, 'data/bday_dataset/time_series_dataset')
    save_model_dir = os.path.join(MRP_dir, 'src/neural_nets/saved_models_final')

    # setup pytorch
    device = torch.device("cpu")
    print(f'running on device: {device}')

    AE_name = 'WAE'

    params = get_params(AE_name)

    save_AE_performance(device, params, dataset_dir, save_model_dir)


if __name__ == "__main__":
    main()
