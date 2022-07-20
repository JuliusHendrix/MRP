import os
import glob
import torch
from pathlib import Path
import shutil
from tqdm import tqdm
import pickle
import copy
import numpy as np
import multiprocessing as mp

# own modules
from dataset_utils import scale_example


def interp_y_mix(y_mix):
    interp_y_mix = y_mix.clone()

    # loop over layers
    for height_layer in range(len(y_mix)):
        if torch.isnan(y_mix[height_layer]):
            next_non_nan_idx = None
            prev_non_nan_idx = None

            for next_height_layer in range(height_layer + 1, len(y_mix)):
                if not torch.isnan(y_mix[next_height_layer]):
                    next_non_nan_idx = next_height_layer
            for prev_height_layer in range(height_layer):
                if not torch.isnan(y_mix[prev_height_layer]):
                    prev_non_nan_idx = prev_height_layer

            next_non_nan_element = y_mix[next_non_nan_idx] if next_non_nan_idx is not None else None
            prev_non_nan_element = y_mix[prev_non_nan_idx] if prev_non_nan_idx is not None else None

            if next_non_nan_element and prev_non_nan_element:
                # interpolate
                interpolated_value = (next_non_nan_element - prev_non_nan_element) /\
                                             (next_non_nan_idx - prev_non_nan_idx) *\
                                             (height_layer - prev_non_nan_idx) + prev_non_nan_element
                interp_y_mix[height_layer] = interpolated_value
            elif next_non_nan_element:
                interp_y_mix[height_layer] = next_non_nan_element
            elif prev_non_nan_element:
                interp_y_mix[height_layer] = prev_non_nan_element
            else:
                raise ValueError('All non-nan element found for interpolation!')

    return interp_y_mix


def interpolate_y_mixs(y_mixs):
    interp_y_mixs = y_mixs.clone()

    # loop over mixing ratios
    for i_y in range(y_mixs.shape[-1]):
        y_mix = y_mixs[:, i_y]  # [height_layers]
        interp_y_mixs[:, i_y] = interp_y_mix(y_mix)

    return interp_y_mixs


def interpolate_example(scaled_example, time_series=False):
    interp_example = copy.deepcopy(scaled_example)
    interp_example['inputs']['y_mix_ini'] = interpolate_y_mixs(scaled_example['inputs']['y_mix_ini'])

    if time_series:
        for i_y_mix in range(interp_example['outputs']['y_mixs'].shape[0]):    # (steps, height_layers, num_species)
            interp_example['outputs']['y_mixs'][i_y_mix, :, :] = interpolate_y_mixs(interp_example['outputs']['y_mixs'][i_y_mix, :, :])
    else:
        interp_example['outputs']['y_mix'] = interpolate_y_mixs(scaled_example['outputs']['y_mix'])
    return interp_example


def interpolate_torch_file(params):
    torch_file, interp_ds_dir, scaling_dict, time_series = params
    example = torch.load(torch_file)

    scaled_example = scale_example(example, scaling_dict, nans=True)
    interp_example = interpolate_example(scaled_example, time_series=time_series)

    torch_filename = os.path.basename(torch_file)
    interp_torch_file = os.path.join(interp_ds_dir, torch_filename)
    torch.save(interp_example, interp_torch_file)

    return 0


def interpolate_dataset(ds_dir, num_workers, time_series=False):
    interp_ds_dir = os.path.join(ds_dir, 'interpolated_dataset')

    # remake the config directory
    if os.path.isdir(interp_ds_dir):
        shutil.rmtree(interp_ds_dir)
    os.mkdir(interp_ds_dir)

    # get scaling parameters
    scaling_file = os.path.join(ds_dir, 'scaling_dict.pkl')
    with open(scaling_file, 'rb') as f:
        scaling_dict = pickle.load(f)

    torch_files = glob.glob(os.path.join(ds_dir, '*.pt'))

    if num_workers > 1:
        mp_params = [(torch_file, interp_ds_dir, scaling_dict, time_series) for torch_file in torch_files]

        print(f'running with {num_workers} workers...')
        with mp.get_context("spawn").Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(interpolate_torch_file, mp_params),  # return results otherwise it doesn't work properly
                                total=len(mp_params), desc='interpolating and scaling torch files'))

    else:
        # loop through examples
        for torch_file in tqdm(torch_files, desc='interpolating and scaling torch files'):
            interpolate_torch_file((torch_file, scaling_dict))


def main():
    # setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MRP_dir = str(Path(script_dir).parents[1])
    dataset_dir = os.path.join(MRP_dir, 'data/bday_dataset/time_series_dataset')

    interpolate_dataset(dataset_dir, num_workers=mp.cpu_count() - 1, time_series=True)


if __name__ == '__main__':
    main()
