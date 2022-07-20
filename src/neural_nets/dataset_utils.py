import os
import glob
import pickle
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import shutil

zero_value = torch.tensor(1e-45).double()
inf_value = torch.tensor(1e38).double()

# same_scale_items = np.array([
#     ['inputs', 'y_mix_ini', 'outputs', 'y_mix']
# ])

same_scale_items = np.array([
    ['inputs', 'y_mix_ini', 'outputs', 'y_mixs']
])


def copy_output_to_input(example):
    """
    Make new example with the y_ini_mix input replaced with the y_mix output.

    Args:
        example: dict, the example

    Returns:
        new_example: dict, the new example

    """
    new_example = example.copy()
    new_example['inputs']['y_mix_ini'] = example['outputs']['y_mix']

    # if example['outputs']['y_mix'].detach().numpy().any() == np.nan:
    #     print('nans!')

    return new_example


def calculate_log_mean_std_min_max(value):
    value = value.detach().numpy()
    value = np.where(value == 0.0, zero_value, value)

    log_value = np.log10(value)
    mean = np.mean(log_value)
    std = np.std(log_value)

    standardized_value = distribution_standardization(log_value, mean, std)

    return mean, std, np.min(standardized_value), np.max(standardized_value)


def create_scaling_dict(ds_dir):
    examples_files = glob.glob(os.path.join(ds_dir, '*.pt'))
    num_files = len(examples_files)

    test_example = torch.load(examples_files[0])
    num_input_items = len(test_example['inputs'])
    inputs_keys = test_example['inputs'].keys()
    num_output_items = len(test_example['outputs'])
    outputs_keys = test_example['outputs'].keys()

    input_means = np.empty(shape=(num_files, num_input_items))
    input_stds = np.empty(shape=(num_files, num_input_items))
    input_min = np.ones(shape=num_input_items) * np.inf
    input_max = np.ones(shape=num_input_items) * -np.inf

    output_means = np.empty(shape=(num_files, num_output_items))
    output_stds = np.empty(shape=(num_files, num_output_items))
    output_min = np.ones(shape=num_input_items) * np.inf
    output_max = np.ones(shape=num_input_items) * -np.inf

    # extract all means, stds, mins, maxs
    for i_file, example_file in enumerate(tqdm(examples_files, desc='calculating means, stds, mins en maxs')):
        example = torch.load(example_file)

        inputs = example['inputs']
        outputs = example['outputs']

        for i_dict, (key, value) in enumerate(inputs.items()):
            mean, std, v_min, v_max = calculate_log_mean_std_min_max(value)
            input_means[i_file, i_dict] = mean
            input_stds[i_file, i_dict] = std

            if v_min < input_min[i_dict]:
                input_min[i_dict] = v_min
            if v_max > input_max[i_dict]:
                input_max[i_dict] = v_max

        for i_dict, (key, value) in enumerate(outputs.items()):
            mean, std, v_min, v_max = calculate_log_mean_std_min_max(value)
            output_means[i_file, i_dict] = mean
            output_stds[i_file, i_dict] = std

            if v_min < output_min[i_dict]:
                output_min[i_dict] = v_min
            if v_max > output_max[i_dict]:
                output_max[i_dict] = v_max


    # calculate means
    avg_input_means = np.mean(input_means, axis=0)
    avg_input_stds = np.mean(input_stds, axis=0)
    avg_output_means = np.mean(output_means, axis=0)
    avg_output_stds = np.mean(output_stds, axis=0)

    # create scaling dict
    scaling_dict_inputs = {}
    for key, mean, std, min, max in zip(inputs_keys, avg_input_means, avg_input_stds, input_min, input_max):
        scaling_dict_inputs.update(
            {key: (mean, std, min, max)}
        )

    scaling_dict_outputs = {}
    for key, mean, std, min, max in zip(outputs_keys, avg_output_means, avg_output_stds, output_min, output_max):
        scaling_dict_outputs.update(
            {key: (mean, std, min, max)}
        )

    scaling_dict = {
        'inputs': scaling_dict_inputs,
        'outputs': scaling_dict_outputs
    }

    # copy dict
    same_scaling_dict = {key: {} for key, _ in scaling_dict.items()}
    for top_key, top_value in same_scaling_dict.items():
        same_scaling_dict[top_key] = {key: value[:] for key, value in scaling_dict[top_key].items()}

    for top_key, top_value in scaling_dict.items():
        for key, value in top_value.items():
            if key in same_scale_items:
                row = np.where(same_scale_items == key)[0][0]
                mean = np.mean([
                    scaling_dict[same_scale_items[row, 0]][same_scale_items[row, 1]][0],
                    scaling_dict[same_scale_items[row, 2]][same_scale_items[row, 3]][0]
                ])

                std = np.mean([
                    scaling_dict[same_scale_items[row, 0]][same_scale_items[row, 1]][1],
                    scaling_dict[same_scale_items[row, 2]][same_scale_items[row, 3]][1]
                ])

                prop_min = np.mean([
                    scaling_dict[same_scale_items[row, 0]][same_scale_items[row, 1]][2],
                    scaling_dict[same_scale_items[row, 2]][same_scale_items[row, 3]][2]
                ])

                prop_max = np.mean([
                    scaling_dict[same_scale_items[row, 0]][same_scale_items[row, 1]][3],
                    scaling_dict[same_scale_items[row, 2]][same_scale_items[row, 3]][3]
                ])

                same_scaling_dict[top_key][key] = (mean, std, prop_min, prop_max)

    scaling_dict = same_scaling_dict

    # save dict
    scaling_dict_file = os.path.join(ds_dir, 'scaling_dict.pkl')
    with open(scaling_dict_file, 'wb') as f:
        pickle.dump(scaling_dict, f)


def make_data_loaders(dataloader, dataset_dir, train_test_validation_ratios, batch_size, shuffle, num_workers):
    # dataset loader
    vulcan_dataset = dataloader(dataset_dir)

    # split like this to make sure len(subsets) = len(dataset)
    train_size = int(train_test_validation_ratios[0] * len(vulcan_dataset))
    test_size = int(train_test_validation_ratios[1] * len(vulcan_dataset))
    validation_size = len(vulcan_dataset) - train_size - test_size

    train_dataset, test_dataset, validation_dataset = torch.utils.data.random_split(vulcan_dataset,
                                                                                    [train_size, test_size,
                                                                                     validation_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                              pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                                   shuffle=shuffle,
                                   num_workers=num_workers,
                              pin_memory=True)

    return train_loader, test_loader, validation_loader


def distribution_standardization(prop, prop_mean, prop_std):
    if prop_std == 0.0:
        return prop - prop_mean
    else:
        return (prop - prop_mean) / prop_std


def reverse_distribution_standardization(prop, prop_mean, prop_std):
    if prop_std == 0.0:
        return prop + prop_mean
    else:
        return prop * prop_std + prop_mean


def scale(prop, prop_mean, prop_std, prop_min, prop_max, nans=False):
    # cap values

    prop = prop.double()
    prop = torch.where(prop < zero_value,
                       float('nan') if nans else zero_value,    # TODO: float to double?
                       prop)

    prop = torch.where(prop > inf_value, inf_value, prop)

    standardized_prop = distribution_standardization(torch.log10(prop), prop_mean, prop_std)

    if prop_min == prop_max:
        return standardized_prop
    else:
        return (standardized_prop - prop_min) / (prop_max - prop_min)


def unscale(prop, prop_mean, prop_std, prop_min, prop_max):
    if prop_min == prop_max:
        unnorm_prop = prop
    else:
        unnorm_prop = prop * (prop_max - prop_min) + prop_min
    unscaled_prop = 10**reverse_distribution_standardization(unnorm_prop, prop_mean, prop_std)
    # unscaled_prop[unscaled_prop <= zero_value] = 0.0
    return unscaled_prop


def scale_example(example, scaling_dict, nans=False):
    scaled_example = {
        'inputs': {},
        'outputs': {}
    }

    for top_key in scaled_example.keys():
        for (key, value), scales in zip(example[top_key].items(), scaling_dict[top_key].values()):
            # scale values
            scaled_value = scale(value, *scales, nans=nans)
            scaled_example[top_key].update(
                {key: scaled_value}
            )

    return scaled_example


def unscale_example(example, scaling_params):
    unscaled_example = example.copy()
    for top_key, top_value in example.items():
        for key, value in top_value.items():
            scales = scaling_params[top_key][key]
            uncsaled_value = unscale(value, *scales)
            unscaled_example[top_key][key] = uncsaled_value.detach().numpy()[0]
    return unscaled_example


def unscale_inputs_outputs(inputs, outputs, scaling_params):
    # unscale inputs and outputs
    unscaled_dict = {
        'inputs': {},
        'outputs': {}
    }

    for key, i_value in inputs.items():
        scales = scaling_params['inputs'][key]

        unscaled_input = unscale(i_value, *scales)
        unscaled_dict['inputs'].update(
            {key: unscaled_input.detach().numpy()[0]}
        )

        o_value = outputs[key]
        unscaled_output = unscale(o_value, *scales)
        unscaled_dict['outputs'].update(
            {key: unscaled_output.detach().numpy()[0]}
        )

    return unscaled_dict


def unscale_inputs_outputs_model_outputs(inputs, outputs, decoded_outputs, decoded_model_outputs, scaling_params):
    # unscale outputs and model outputs
    unscaled_dict = {
        'inputs': {},
        'outputs': {},
        'decoded_outputs': {},
        'decoded_model_outputs': {}
    }

    # inputs
    for i_key, i_value in inputs.items():
        scales = scaling_params['inputs'][i_key]

        unscaled_input = unscale(i_value, *scales)
        unscaled_dict['inputs'].update(
            {i_key: unscaled_input.detach().numpy()[0]}
        )

    # outputs
    for o_key, o_value in outputs.items():
        scales = scaling_params['outputs'][o_key]
        unscaled_output = unscale(o_value, *scales)
        unscaled_dict['outputs'].update(
            {o_key: unscaled_output.detach().numpy()[0]}
        )

    # model outputs
    for do_key, do_value in decoded_outputs.items():
        scales = scaling_params['inputs'][do_key]
        unscaled_do_output = unscale(do_value, *scales)
        unscaled_dict['decoded_outputs'].update(
            {do_key: unscaled_do_output.detach().numpy()[0]}
        )

    # decoded model outputs
    for dmo_key, dmo_value in decoded_model_outputs.items():
        scales = scaling_params['inputs'][dmo_key]
        unscaled_dmo_output = unscale(dmo_value, *scales)
        unscaled_dict['decoded_model_outputs'].update(
            {dmo_key: unscaled_dmo_output.detach().numpy()[0]}
        )

    return unscaled_dict


def scale_dataset(dataset_dir):
    # get scaling parameters
    scaling_file = os.path.join(dataset_dir, 'scaling_dict.pkl')
    with open(scaling_file, 'rb') as f:
        scaling_dict = pickle.load(f)

    torch_files = glob.glob(os.path.join(dataset_dir, '*.pt'))

    scaled_dataset_dir = os.path.join(dataset_dir, 'scaled_dataset/')

    # remake the config directory
    if os.path.isdir(scaled_dataset_dir):
        shutil.rmtree(scaled_dataset_dir)
    os.mkdir(scaled_dataset_dir)

    # loop through examples
    for torch_file in tqdm(torch_files, desc='scaling torch files'):
        example = torch.load(torch_file)
        scaled_example = scale_example(example, scaling_dict)

        torch_filename = os.path.basename(torch_file)
        scaled_torch_file = os.path.join(scaled_dataset_dir, torch_filename)
        torch.save(scaled_example, scaled_torch_file)


if __name__ == "__main__":
    ds_dir = os.path.expanduser('~/git/MRP/data/bday_dataset/dataset')
    # create_scaling_dict(ds_dir)
    # scale_dataset(ds_dir)
