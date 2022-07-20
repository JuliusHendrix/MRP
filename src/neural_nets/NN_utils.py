import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import numpy as np

# own modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = str(Path(script_dir).parents[1])
sys.path.append(src_dir)

from src.neural_nets.dataset_utils import unscale_inputs_outputs, unscale_inputs_outputs_model_outputs, unscale
from src.neural_nets.AE.visualize_example import plot_all, plot_y_mix_core, plot_individual_y_mix, plot_single_variable


class LossWeightScheduler:
    def __init__(self, start_epoch, end_epoch, start_weight, end_weight):
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.start_weight = start_weight
        self.end_weight = end_weight

        # line parameters
        self.slope = (self.end_weight - self.start_weight) / (self.end_epoch - self.start_epoch)
        self.offset = self.start_weight - self.slope * self.start_epoch

    def get_weight(self, epoch):
        if epoch < self.start_epoch:
            return self.start_weight
        elif epoch > self.end_epoch:
            return self.end_weight
        else:
            return self.slope * epoch + self.offset


# move non-tensor objects to the gpu
# from https://discuss.pytorch.org/t/pytorch-tensor-to-device-for-a-list-of-dict/66283/2
def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    elif isinstance(obj, float) or isinstance(obj, int):
        return move_to(torch.tensor(obj), device)
    else:
        raise TypeError("Invalid type for move_to")


# getting Product of a tuple
def tuple_product(val):
    res = 1
    for ele in val:
        res *= ele
    return res


def calculate_padding(input_shape, kernel_size, stride, padding=None):
    H = input_shape[0]
    W = input_shape[1]

    def output_dim(input_dim, pad, k_s, s):
        output_d = (input_dim - k_s + 2 * pad) / s + 1
        return output_d

    if padding:
        H_out = output_dim(H, padding[0], kernel_size[0], stride[0])
        W_out = output_dim(W, padding[1], kernel_size[1], stride[1])
    else:
        padding = [-1, -1]

        H_out = 0.5
        while H_out % 1 != 0:
            padding[0] += 1
            if padding[0] > 10:
                raise ValueError('Too much padding...')
            H_out = output_dim(H, padding[0], kernel_size[0], stride[0])
            print(f'{H_out = }')

        W_out = 0.5
        while W_out % 1 != 0:
            padding[1] += 1
            if padding[1] > 10:
                raise ValueError('Too much padding...')
            W_out = output_dim(W, padding[1], kernel_size[1], stride[1])
            print(f'{W_out = }')

    print(f'\ninput_shape: {(H, W)}\n'
          f'output_shape: {(H_out, W_out)}\n'
          f'kernel_size={kernel_size}, stride={stride}, padding={tuple(padding)}')


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

    loss = torch.sum(loss)

    # return torch.log10(loss), torch.log10(loss_arr)
    return loss, loss_arr


def multiple_MSELoss_dict(device, inputs, outputs, weights):
    """
    Calculates the weighted sum of the MSE losses of the inputs and outputs.

    Args:
        device: str, pytorch device (gpu or cpu)
        inputs: dict, inputs
        outputs: dict, outputs
        weights: dict, weights

    Returns:
        loss: torch.Tensor, the weighted mean loss
        loss_arr: torch.Tensor, the individual MSE losses
    """

    loss = torch.zeros(len(inputs), device=device)
    loss_arr = torch.zeros(len(inputs), device=device)

    for i, (key, i_value) in enumerate(inputs.items()):
        o_value = outputs[key]
        weight = weights[key]

        i_loss = torch.mean((o_value - i_value) ** 2)    # mean of batch
        loss[i] = i_loss * weight
        loss_arr[i] = i_loss

    loss_sum = torch.sum(loss)    # TODO: sum or mean?
    # loss = torch.mean(loss)

    return loss_sum, loss_arr


# formula from:
# https://stackoverflow.com/questions/40226357/second-derivative-in-python-scipy-numpy-pandas
def double_derivative(x, y):
    dy = torch.diff(y)
    dx = torch.diff(x)
    y_1 = dy / dx
    x_1 = 0.5*(x[..., -1] + x[..., 1:])

    dy2 = torch.diff(y_1)
    dx2 = torch.diff(x_1)
    y_2 = dy2 / dx2

    return y_2


# formula from:
# https://stackoverflow.com/questions/40226357/second-derivative-in-python-scipy-numpy-pandas
def derivative(x, y):
    dy = torch.diff(y)
    dx = torch.diff(x)
    y_1 = dy / dx
    x_1 = 0.5*(x[..., :-1] + x[..., 1:])

    return x_1, y_1


def derivative_MSE(x_i, y_i, x_o, y_o):
    # input derivatives
    x_1_i, y_1_i = derivative(
        x=x_i,
        y=y_i,
    )

    # output derivatives
    x_1_o, y_1_o = derivative(
        x=x_o,
        y=y_o,
    )

    mse = torch.mean((y_1_i - y_1_o)**2)

    return mse


def double_derivative_MSE(x_i, y_i, x_o, y_o):
    # input derivatives
    x_1_i, y_1_i = derivative(
        x=x_i,
        y=y_i,
    )

    x_2_i, y_2_i = derivative(
        x=x_1_i,
        y=y_1_i
    )

    # output derivatives
    x_1_o, y_1_o = derivative(
        x=x_o,
        y=y_o,
    )

    x_2_o, y_2_o = derivative(
        x=x_1_o,
        y=y_1_o
    )

    mse = torch.mean((y_2_i - y_2_o) ** 2)

    return mse


# changed from scipy
# https://stackoverflow.com/questions/60534909/gaussian-filter-in-pytorch
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.gaussian.html
def gaussian_kernel_1d(M, std, sym=True):
    if M < 1:
        return torch.tensor([])
    if M == 1:
        return torch.ones(1)
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    if not sym and not odd:
        w = w[:-1]
    return w


def plot_vars(inputs, outputs, scaling_params, spec_list, model_name):
    unscaled_dict = unscale_inputs_outputs(inputs, outputs, scaling_params)
    fig = plot_all(unscaled_dict, spec_list, model_name, show=False, save=False)
    return fig


def plot_y_mix(inputs, outputs, decoded_outputs, decoded_model_outputs, scaling_params, spec_list, model_name):
    unscaled_dict = unscale_inputs_outputs_model_outputs(inputs, outputs, decoded_outputs, decoded_model_outputs, scaling_params)
    fig = plot_y_mix_core(unscaled_dict, spec_list, model_name, show=False, save=False, Pco=True)
    return fig


def plot_core_y_mixs( y_mix_decoded_outputs, y_mix_decoded_model_outputs, scales, spec_list, model_name):
    # y_mixs_unscaled = unscale(y_mixs, *scales).detach().numpy()[0]
    y_mix_decoded_outputs_unscaled = unscale(y_mix_decoded_outputs, *scales).detach().numpy()[0]
    y_mix_decoded_model_outputs_unscaled = unscale(y_mix_decoded_model_outputs, *scales).detach().numpy()[0]
    unscaled_dict = {
        # 'outputs': {
        #     'y_mix': y_mixs_unscaled,
        # },
        'decoded_outputs': {
            'y_mix_ini': y_mix_decoded_outputs_unscaled,
        },
        'decoded_model_outputs': {
            'y_mix_ini': y_mix_decoded_model_outputs_unscaled,
        }
    }
    fig = plot_y_mix_core(unscaled_dict, spec_list, model_name, show=False, save=False, Pco=False)
    return fig


def plot_single_y_mix(y_mix, y_mix_decoded, sp_idx, spec_list, scales, model_name):
    y_mix_unscale = unscale(y_mix, *scales).detach().numpy()[0]
    y_mix_decoded_unscale = unscale(y_mix_decoded, *scales).detach().numpy()[0]
    fig = plot_individual_y_mix(y_mix_unscale, y_mix_decoded_unscale, sp_idx, spec_list, model_name)
    return fig


def plot_variable(x, y, y_o, scales, model_name, xlabel, ylabel, xlog=False, ylog=False):
    y_unscale = unscale(y, *scales).detach().numpy()[0]
    y_o_unscale = unscale(y_o, *scales).detach().numpy()[0]
    fig = plot_single_variable(x, y_unscale, y_o_unscale, model_name, xlabel, ylabel, xlog, ylog)
    return fig


# from : https://towardsdatascience.com/weight-decay-and-its-peculiar-effects-66e0aee3e7b8
# and: https://arxiv.org/abs/1711.05101
def weight_decay(lam_norm, batch_size, num_training_points, num_epochs):
    return lam_norm * np.sqrt( batch_size / (num_training_points * num_epochs) )


if __name__ == "__main__":
    # calculate_padding(input_shape=(150, 69),
    #                   kernel_size=(4, 1),
    #                   stride=(2, 1),
    #                   padding=(0, 0)
    #                   )

    x = torch.from_numpy(
        np.linspace(0, 10, 10)
    ).log10()

    y = torch.log10(x**2)

    print(f'{x = }')
    print(f'{y = }')

    dd = double_derivative(x,y)
    print(f'{dd = }')
