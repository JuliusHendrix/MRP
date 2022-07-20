import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


# from: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#sphx-glr-intermediate-seq2seq-translation-tutorial-py
# also usefull: https://stats.stackexchange.com/questions/404955/examples-of-one-to-many-for-rnn-lstm
class GRUCore(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, steps, activation_function, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.steps = steps

        # set activation function
        if activation_function == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        elif activation_function == 'tanh':
            self.activation_function = nn.Tanh()
        else:
            raise ValueError('Activation function not supported')

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            self.activation_function,
        )

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device=device, dtype=torch.double)

    def forward(self, input, hidden):
        output, hidden = self.gru(
            input, hidden)  # (b, 1, hidden_size), (1, b, hidden_size)
        output = self.out(output[:, 0, :])  # (b, hidden_size)

        return output, hidden
