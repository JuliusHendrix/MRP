import torch.nn as nn


class CopyAE(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, x):
        return x

    def decode(self, x):
        return x

    def forward(self, x):
        return x
