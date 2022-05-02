import torch.nn as nn


class MixingRatioAE(nn.Module):
    def __init__(self, latent_dim, layer_size, activation_function):
        super().__init__()

        self.latent_dim = latent_dim
        self.layer_size = layer_size

        # set activation function
        if activation_function == 'leaky_relu':
            self.activation_function = nn.LeakyReLU
        elif activation_function == 'tanh':
            self.activation_function = nn.Tanh
        else:
            raise ValueError('Activation function not supported')

        self.encoder = nn.Sequential(
            nn.Linear(150, self.layer_size),
            self.activation_function(),
            nn.Linear(self.layer_size, self.layer_size),
            self.activation_function(),
            nn.Linear(self.layer_size, self.layer_size),
            self.activation_function(),
            nn.Linear(self.layer_size, self.layer_size),
            self.activation_function(),
            nn.Linear(self.layer_size, self.layer_size),
            self.activation_function(),
            nn.Linear(self.layer_size, self.layer_size),
            self.activation_function(),
            nn.Linear(self.layer_size, self.latent_dim),
            self.activation_function(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.layer_size),
            self.activation_function(),
            nn.Linear(self.layer_size, self.layer_size),
            self.activation_function(),
            nn.Linear(self.layer_size, self.layer_size),
            self.activation_function(),
            nn.Linear(self.layer_size, self.layer_size),
            self.activation_function(),
            nn.Linear(self.layer_size, self.layer_size),
            self.activation_function(),
            nn.Linear(self.layer_size, self.layer_size),
            self.activation_function(),
            nn.Linear(self.layer_size, 150),
            self.activation_function(),
        )

    def encode(self, y_mix):
        return self.encoder(y_mix)

    def decode(self, y_mix_latent):
        return self.decoder(y_mix_latent)

    def forward(self, y_mix):
        y_mix_latent = self.encode(y_mix)
        y_mix_decoded = self.decode(y_mix_latent)
        return y_mix_decoded
