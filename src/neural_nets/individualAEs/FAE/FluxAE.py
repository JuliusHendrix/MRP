import torch.nn as nn


class FluxAE(nn.Module):
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
            nn.Linear(2500, self.layer_size),
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
            nn.Linear(self.layer_size, 2500),
            self.activation_function(),
        )

    def encode(self, flux):
        return self.encoder(flux)

    def decode(self, flux_latent):
        return self.decoder(flux_latent)

    def forward(self, flux):
        flux_latent = self.encode(flux)
        flux_decoded = self.decode(flux_latent)
        return flux_decoded
