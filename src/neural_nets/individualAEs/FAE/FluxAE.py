import torch.nn as nn


class FluxAE(nn.Module):
    def __init__(self, latent_dim, layer_size):
        super().__init__()

        self.latent_dim = latent_dim
        self.layer_size = layer_size

        self.encoder = nn.Sequential(
            nn.Linear(2500, self.layer_size),
            nn.LeakyReLU(),
            nn.Linear(self.layer_size, self.layer_size),
            nn.LeakyReLU(),
            nn.Linear(self.layer_size, self.layer_size),
            nn.LeakyReLU(),
            nn.Linear(self.layer_size, self.layer_size),
            nn.LeakyReLU(),
            nn.Linear(self.layer_size, self.layer_size),
            nn.LeakyReLU(),
            nn.Linear(self.layer_size, self.layer_size),
            nn.LeakyReLU(),
            nn.Linear(self.layer_size, self.latent_dim),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.layer_size),
            nn.LeakyReLU(),
            nn.Linear(self.layer_size, self.layer_size),
            nn.LeakyReLU(),
            nn.Linear(self.layer_size, self.layer_size),
            nn.LeakyReLU(),
            nn.Linear(self.layer_size, self.layer_size),
            nn.LeakyReLU(),
            nn.Linear(self.layer_size, self.layer_size),
            nn.LeakyReLU(),
            nn.Linear(self.layer_size, self.layer_size),
            nn.LeakyReLU(),
            nn.Linear(self.layer_size, 2500),
            nn.LeakyReLU(),
        )

    def encode(self, flux):
        return self.encoder(flux)

    def decode(self, flux_latent):
        return self.decoder(flux_latent)

    def forward(self, flux):
        flux_latent = self.encode(flux)
        flux_decoded = self.decode(flux_latent)
        return flux_decoded
