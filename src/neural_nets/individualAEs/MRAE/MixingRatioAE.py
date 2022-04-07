import torch.nn as nn


class MixingRatioAE(nn.Module):
    def __init__(self, latent_dim, layer_size):
        super().__init__()

        self.latent_dim = latent_dim
        self.layer_size = layer_size

        self.encoder = nn.Sequential(
            nn.Linear(150, self.layer_size),
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
            nn.Linear(self.layer_size, 150),
            nn.LeakyReLU(),
        )

    def encode(self, y_mix):
        return self.encoder(y_mix)

    def decode(self, y_mix_latent):
        return self.decoder(y_mix_latent)

    def forward(self, y_mix):
        y_mix_latent = self.encode(y_mix)
        y_mix_decoded = self.decode(y_mix_latent)
        return y_mix_decoded
