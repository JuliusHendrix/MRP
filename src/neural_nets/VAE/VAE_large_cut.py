import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

num_species = 45


# resources:
# https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b
# https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
# https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
class VariationalAutoEncoder(nn.Module):
    def __init__(self, device, latent_dim):
        super().__init__()

        self.device = device
        self.latent_dim = latent_dim

        # for reparameterization trick
        self.N = torch.distributions.Normal(0, 1)
        if self.device != torch.device('cpu'):
            self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()

        # ENCODERS
        self.y_mix_ini_encoder = nn.Sequential(  # [batch, 1, 150, num_species]
            nn.Conv2d(1, 2, kernel_size=(4, 1), stride=(2, 1), padding=(0, 0)),  # [batch, 2, 74, num_species]
            nn.LeakyReLU(),
            nn.Conv2d(2, 2, kernel_size=(4, 1), stride=(2, 1), padding=(0, 0)),  # [batch, 2, 36, num_species]
            nn.LeakyReLU()
        )

        self.flux_wl_encoder = nn.Sequential(    # [b, 2, 2500]
            nn.Unflatten(1, (1, 2)),    # [b, 1, 2, 2500]
            nn.Conv2d(1, 2, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),    # [b, 2, 2, 1250]
            nn.LeakyReLU(),
            nn.Conv2d(2, 2, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),  # [b, 2, 2, 625]
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(2*2*625, 2*18*num_species),
            nn.LeakyReLU(),
            nn.Unflatten(1, (2, 18, num_species))    # [b, 2, 18, num_species]
        )

        self.Tco_encoder = nn.Sequential(
            nn.Linear(150, num_species),
            nn.LeakyReLU()
        )

        self.Pco_encoder = nn.Sequential(
            nn.Linear(150, num_species),
            nn.LeakyReLU()
        )

        self.g_encoder = nn.Sequential(
            nn.Linear(150, num_species),
            nn.LeakyReLU()
        )

        self.TPg_encoder = nn.Sequential(    # [b, 3, num_species]
            nn.Unflatten(1, (1, 3)),    # [b, 1, 3, num_species]
            nn.Conv2d(1, 2, kernel_size=(3, 1), stride=(3, 1), padding=(0, 0)),    # [b, 2, 1, num_species]
            nn.LeakyReLU()
        )

        self.gravity_encoder = nn.Sequential(    # [b, 1]
            nn.Linear(1, 2*num_species),
            nn.LeakyReLU(),
            nn.Unflatten(1, (2, 1, num_species))    # [b, 4, 1, num_species]
        )

        self.latent_encoder = nn.Sequential(    # [b, 2, 56, num_species]
            nn.Flatten(),    # [b, 7728]
            nn.Linear(2*56*num_species, self.latent_dim),
            nn.LeakyReLU()
        )

        self.mu_encoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        self.sigma_encoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        # DECODERS
        self.latent_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 2*56*num_species),
            nn.LeakyReLU(),
            nn.Unflatten(1, (2, 56, num_species)),
        )

        self.y_mix_ini_decoder = nn.Sequential(
            nn.ConvTranspose2d(2, 2, kernel_size=(4, 1), stride=(2, 1), padding=(0, 0)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2, 1, kernel_size=(4, 1), stride=(2, 1), padding=(0, 0)),    # [batch, 1, 150, num_species]
            nn.LeakyReLU(),
        )

        self.flux_wl_decoder = nn.Sequential(    # [b, 2, 18, num_species]
            nn.Flatten(),
            nn.Linear(2*18*num_species, 2*2*625),
            nn.LeakyReLU(),
            nn.Unflatten(1, (2, 2, 625)),    # [b, 4, 2, 157]
            nn.ConvTranspose2d(2, 2, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),  # [b, 2, 2, 1250]    # ks=(1, 4) otherwise the shape was not correct...
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2, 1, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),  # [b, 1, 2, 2500]
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1, end_dim=2),    # [b, 2, 2500]
        )

        self.Tco_decoder = nn.Sequential(
            nn.Linear(num_species, 150),
            nn.LeakyReLU(),
        )

        self.Pco_decoder = nn.Sequential(
            nn.Linear(num_species, 150),
            nn.LeakyReLU(),
        )

        self.g_decoder = nn.Sequential(
            nn.Linear(num_species, 150),
            nn.LeakyReLU(),
        )

        self.TPg_decoder = nn.Sequential(    # [b, 2, 1, num_species]
            nn.ConvTranspose2d(2, 1, kernel_size=(3, 1), stride=(3, 1), padding=(0, 0)),  # [b, 1, 3, num_species],
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1, end_dim=2)    # [b, 3, num_species]
        )

        self.gravity_decoder = nn.Sequential(    # [b, 2, 1, num_species]
            nn.Flatten(),    # [b, 4*1*num_species]
            nn.Linear(2*1*num_species, 1),
            nn.LeakyReLU()
        )

    def encode(self, y_mix_ini, top_flux, wavelengths, Tco, Pco, g, gravity):
        # encode inputs
        y_mix_ini = y_mix_ini[:, None, :, :]  # add image channel [b, 1, 150, num_species]
        encoded_y_mix_ini = self.y_mix_ini_encoder(y_mix_ini)  # [batch, 2, 36, num_species]

        flux_wl = torch.concat(
            (top_flux[:, None, :],
            wavelengths[:, None, :]),
            dim=1
        )    # [b, 2, 2500]

        encoded_flux_wl = self.flux_wl_encoder(flux_wl)  # [b, 2, 18, num_species]

        encoded_Tco = self.Tco_encoder(Tco)  # [b, num_species]
        encoded_Pco = self.Tco_encoder(Pco)  # [b, num_species]
        encoded_g = self.Tco_encoder(g)  # [b, num_species]

        concat_TPg = torch.concat(    # [b, 3, num_species]
            (encoded_Tco[:, None, :],
             encoded_Pco[:, None, :],
             encoded_g[:, None, :]),
            dim=1
        )

        encoded_TPg = self.TPg_encoder(concat_TPg)    # [b, 2, 1, num_species]

        gravity = gravity[:, None].double()  # [b, 1]
        encoded_gravity = self.gravity_encoder(gravity)  # [b, 2, 1, num_species]

        # concatenate encoded inputs
        concat_encoded = torch.cat(
            (encoded_y_mix_ini,
             encoded_flux_wl,
             encoded_TPg,
             encoded_gravity),  # [b, 2, 56, num_species]
            dim=2
        )

        # create latent
        latent = self.latent_encoder(concat_encoded)  # [b, latent_dim]

        mu = self.mu_encoder(latent)
        sigma = torch.exp(self.sigma_encoder(latent))    # exp to make sure sigma is positive
        # sigma = self.sigma_encoder(flat_latent)

        # reparametrization trick
        z = mu + sigma * self.N.sample(mu.shape)    # sample from distribution

        # KL divergence between normal distributions
        # from: http://allisons.org/ll/MML/KL/Normal/
        kl_divs = (mu**2 + sigma**2 - 1.) / 2. - torch.log(sigma)
        kl_div = torch.sum(kl_divs)

        return z, kl_div

    def decode(self, z):
        # decode latent
        decoded_latent = self.latent_decoder(z)  # [b, 2, 56, num_species]

        # separate out components
        decoded_y_mix_ini = decoded_latent[:, :, 0:36, :]
        decoded_flux_wl = decoded_latent[:, :, 36:54, :]
        decoded_TPg = decoded_latent[:, :, 54:55, :]
        decoded_gravity = decoded_latent[:, :, 55:56, :]

        # retreive original shapes
        gravity = self.gravity_decoder(decoded_gravity)

        TPg = self.TPg_decoder(decoded_TPg)   # [b, 3, num_species]
        decoded_Tco = TPg[:, 0, :].flatten(start_dim=1, end_dim=-1)    # [b, num_species]
        Tco = self.Tco_decoder(decoded_Tco)

        decoded_Pco = TPg[:, 1, :].flatten(start_dim=1, end_dim=-1)
        Pco = self.Pco_decoder(decoded_Pco)

        decoded_g = TPg[:, 2, :].flatten(start_dim=1, end_dim=-1)
        g = self.g_decoder(decoded_g)

        flux_wl = self.flux_wl_decoder(decoded_flux_wl)
        top_flux = flux_wl[:, 0, :]
        wavelengths = flux_wl[:, 1, :]

        y_mix_ini = self.y_mix_ini_decoder(decoded_y_mix_ini).flatten(start_dim=1, end_dim=2)    # [batch, 1, 150, num_species]

        outputs = {
            'y_mix_ini': y_mix_ini,
            'top_flux': top_flux,
            'wavelengths': wavelengths,
            'Tco': Tco,
            'Pco': Pco,
            'g': g,
            'gravity': gravity
        }

        return outputs

    def forward(self, inputs):
        # encode
        z, kl_div = self.encode(**inputs)

        # decode
        outputs = self.decode(z)

        metrics = {
            'kl_div': kl_div
        }

        return outputs, metrics
