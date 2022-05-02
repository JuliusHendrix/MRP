import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# own modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = str(Path(script_dir).parents[2])
sys.path.append(src_dir)

from src.neural_nets.NN_utils import tuple_product


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim

        # TODO: kernel size 3 met padding 1? precies halveren
        # ENCODERS
        self.y_mix_ini_encoder = nn.Sequential(  # [batch, 1, 150, 69]
            nn.Conv2d(1, 2, kernel_size=(4, 1), stride=(2, 1), padding=(0, 0)),  # [batch, 2, 74, 69]
            nn.LeakyReLU(),
            # nn.Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # [batch, 2, 74, 69]
            # nn.LeakyReLU(),
            nn.Conv2d(2, 2, kernel_size=(4, 1), stride=(2, 1), padding=(0, 0)),  # [batch, 2, 36, 69]
            nn.LeakyReLU(),
            # nn.Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # [batch, 2, 36, 69]
            # nn.LeakyReLU(),
            nn.Conv2d(2, 4, kernel_size=(4, 1), stride=(2, 1), padding=(0, 0)),  # [batch, 4, 17, 69]
            nn.LeakyReLU(),
            # nn.Conv2d(4, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # [batch, 4, 17, 69]
            # nn.LeakyReLU(),
            nn.Conv2d(4, 4, kernel_size=(3, 1), stride=(2, 1), padding=(0, 0)),  # [batch, 4, 8, 69]
            nn.LeakyReLU()
        )

        self.conv_output_size = (4, 8, 69)
        self.conv_output_product = tuple_product(self.conv_output_size)

        self.flux_wl_encoder = nn.Sequential(    # [b, 2, 2500]
            nn.Unflatten(1, (1, 2)),    # [b, 1, 2, 2500]
            nn.Conv2d(1, 2, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),    # [b, 2, 2, 1250]
            nn.LeakyReLU(),
            nn.Conv2d(2, 2, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),  # [b, 2, 2, 625]
            nn.LeakyReLU(),
            nn.Conv2d(2, 2, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),  # [b, 2, 2, 313]
            nn.LeakyReLU(),
            nn.Conv2d(2, 4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),  # [b, 4, 2, 157]
            nn.LeakyReLU(),
            nn.Flatten(),    # [b, 4*2*157]
            nn.Linear(4*2*157, 4*4*69),
            nn.LeakyReLU(),
            nn.Unflatten(1, (4, 4, 69))    # [b, 4, 4, 69]
        )

        self.Tco_encoder = nn.Sequential(
            nn.Linear(150, 69),
            nn.LeakyReLU()
        )

        self.Pco_encoder = nn.Sequential(
            nn.Linear(150, 69),
            nn.LeakyReLU()
        )

        self.g_encoder = nn.Sequential(
            nn.Linear(150, 69),
            nn.LeakyReLU()
        )

        self.TPg_encoder = nn.Sequential(    # [b, 3, 69]
            nn.Unflatten(1, (1, 3)),    # [b, 1, 3, 69]
            nn.Conv2d(1, 4, kernel_size=(3, 1), stride=(3, 1), padding=(0, 0)),    # [b, 4, 1, 69]
            nn.LeakyReLU()
        )

        self.gravity_encoder = nn.Sequential(    # [b, 1]
            nn.Linear(1, 4*69),
            nn.LeakyReLU(),
            nn.Unflatten(1, (4, 1, 69))    # [b, 4, 1, 69]
        )

        self.latent_conv_output_size = (2, 2, 1103)
        self.latent_conv_output_product = tuple_product(self.latent_conv_output_size)

        self.latent_encoder = nn.Sequential(    # [b, 4, 14, 69]
            # nn.Conv2d(4, 8, kernel_size=(4, 5), stride=(2, 2), padding=(0, 0)),  # [b, 8, 5, 33]
            # nn.LeakyReLU(),
            nn.Flatten(),    # [b, 3312]
            nn.Linear(4*14*69, self.latent_dim),
            nn.LeakyReLU()
        )

        # DECODERS
        self.latent_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 4*14*69),
            nn.LeakyReLU(),
            nn.Unflatten(1, (4, 14, 69)),   # [b, 8, 5, 33]
            # nn.ConvTranspose2d(8, 4, kernel_size=(4, 5), stride=(2, 2), padding=(0, 0)),   # [b, 4, 12, 69]
            # nn.LeakyReLU()
        )

        self.y_mix_ini_decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 4, kernel_size=(3, 1), stride=(2, 1), padding=(0, 0)),
            nn.LeakyReLU(),
            # nn.ConvTranspose2d(4, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.LeakyReLU(),
            nn.ConvTranspose2d(4, 2, kernel_size=(4, 1), stride=(2, 1), padding=(0, 0)),
            nn.LeakyReLU(),
            # nn.ConvTranspose2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.LeakyReLU(),
            nn.ConvTranspose2d(2, 2, kernel_size=(4, 1), stride=(2, 1), padding=(0, 0)),
            nn.LeakyReLU(),
            # nn.ConvTranspose2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.LeakyReLU(),
            nn.ConvTranspose2d(2, 1, kernel_size=(4, 1), stride=(2, 1), padding=(0, 0)),    # [batch, 1, 150, 69]
            nn.LeakyReLU()
        )

        self.flux_wl_decoder = nn.Sequential(    # [b, 4, 4, 69]
            nn.Flatten(),
            nn.Linear(4*4*69, 4*2*157),
            nn.LeakyReLU(),
            nn.Unflatten(1, (4, 2, 157)),    # [b, 4, 2, 157]
            nn.ConvTranspose2d(4, 2, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),    # [b, 2, 2, 313]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2, 2, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),  # [b, 2, 2, 625]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2, 2, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),  # [b, 2, 2, 1250]    # ks=(1, 4) otherwise the shape was not correct...
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2, 1, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),  # [b, 1, 2, 2500]
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1, end_dim=2)    # [b, 2, 2500]
        )

        self.Tco_decoder = nn.Sequential(
            nn.Linear(69, 150),
            nn.LeakyReLU()
        )

        self.Pco_decoder = nn.Sequential(
            nn.Linear(69, 150),
            nn.LeakyReLU()
        )

        self.g_decoder = nn.Sequential(
            nn.Linear(69, 150),
            nn.LeakyReLU()
        )

        self.TPg_decoder = nn.Sequential(    # [b, 4, 1, 69]
            nn.ConvTranspose2d(4, 1, kernel_size=(3, 1), stride=(3, 1), padding=(0, 0)),  # [b, 1, 3, 69],
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1, end_dim=2)    # [b, 3, 69]
        )

        self.gravity_decoder = nn.Sequential(    # [b, 4, 1, 69]
            nn.Flatten(),    # [b, 4*1*69]
            nn.Linear(4*1*69, 1),
            nn.LeakyReLU()
        )

    def encode(self, y_mix_ini, top_flux, wavelengths, Tco, Pco, g, gravity):
        # encode inputs
        y_mix_ini = y_mix_ini[:, None, :, :]  # add image channel [b, 1, 150, 69]
        encoded_y_mix_ini = self.y_mix_ini_encoder(y_mix_ini)  # [batch, 4, 8, 69]

        flux_wl = torch.concat(
            (top_flux[:, None, :],
            wavelengths[:, None, :]),
            dim=1
        )    # [b, 2, 2500]

        encoded_flux_wl = self.flux_wl_encoder(flux_wl)  # [b, 4, 4, 69]

        encoded_Tco = self.Tco_encoder(Tco)  # [b, 69]
        encoded_Pco = self.Tco_encoder(Pco)  # [b, 69]
        encoded_g = self.Tco_encoder(g)  # [b, 69]

        concat_TPg = torch.concat(    # [b, 3, 69]
            (encoded_Tco[:, None, :],
             encoded_Pco[:, None, :],
             encoded_g[:, None, :]),
            dim=1
        )

        encoded_TPg = self.TPg_encoder(concat_TPg)    # [b, 4, 1, 69]

        gravity = gravity[:, None].double()  # [b, 1]
        encoded_gravity = self.gravity_encoder(gravity)  # [b, 4, 1, 69]

        # concatenate encoded inputs
        concat_encoded = torch.cat(
            (encoded_y_mix_ini,
             encoded_flux_wl,
             encoded_TPg,
             encoded_gravity),  # [b, 4, 14, 69]
            dim=2
        )

        # create latent
        latent = self.latent_encoder(concat_encoded)  # [b, latent_dim]

        return latent

    def decode(self, latent):
        # decode latent
        decoded_latent = self.latent_decoder(latent)  # [b, 4, 12, 69]

        # separate out components
        decoded_y_mix_ini = decoded_latent[:, :, 0:8, :]
        decoded_flux_wl = decoded_latent[:, :, 8:12, :]
        decoded_TPg = decoded_latent[:, :, 12:13, :]
        decoded_gravity = decoded_latent[:, :, 13:14, :]

        # retreive original shapes
        gravity = self.gravity_decoder(decoded_gravity)

        TPg = self.TPg_decoder(decoded_TPg)   # [b, 3, 69]
        decoded_Tco = TPg[:, 0, :].flatten(start_dim=1, end_dim=-1)    # [b, 69]
        Tco = self.Tco_decoder(decoded_Tco)

        decoded_Pco = TPg[:, 1, :].flatten(start_dim=1, end_dim=-1)
        Pco = self.Pco_decoder(decoded_Pco)

        decoded_g = TPg[:, 2, :].flatten(start_dim=1, end_dim=-1)
        g = self.g_decoder(decoded_g)

        flux_wl = self.flux_wl_decoder(decoded_flux_wl)
        top_flux = flux_wl[:, 0, :]
        wavelengths = flux_wl[:, 1, :]

        y_mix_ini = self.y_mix_ini_decoder(decoded_y_mix_ini).flatten(start_dim=1, end_dim=2)    # [batch, 1, 150, 69]

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
        latent = self.encode(**inputs)

        # decode
        outputs =  self.decode(latent)

        return outputs
