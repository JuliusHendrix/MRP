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

        # ENCODERS
        self.y_mix_ini_encoder = nn.Sequential(  # [batch, 1, 150, 69]
            nn.Conv2d(1, 2, kernel_size=(4, 1), stride=(2, 1), padding=(0, 0)),  # [batch, 2, 74, 69]
            nn.LeakyReLU(),
            nn.Conv2d(2, 2, kernel_size=(4, 1), stride=(2, 1), padding=(0, 0)),  # [batch, 2, 36, 69]
            nn.LeakyReLU(),
            nn.Conv2d(2, 4, kernel_size=(4, 1), stride=(2, 1), padding=(0, 0)),  # [batch, 4, 17, 69]
            nn.LeakyReLU(),
            nn.Conv2d(4, 4, kernel_size=(3, 1), stride=(2, 1), padding=(0, 0)),  # [batch, 4, 8, 69]
            nn.LeakyReLU(),
            nn.Flatten()  # [b, conv_output_product]
        )

        # TODO: calculate automatically
        self.conv_output_size = (4, 8, 69)
        self.conv_output_product = tuple_product(self.conv_output_size)

        # TODO: maybe instead of linear layers to conv_output_size, to 69 and concat with compressed y_mix?

        self.top_flux_encoder = nn.Sequential(
            nn.Linear(2500, self.conv_output_product),
            nn.LeakyReLU()
        )

        self.Tco_encoder = nn.Sequential(
            nn.Linear(150, self.conv_output_product),
            nn.LeakyReLU()
        )

        self.Pco_encoder = nn.Sequential(
            nn.Linear(150, self.conv_output_product),
            nn.LeakyReLU()
        )

        self.g_encoder = nn.Sequential(
            nn.Linear(150, self.conv_output_product),
            nn.LeakyReLU()
        )

        self.gravity_encoder = nn.Sequential(
            nn.Linear(1, self.conv_output_product),
            nn.LeakyReLU()
        )

        self.latent_conv_output_size = (2, 2, 1103)
        self.latent_conv_output_product = tuple_product(self.latent_conv_output_size)

        self.latent_encoder = nn.Sequential(    # [b, 1, 6, conv_output_size]
            nn.Conv2d(1, 2, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),  # [b, 2, 2, 1103]
            nn.LeakyReLU(),
            nn.Flatten(),    # [b, 2*2*827]
            nn.Linear(self.latent_conv_output_product, self.latent_dim),
            nn.LeakyReLU()
        )

        # DECODERS
        self.latent_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_conv_output_product),
            nn.LeakyReLU(),
            nn.Unflatten(1, self.latent_conv_output_size),
            nn.ConvTranspose2d(2, 1, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),  # [b, 1, 6, conv_output_size]
            nn.LeakyReLU()
        )

        self.y_mix_ini_decoder = nn.Sequential(
            nn.Unflatten(1, self.conv_output_size),
            nn.ConvTranspose2d(4, 4, kernel_size=(3, 1), stride=(2, 1), padding=(0, 0)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4, 2, kernel_size=(4, 1), stride=(2, 1), padding=(0, 0)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2, 2, kernel_size=(4, 1), stride=(2, 1), padding=(0, 0)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2, 1, kernel_size=(4, 1), stride=(2, 1), padding=(0, 0)),
            nn.LeakyReLU()
        )

        self.top_flux_decoder = nn.Sequential(
            nn.Linear(self.conv_output_product, 2500),
            nn.LeakyReLU()
        )

        self.Tco_decoder = nn.Sequential(
            nn.Linear(self.conv_output_product, 150),
            nn.LeakyReLU()
        )

        self.Pco_decoder = nn.Sequential(
            nn.Linear(self.conv_output_product, 150),
            nn.LeakyReLU()
        )

        self.g_decoder = nn.Sequential(
            nn.Linear(self.conv_output_product, 150),
            nn.LeakyReLU()
        )

        self.gravity_decoder = nn.Sequential(
            nn.Linear(self.conv_output_product, 1),
            nn.LeakyReLU()
        )

    def encode(self, y_mix_ini, top_flux, Tco, Pco, g, gravity):
        # encode inputs
        y_mix_ini = y_mix_ini[:, None, :, :]  # add image channel [b, 1, 150, 69]
        encoded_y_mix_ini = self.y_mix_ini_encoder(y_mix_ini)  # [b, conv_output_product]

        encoded_top_flux = self.top_flux_encoder(top_flux)  # [b, conv_output_product]
        encoded_Tco = self.Tco_encoder(Tco)  # [b, conv_output_product]
        encoded_Pco = self.Tco_encoder(Pco)  # [b, conv_output_product]
        encoded_g = self.Tco_encoder(g)  # [b, conv_output_product]

        gravity = gravity[:, None].double()  # [b, 1]
        encoded_gravity = self.gravity_encoder(gravity)  # [b, conv_output_product]

        # concatenate encoded inputs
        concat_encoded = torch.cat(
            (encoded_y_mix_ini[:, None, :],
             encoded_top_flux[:, None, :],
             encoded_Tco[:, None, :],
             encoded_Pco[:, None, :],
             encoded_g[:, None, :],
             encoded_gravity[:, None, :]),  # [b, 6, conv_output_product]
            dim=1
        )

        # create latent
        concat_encoded = concat_encoded[:, None, :, :]  # [b, 1, 6, conv_output_product]
        latent = self.latent_encoder(concat_encoded)  # [b, latent_dim]

        return latent

    def decode(self, latent):
        # decode latent
        decoded_latent = self.latent_decoder(latent)  # [b, 1, 6, conv_output_product]
        b, _, concats, conv_length = decoded_latent.shape

        # remove extra dimension
        decoded_latent = decoded_latent.reshape(shape=(b, concats, conv_length))  # [b, 6, 952]

        # separate out components
        decoded_y_mix_ini, decoded_top_flux, decoded_Tco, decoded_Pco, decoded_g, decoded_gravity = torch.split(
            decoded_latent, split_size_or_sections=1, dim=1  # 6x [b, 1, 952]
        )

        # decode components
        decoded_y_mix_ini = decoded_y_mix_ini.reshape(shape=(b, conv_length))
        decoded_y_mix_ini = self.y_mix_ini_decoder(decoded_y_mix_ini)  # [b, 1, 150, 69]
        decoded_y_mix_ini = decoded_y_mix_ini.reshape(shape=(b, 150, 69))

        decoded_top_flux = decoded_top_flux.reshape(shape=(b, conv_length))
        decoded_top_flux = self.top_flux_decoder(decoded_top_flux)

        decoded_Tco = decoded_Tco.reshape(shape=(b, conv_length))
        decoded_Tco = self.Tco_decoder(decoded_Tco)

        decoded_Pco = decoded_Pco.reshape(shape=(b, conv_length))
        decoded_Pco = self.Pco_decoder(decoded_Pco)

        decoded_g = decoded_g.reshape(shape=(b, conv_length))
        decoded_g = self.g_decoder(decoded_g)

        decoded_gravity = decoded_gravity.reshape(shape=(b, conv_length))
        decoded_gravity = self.gravity_decoder(decoded_gravity)

        return decoded_y_mix_ini, decoded_top_flux, decoded_Tco, decoded_Pco, decoded_g, decoded_gravity

    def forward(self, inputs):
        # encode
        latent = self.encode(**inputs)

        # decode
        decoded_y_mix_ini, decoded_top_flux, decoded_Tco, decoded_Pco, decoded_g, decoded_gravity = self.decode(latent)

        outputs = {
            'y_mix_ini': decoded_y_mix_ini,
            'top_flux': decoded_top_flux,
            'Tco': decoded_Tco,
            'Pco': decoded_Pco,
            'g': decoded_g,
            'gravity': decoded_gravity
        }

        return outputs
