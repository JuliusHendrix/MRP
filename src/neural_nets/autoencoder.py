import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # ENCODERS
        # TODO: pooling?
        self.y_mix_ini_encoder = nn.Sequential(    # [batch, 1, 150, 69]
            nn.Conv2d(1, 2, kernel_size=(4,3), stride=(2,2), padding=(0,0)),    # [batch, 2, 74, 34]
            nn.LeakyReLU(),
            nn.Conv2d(2, 4, kernel_size=(4,4), stride=(2,2), padding=(0,0)),  # [batch, 4, 36, 16]
            nn.LeakyReLU(),
            nn.Conv2d(4, 8, kernel_size=(4,4), stride=(2,2), padding=(0,0)),  # [batch, 8, 17, 7]
            nn.LeakyReLU(),
            nn.Flatten()    # [b, 952]
        )

        self.top_flux_encoder = nn.Sequential(
            nn.Linear(2500, 952),
            nn.LeakyReLU()
        )

        self.Tco_encoder = nn.Sequential(
            nn.Linear(150, 952),
            nn.LeakyReLU()
        )

        self.Pco_encoder = nn.Sequential(
            nn.Linear(150, 952),
            nn.LeakyReLU()
        )

        self.g_encoder = nn.Sequential(
            nn.Linear(150, 952),
            nn.LeakyReLU()
        )

        self.gravity_encoder = nn.Sequential(
            nn.Linear(1, 952),
            nn.LeakyReLU()
        )

        self.latent_encoder = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=(4,4), stride=(2,2), padding=(0,0)),    # [b, 2, 2, 475]
            nn.LeakyReLU(),
            # nn.Flatten(),    # [b, 543]
            # nn.Linear(543, 512),
            # nn.ReLU()
        )

        # DECODERS
        self.latent_decoder = nn.Sequential(
            nn.ConvTranspose2d(2, 1, kernel_size=(4,4), stride=(2,2), padding=(0,0)),    # [b, 1, 6, 952]
            nn.LeakyReLU()
        )

        self.y_mix_ini_decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 4, kernel_size=(4,4), stride=(2,2), padding=(0,0)),    # [b, 4, 36, 16]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4, 2, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),  # [b, 2, 74, 34]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2, 1, kernel_size=(4, 3), stride=(2, 2), padding=(0, 0)),  # [b, 1, 150, 69]
            nn.LeakyReLU()
        )

        self.top_flux_decoder = nn.Sequential(
            nn.Linear(952, 2500),    # [b, 2500]
            nn.LeakyReLU()
        )

        self.Tco_decoder = nn.Sequential(
            nn.Linear(952, 150),
            nn.LeakyReLU()
        )

        self.Pco_decoder = nn.Sequential(
            nn.Linear(952, 150),
            nn.LeakyReLU()
        )

        self.g_decoder = nn.Sequential(
            nn.Linear(952, 150),
            nn.LeakyReLU()
        )

        self.gravity_decoder = nn.Sequential(
            nn.Linear(952, 1),    # [b, 1]
            nn.LeakyReLU()
        )

    def encode(self, y_mix_ini, top_flux, Tco, Pco, g, gravity):
        # encode inputs
        y_mix_ini = y_mix_ini[:, None, :, :]  # add image channel [b, 1, 150, 69]
        encoded_y_mix_ini = self.y_mix_ini_encoder(y_mix_ini)  # [b, 952]

        encoded_top_flux = self.top_flux_encoder(top_flux)  # [b, 952]
        encoded_Tco = self.Tco_encoder(Tco)  # [b, 952]
        encoded_Pco = self.Tco_encoder(Pco)  # [b, 952]
        encoded_g = self.Tco_encoder(g)  # [b, 952]

        gravity = gravity[:, None].double()    # [b, 1]
        encoded_gravity = self.gravity_encoder(gravity)  # [b, 952]

        # concatenate encoded inputs
        concat_encoded = torch.cat(
            (encoded_y_mix_ini[:, None, :],
             encoded_top_flux[:, None, :],
             encoded_Tco[:, None, :],
             encoded_Pco[:, None, :],
             encoded_g[:, None, :],
             encoded_gravity[:, None, :]),  # [b, 6, 952]
            dim=1
        )

        # create latent
        concat_encoded = concat_encoded[:, None, :, :]  # [b, 1, 6, 952]
        latent = self.latent_encoder(concat_encoded)  # [b, 2, 2, 475]

        return latent

    def decode(self, latent):
        # decode latent
        decoded_latent = self.latent_decoder(latent)  # [b, 1, 6, 952]
        b, _, concats, conv_length = decoded_latent.shape

        # remove extra dimension
        decoded_latent = decoded_latent.reshape(shape=(b, concats, conv_length))    # [b, 6, 952]

        # separate out components
        decoded_y_mix_ini, decoded_top_flux, decoded_Tco, decoded_Pco, decoded_g, decoded_gravity = torch.split(
            decoded_latent, split_size_or_sections=1, dim=1  # 6x [b, 1, 952]
        )

        # decode components
        decoded_y_mix_ini = decoded_y_mix_ini.reshape(shape=(b, 8, 17, 7))
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

    def forward(self, y_mix_ini, top_flux, Tco, Pco, g, gravity):
        # encode
        latent = self.encode(y_mix_ini, top_flux, Tco, Pco, g, gravity)

        # decode
        decoded_y_mix_ini, decoded_top_flux, decoded_Tco, decoded_Pco, decoded_g, decoded_gravity = self.decode(latent)

        return decoded_y_mix_ini, decoded_top_flux, decoded_Tco, decoded_Pco, decoded_g, decoded_gravity


def calculate_padding(input_shape, kernel_size, stride):
    H = input_shape[0]
    W = input_shape[1]

    def output_dim(input_dim, pad, k_s, s):
        output_d = (input_dim - k_s + 2*pad) / s + 1
        return output_d

    padding = [-1,-1]

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
          f'output_shape: {(int(H_out), int(W_out))}\n'
          f'kernel_size: {kernel_size}\n'
          f'stride: {stride}\n'
          f'padding: {padding}')


if __name__ == "__main__":
    calculate_padding(input_shape=(6, 952),
                      kernel_size=(4,4),
                      stride=(2,2))