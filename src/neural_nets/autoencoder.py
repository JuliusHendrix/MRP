import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # encoders
        # TODO: pooling?
        self.height_encoder = nn.Sequential(    # [batch, 1, 150, 72]
            nn.Conv2d(1, 2, kernel_size=(4,4), stride=(2,2), padding=(0,0)),    # [batch, 2, 74, 35]
            nn.ReLU(),
            nn.Conv2d(2, 4, kernel_size=(4,3), stride=(2,2), padding=(0,0)),  # [batch, 4, 36, 17]
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=(4,3), stride=(2,2), padding=(0,0)),  # [batch, 8, 17, 8]
            nn.ReLU(),
            nn.Flatten()
        )

        self.flux_encoder = nn.Sequential(
            nn.Linear(2500, 1088),
            nn.ReLU()
        )

        self.constants_encoder = nn.Sequential(
            nn.Linear(1, 1088),
            nn.ReLU()
        )

        self.latent_encoder = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3,4), stride=(1,2), padding=(0,0)),    # [b, 1, 1, 543]
            nn.ReLU(),
            nn.Flatten(),    # [b, 543]
            nn.Linear(543, 512),
            nn.ReLU()
        )

        # decoders
        self.latent_decoder_linear = nn.Sequential(
            nn.Linear(512, 543),
            nn.ReLU()
        )

        self.latent_decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=(3,4), stride=(1,2), padding=(0,0)),    # [b, 1, 3, 1088]
            nn.ReLU()
        )

    def encode(self, height_arr, top_flux, const):
        height_arr = height_arr[:, None, :, :]  # add image channel [batch, 1, 150, 72]
        encoded_height_arr = self.height_encoder(height_arr)  # [b, 1088]

        encoded_flux = self.flux_encoder(top_flux)  # [b, 1088]

        const = const[:, None].double()    # [b, 1]
        encoded_const = self.constants_encoder(const)  # [b, 1088]

        concat_encoded = torch.cat(
            (encoded_height_arr[:, None, :], encoded_flux[:, None, :], encoded_const[:, None, :]),  # [b, 3, 1088]
            dim=1
        )

        concat_encoded = concat_encoded[:, None, :, :]  # [b, 1, 3, 1088]
        latent = self.latent_encoder(concat_encoded)  # [b, 512]

        return latent

    def decode(self, latent):
        concat_encoded_linear = self.latent_decoder_linear(latent)

        return 0,0,0

    def forward(self, height_arr, top_flux, const):
        # encode
        latent = self.encode(height_arr, top_flux, const)

        # decode
        dec_height_arr, dec_top_flux, dec_const = self.decode(latent)

        return dec_height_arr, dec_top_flux, dec_const


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
          f'output_shape: {(H_out, W_out)}\n'
          f'kernel_size: {kernel_size}\n'
          f'stride: {stride}\n'
          f'padding: {padding}')


if __name__ == "__main__":
    calculate_padding(input_shape=(3, 1088),
                      kernel_size=(3,4),
                      stride=(1,2))