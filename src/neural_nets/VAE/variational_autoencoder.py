import torch
import torch.nn as nn


# resources:
# https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b
# https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
# https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
class VariationalAutoEncoder(nn.Module):
    def __init__(self, device, latent_dim):
        super().__init__()

        self.device = device

        # for reparameterization trick
        self.N = torch.distributions.Normal(0, 1)
        if self.device != 'cpu':
            self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()

        # ENCODERS
        # TODO: pooling?
        self.y_mix_ini_encoder = nn.Sequential(  # [batch, 1, 150, 69]
            nn.Conv2d(1, 2, kernel_size=(4, 3), stride=(2, 2), padding=(0, 0)),  # [batch, 2, 74, 34]
            nn.LeakyReLU(),
            nn.Conv2d(2, 4, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),  # [batch, 4, 36, 16]
            nn.LeakyReLU(),
            nn.Conv2d(4, 8, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),  # [batch, 8, 17, 7]
            nn.LeakyReLU(),
            nn.Flatten()  # [b, 8*17*7]
        )

        self.top_flux_encoder = nn.Sequential(
            nn.Linear(2500, 8*17*7),
            nn.LeakyReLU()
        )

        self.Tco_encoder = nn.Sequential(
            nn.Linear(150, 8*17*7),
            nn.LeakyReLU()
        )

        self.Pco_encoder = nn.Sequential(
            nn.Linear(150, 8*17*7),
            nn.LeakyReLU()
        )

        self.g_encoder = nn.Sequential(
            nn.Linear(150, 8*17*7),
            nn.LeakyReLU()
        )

        self.gravity_encoder = nn.Sequential(
            nn.Linear(1, 8*17*7),
            nn.LeakyReLU()
        )

        self.latent_encoder = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),  # [b, 2, 2, 475]
            nn.LeakyReLU(),
            nn.Conv2d(2, 2, kernel_size=(2, 2), stride=(2, 1), padding=(0, 0)),  # [b, 2, 1, 474]
            nn.LeakyReLU(),
            nn.Flatten(),    # [b, 2*1*474]
        )

        self.mu_encoder = nn.Sequential(
            nn.Linear(2*1*474, latent_dim)
        )

        self.sigma_encoder = nn.Sequential(
            nn.Linear(2*1*474, latent_dim),
            # nn.Softplus()    # standard deviation needs to be positive
        )

        # DECODERS
        self.latent_decoder = nn.Sequential(
            nn.Linear(latent_dim, 2*1*474),    # [b, 2*1*474]
            nn.LeakyReLU(),
            nn.Unflatten(dim=-1, unflattened_size=(2, 1, 474)),    # [b, 2, 1, 474]
            nn.ConvTranspose2d(2, 2, kernel_size=(2, 2), stride=(2, 1), padding=(0, 0)),  # [b, 2, 2, 475]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2, 1, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),  # [b, 1, 6, 952]
            nn.LeakyReLU()
        )

        self.y_mix_ini_decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 4, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),  # [b, 4, 36, 16]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4, 2, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),  # [b, 2, 74, 34]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2, 1, kernel_size=(4, 3), stride=(2, 2), padding=(0, 0)),  # [b, 1, 150, 69]
            nn.LeakyReLU()
        )

        self.top_flux_decoder = nn.Sequential(
            nn.Linear(952, 2500),  # [b, 2500]
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
            nn.Linear(952, 1),  # [b, 1]
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

        gravity = gravity[:, None].double()  # [b, 1]
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
        flat_latent = self.latent_encoder(concat_encoded)  # [b, 2*1*474]

        mu = self.mu_encoder(flat_latent)
        sigma = torch.exp(self.sigma_encoder(flat_latent))    # exp to make sure sigma is positive
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
        decoded_latent = self.latent_decoder(z)  # [b, 1, 6, 952]
        b, _, concats, conv_length = decoded_latent.shape

        # remove extra dimension
        decoded_latent = decoded_latent.reshape(shape=(b, concats, conv_length))  # [b, 6, 952]

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

    def forward(self, inputs):
        # encode
        z, kl_div = self.encode(**inputs)

        # decode
        decoded_y_mix_ini, decoded_top_flux, decoded_Tco, decoded_Pco, decoded_g, decoded_gravity = self.decode(z)

        outputs = {
            'y_mix_ini': decoded_y_mix_ini,
            'top_flux': decoded_top_flux,
            'Tco': decoded_Tco,
            'Pco': decoded_Pco,
            'g': decoded_g,
            'gravity': decoded_gravity
        }

        metrics = {
            'kl_div': kl_div
        }

        return outputs, metrics
