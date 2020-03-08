import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn import functional as F

import IPython
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
from utils import mnist, initialize_weights, show_images_square
from utils import interrupted, enumerate_cycle


class Encoder(nn.Module):
    def __init__(self, latent_dim, device):
        """
        The input is (None, 1, 28, 28),
        """
        super().__init__()
        self.device = device

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=(1, 1)),  # (16, 14, 14)
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=(1, 1)),  # (32, 7, 7)
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=(1, 1)),  # (64, 4, 4)
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(64),

            Flatten()
        )
        self.mu = nn.Linear(1024, latent_dim)
        self.log_var = nn.Linear(1024, latent_dim)

        self.apply(initialize_weights)

    def forward(self, x):
        """
        Input
        - x: Tensor of (Batch, 1, 28, 28)

        Output:
        - z: sample from the latent variable
        - mu
        - std
        """
        shared_encoded = self.model(x)
        mu = self.mu(shared_encoded)
        log_var = self.log_var(shared_encoded)
        std = torch.exp(log_var / 2)

        return self.sample(mu, std), mu, log_var

    def sample(self, mu, std):
        """
        The reparametrization trick

        Output:
        - x - tensor of (None, latent_dim), sampled from the encoding
        """
        epsilon = torch.randn(mu.size()).to(self.device)

        return mu + epsilon * std


class DC_Generator(nn.Module):
    def __init__(self, noise_dim):
        super().__init__()

        # Linear -> Linear -> ConvTrans2D -> ConvTrans2D
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 7 * 7 * 128),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(7 * 7 * 128),
            Unflatten(-1, 128, 7, 7),

            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 1, kernel_size=(4, 4), stride=2, padding=1),

            nn.Tanh()
        )

        self.apply(initialize_weights)

    def forward(self, x):
        """
        Input:
        - x: tensor of (None, noise_dim)

        Output:
        - image: tensor of (None, 1, 28, 28)
        """
        return self.model(x)


class DC_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv2D -> Conv2D -> Flatten -> Fully connected -> Fully connected
        self.part_1 = nn.Sequential(
            Unflatten(-1, 1, 28, 28),
            nn.Conv2d(1, 32, kernel_size=(5, 5), stride=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            Flatten(),
        )

        self.flatten = Flatten()

        self.part_2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1)
        )

        self.apply(initialize_weights)

    def forward(self, x):
        """
        Input:
        - x: tensor of (None, 1, 28, 28)

        Output:
        - prediction_logits: tensor of (None, 1) (no sigmoid!!!)
        - l_layer (None, -1) - values flattened for th el-th layer (used to compute the loss)
        """
        part_1 = self.part_1(x)
        l_layer = self.flatten(part_1)
        part_2 = self.part_2(part_1)

        return part_2, l_layer


class Reshaper(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.reshape(x, self.dim)


class BetaVAE_course(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(9216, 128),
            nn.LeakyReLU()
        )

        self.fc_mu = nn.Linear(128, latent_dim)
        # variance is always positive, thus this layer encodes log(variance).
        # we compute the std in reparameterize()
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1728),
            nn.LeakyReLU(),
            Reshaper((-1, 12, 12, 12)),
            nn.Conv2d(12, 36, 3, 1),
            nn.LeakyReLU(),
            Reshaper((-1, 4, 30, 30)),
            nn.Conv2d(4, 4, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(4, 1, 1, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        shared_latent = F.relu(self.shared_encoder(x))
        mu = self.fc_mu(shared_latent)
        logvar = self.fc_logvar(shared_latent)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        return self.decoder(z), mu, logvar, z

    @staticmethod
    def loss_function(reconstructed_x, x, mu, logvar, beta=1):
        BCE = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')

        KLD = beta * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD


class BetaVAE(nn.Module):
    def __init__(self, latent_dim, device):
        super().__init__()

        self.encoder = Encoder(latent_dim, device)
        self.decoder = DC_Generator(latent_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z, mu, log_var = self.encoder(x)
        output = self.sigmoid(self.decoder(z))

        return output, mu, log_var

    @staticmethod
    def loss_function(x_input, x_recons, mu, logvar, beta=1):
        BCE = F.binary_cross_entropy(x_recons, x_input, reduction='sum')

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + beta * KLD


def sample_noise(batch_size, dim, dtype, device):
    """
    Generate a PyTorch tensor of uniform random noise.

    Input:
    - dim: Integer giving the dimension of noise to generate

    Output:
    - PyTorch Tensor of shape (batch_size, dim) containing uniform random noise in the range (-1, 1)
    """
    return torch.rand([batch_size, dim], dtype=dtype, device=device) * 2 - 1


def discriminator_loss(logits_real, logits_fake, dtype, device):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """

    loss = bce_loss(logits_real, torch.ones(logits_real.size(), dtype=dtype, device=device)) \
           + bce_loss(logits_fake, torch.zeros(logits_real.size(), dtype=dtype, device=device))

    return loss


def generator_loss(logits_fake, dtype, device):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """

    loss = bce_loss(logits_fake, torch.ones(logits_fake.size(), dtype=dtype, device=device))

    return loss


class Flatten(nn.Module):
    """
    Flatten all dimensions except batch_size
    """

    def forward(self, x):
        N = x.size()[0]
        return x.view(N, -1)


class Unflatten(nn.Module):
    """
    Receives as input a vector of N, C, H, W
    """

    def __init__(self, N, C, H, W):
        super().__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)


if __name__ == "__main__":
    noise_dim = 10

    generator = DC_Generator(noise_dim)
    generator.apply(initialize_weights)

    discriminator = DC_Discriminator()
    discriminator.apply(initialize_weights)

    generator_output = generator(sample_noise(4, noise_dim))
    # show_images_square(generator_output.data.cpu())

    print(discriminator(generator_output).data.numpy())
