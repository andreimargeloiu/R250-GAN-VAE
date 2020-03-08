import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
from torch.nn import functional as F

import IPython
from utils import mnist, initialize_weights, show_images_square
from utils import interrupted, enumerate_cycle


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
        self.model = nn.Sequential(
            Unflatten(-1, 1, 28, 28),
            nn.Conv2d(1, 32, kernel_size=(5, 5), stride=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            Flatten(),

            nn.Linear(1024, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        """
        Input:
        - x: tensor of (None, 1, 28, 28)

        Output:
        - predictions: tensor of (None, 1)
        """
        return self.model(x)


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
        N, C, H, W = x.size()
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
