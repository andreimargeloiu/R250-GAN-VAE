"""
Usage:
    train.py [options]

Options:
    -h
    --debug                         Enable debug routines. [default: False]
    --epochs=INT                    Number of epochs to run [default: 5]
    --show-every=INT                Print statistics every X epochs [default:1]
    --base-path=NAME                Path to the log file [default: .]
"""
import json
from datetime import datetime
from os import path

import git
import logging
import matplotlib.pyplot as plt
import torch
from docopt import docopt
from dpu_utils.utils import run_and_debug
from torch import nn
from torch.autograd import Variable
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss

from models import DC_Generator, DC_Discriminator, discriminator_loss, generator_loss, sample_noise, Encoder, BetaVAE
from utils import fix_random_seed, get_dataset_iterator, show_images_square

from torch.utils.tensorboard import SummaryWriter

"""
    Global variables
"""
dtype = torch.float
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print('using device:', device)


def initialize_logger(args):
    logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%m-%d %H:%M:%S',
                        filename=path.join(args['--base-path'], 'logs/training.log'),
                        filemode='a')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s'))

    # Attach the console to the root logger
    logging.getLogger('').addHandler(console)
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    logging.info(f"\n\n---Started Training from commit {sha}---")
    logging.info(json.dumps(args, ensure_ascii=True, indent=2, sort_keys=True))


def run(args):
    # train_dcgan(args)
    train_vaegan(args)


def train_dcgan(args, noise_dim=96):
    fix_random_seed(0)
    # Create models and initialize weights
    DC_Gen = DC_Generator(noise_dim).to(device)
    DC_Disc = DC_Discriminator().to(device)

    # Create optimizer
    optimizer_Gen = torch.optim.Adam(DC_Gen.parameters(), lr=1e-3, betas=(0.5, 0.999))
    optimizer_Dis = torch.optim.Adam(DC_Disc.parameters(), lr=1e-3, betas=(0.5, 0.999))

    iter_count = 0
    show_every = 250
    batch_size = 128
    noise_for_images_to_show = sample_noise(36, noise_dim, dtype=dtype, device=device)

    for epoch in range(int(args['--epochs'])):
        batch_iter = get_dataset_iterator(batch_size=batch_size)
        for imgs, _ in batch_iter:
            if len(imgs) != batch_size:
                print("Alert! len(imgs) < batch_size")
                continue

            # Update D
            optimizer_Dis.zero_grad()
            real_data = imgs.to(device)  # Move data to device (GPU)
            real_data = (real_data - 0.5) * 2  # make the input between -1 and 1

            noise_input = sample_noise(batch_size, noise_dim, dtype=real_data.dtype, device=real_data.device)
            fake_image = DC_Gen(noise_input)
            logits_fake, _ = DC_Disc(fake_image)
            logits_real, _ = DC_Disc(real_data)

            D_loss = discriminator_loss(logits_real, logits_fake, dtype=dtype, device=device)
            D_loss.backward()
            optimizer_Dis.step()

            # Update G parameters
            noise_input = sample_noise(batch_size, noise_dim, dtype=real_data.dtype, device=real_data.device)
            fake_image = DC_Gen(noise_input)
            logits_fake, _ = DC_Disc(fake_image)

            optimizer_Gen.zero_grad()
            G_loss = generator_loss(logits_fake, dtype=dtype, device=device)
            G_loss.backward()
            optimizer_Gen.step()

            if iter_count % show_every == 0:
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, D_loss.item(), G_loss.item()))
                imgs_output = DC_Gen(noise_for_images_to_show).data.cpu()
                show_images_square(imgs_output)
                plt.show()
                print()
            iter_count += 1


def train_vaegan(args, latent_dimension=128):
    fix_random_seed(0)

    #### Hyperparameters
    iter_count = 0
    show_every = args['--show-every']
    batch_size = 128
    noise_for_images_to_show = sample_noise(16, latent_dimension, dtype=dtype, device=device)
    gamma = 1
    lr = 1e-3
    beta = 1
    negative_slope = 0.01

    tb_logdir = path.join(args['--base-path'],
                          f"tensorboard/scalars/{datetime.now().strftime('%Y%m%d-%H%M%S')}_lr={lr}_gamma={gamma}_beta={beta}_neg_slope={negative_slope}_batch={batch_size}")
    write = SummaryWriter(log_dir=tb_logdir, flush_secs=20)


    # Create models and initialize weights
    encoder = Encoder(latent_dimension, device, negative_slope).to(device)
    decoder = DC_Generator(latent_dimension, activation='sigmoid', negative_slope=negative_slope).to(device)
    discriminator = DC_Discriminator(negative_slope).to(device)

    ########## Testing Forward Pass #############
    # Test encode - decode
    # input = sample_noise(batch_size, latent_dimension, dtype=dtype, device=device)
    imgs, _ = next(get_dataset_iterator(batch_size=16))  # (batch_size, 1, 28, 28)

    imgs_enc, _, _ = encoder.forward(imgs.to(device))
    print("Latent variables has dimension: {imgs_enc.size()}")

    imgs_dec = decoder.forward(imgs_enc)
    print(f"Decoder dimension is {imgs_dec.size()}")
    # show_images_square(imgs_dec.detach())
    # plt.show()

    # Test discriminator
    decision, _ = discriminator.forward(imgs_dec)
    print(f"Discriminator output shape: {decision.size()}")
    ########## End test Forward Pass #############


    # Optimizer
    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Loss
    criterion = nn.BCELoss()  # Binary cross entropy
    criterion_lth = nn.L1Loss()  # For the 'content loss'
    criterion_rec = nn.MSELoss(reduction='sum')  # MSE loss
    criterion_kl = nn.KLDivLoss(reduction='sum')  # KL divergence


    for epoch in range(int(args['--epochs'])):
        batch_iter = get_dataset_iterator(batch_size=batch_size)
        for imgs, _ in batch_iter:
            # Forward pass
            x_real = imgs.to(device)
            x_encoded, mu, logvar = encoder.forward(x_real)
            x_decoded = decoder.forward(x_encoded)

            logits_real, l_layer_real = discriminator.forward(x_real)
            logits_fake, l_layer_fake = discriminator.forward(x_decoded)
            acc_real = (logits_real >= 0).sum() / batch_size
            acc_fake = (logits_fake < 0).sum() / batch_size

            ###### Compute losses ######
            L_prior = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            z_prior = Variable(torch.randn(batch_size, latent_dimension).to(device))
            x_decoded_prior = decoder.forward(z_prior)
            logits_from_prior, _ = discriminator.forward(x_decoded_prior)
            y_fake = Variable(torch.zeros(logits_from_prior.size(), dtype=dtype, device=device))
            L_discriminator_prior = bce_loss(logits_from_prior, y_fake)

            L_llik_l = criterion_lth(l_layer_real, l_layer_fake)  # Consider the L-th layer as the last layer

            L_GAN = discriminator_loss(logits_real, logits_fake, dtype, device) \
                    + L_discriminator_prior

            ###### Backprop ######
            L_enc = beta * L_prior + L_llik_l
            L_dec = gamma * L_llik_l - L_GAN
            L_dis = L_GAN

            optimizer_encoder.zero_grad()
            L_enc.backward(retain_graph=True)
            optimizer_encoder.step()

            optimizer_decoder.zero_grad()
            L_dec.backward(retain_graph=True)
            optimizer_decoder.step()

            optimizer_discriminator.zero_grad()
            L_dis.backward(retain_graph=True)
            optimizer_discriminator.step()


            #### Tensorboard
            write.add_scalar("Loss/Encoder", L_enc.item(), iter_count)
            write.add_scalar("Loss/Decoder", L_dec.item(), iter_count)
            write.add_scalar("Loss/Discriminator", L_dis.item(), iter_count)
            write.add_scalar("Accuracy/Acc-Real", acc_real, iter_count)
            write.add_scalar("Accuracy/Acc-Fake", acc_fake, iter_count)

            if iter_count % show_every == 0:
                print('Iter: {}, Enc: {:.4}, Dec:{:.4}, Dis:{:.4}'.format(iter_count, L_enc.item(), L_dec.item(),
                                                                          L_dis.item()))
                imgs_output = decoder(noise_for_images_to_show[:16]).data.cpu()
                fig = show_images_square(imgs_output)
                write.add_figure(tag="Figure/Initial_noise", figure=fig, global_step=iter_count)
                plt.show()

                write.add_figure(tag="Figure/Reconstructed", figure=show_images_square(x_decoded.data.cpu()[:16]), global_step=iter_count)
                plt.show()

            iter_count += 1


def train_betavae(args, latent_dimension=64):
    fix_random_seed(0)

    betaVAE = BetaVAE(latent_dimension, device).to(device)
    optimizer_betaVAE = torch.optim.Adam(betaVAE.parameters(), lr=1e-3)

    iter_count = 0
    show_every = 250
    batch_size = 128
    noise_for_images_to_show = sample_noise(16, latent_dimension, dtype=dtype, device=device)

    for epoch in range(int(args['--epochs'])):
        batch_iter = get_dataset_iterator(batch_size=batch_size)
        for imgs, _ in batch_iter:
            # Forward pass
            x_real = imgs.to(device)
            optimizer_betaVAE.zero_grad()
            rx, mu, logvar = betaVAE(x_real)
            loss = BetaVAE.loss_function(rx, x_real, mu, logvar, beta=3)
            loss.backward()
            optimizer_betaVAE.step()

            if iter_count % show_every == 0:
                print('Iter: {}, Loss: {:.4}'.format(iter_count, loss.item()))
                imgs_output = betaVAE.decoder(noise_for_images_to_show)
                show_images_square(imgs_output.data.cpu())
                plt.show()

                show_images_square(rx.data.cpu())
                plt.show()
            iter_count += 1


if __name__ == "__main__":
    args = docopt(__doc__)
    fix_random_seed(0)
    initialize_logger(args)

    run_and_debug(lambda: run(args), args["--debug"])
