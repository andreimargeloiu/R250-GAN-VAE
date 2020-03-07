"""
Usage:
    train.py [options]

Options:
    -h
    --debug                         Enable debug routines. [default: False]
    --epochs=INT                    Number of epochs to run

    --log-file=NAME                 Path to the log file [default: ./logs/training.log]
"""
import json
import logging

import git
import torch
from docopt import docopt
from dpu_utils.utils import run_and_debug
import matplotlib.pyplot as plt

from models import DC_Generator, DC_Discriminator, discriminator_loss, generator_loss, sample_noise
from utils import fix_random_seed, get_dataset_iterator, initialize_weights, show_images_square

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
                        filename=args['--log-file'],
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
    train_dcgan(args)


def create_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.5, 0.999))

def train_dcgan(args, noise_dim=96):
    # Create models and initialize weights
    DC_Gen = DC_Generator(noise_dim).to(device)
    DC_Gen.apply(initialize_weights)
    DC_Disc = DC_Discriminator().to(device)
    DC_Disc.apply(initialize_weights)

    # Create optimizer
    optimizer_Gen = create_optimizer(DC_Gen)
    optimizer_Dis = create_optimizer(DC_Disc)

    no_epochs = 5
    iter_count = 0
    show_every = 1
    batch_size = 128
    noise_for_images_to_show = sample_noise(16, noise_dim, dtype=dtype, device=device)

    for epoch in range(int(args['--epochs'])):
        batch_iter = get_dataset_iterator(batch_size=batch_size)
        for imgs, _ in batch_iter:
            if len(imgs) != batch_size:
                print("Alert! len(imgs) < batch_size")
                continue

            # Update D
            real_data = imgs.to(device)          # Move data to device (GPU)
            real_data = (real_data - 0.5) * 2    # make the input between -1 and 1

            noise_input = sample_noise(batch_size, noise_dim, dtype=real_data.dtype, device=real_data.device)
            fake_image = DC_Gen(noise_input)
            logits_fake = DC_Disc(fake_image)
            logits_real = DC_Disc(real_data)

            optimizer_Dis.zero_grad()
            D_loss = discriminator_loss(logits_real, logits_fake, dtype=dtype, device=device)
            D_loss.backward()
            optimizer_Dis.step()

            # Update G parameters
            noise_input = sample_noise(batch_size, noise_dim, dtype=real_data.dtype, device=real_data.device)
            fake_image = DC_Gen(noise_input)
            logits_fake = DC_Disc(fake_image)

            optimizer_Gen.zero_grad()
            G_loss = generator_loss(logits_fake, dtype=dtype, device=device)
            G_loss.backward()
            optimizer_Gen.step()

            if iter_count % show_every == 0:
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, D_loss.item(), G_loss.item()))
                imgs_output = DC_Gen(noise_for_images_to_show).data.cpu()
                show_images_square(imgs_output[0:16])
                plt.show()
                print()
            iter_count += 1


if __name__ == "__main__":
    args = docopt(__doc__)
    fix_random_seed(0)
    initialize_logger(args)

    run_and_debug(lambda: run(args), args["--debug"])
