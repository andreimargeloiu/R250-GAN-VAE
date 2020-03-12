"""
Usage:
    evalaute.py [options]

Options:
    -h
    --debug                         Enable debug routines. [default: False]
    --model=NAME                    Model type name to run (e.g. vaegan)
    --model-id=NAME                 the ID of the model (i.e. the time of running it)
    --base-path=NAME                Path to the project folder [default: .]
"""
import gzip
import os
import pickle
from os import path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
from docopt import docopt
from dpu_utils.utils import run_and_debug
from sys import exit

from models import Encoder, Decoder, Discriminator, sample_noise
from utils import fix_random_seed, get_dataset_iterator, show_images_square


def run(args):
    """
    Given a model, perform evaluation by making different plots.
    """
    fix_random_seed(0)

    if args['--model'] != 'vaegan':
        print("Support evaluation only VAEGAN. You asked for a different model which is unsupported.")

        exit(-1)


    saved_model_dir = path.join(path.join(args['--base-path'], f"saved_models"), args['--model-id'])
    with open(path.join(saved_model_dir, 'parameters'), 'rb') as input:
        parameters = pickle.load(input)
        print(parameters)

    # Restore models
    device = torch.device('cpu')
    encoder = Encoder(parameters['--latent-dimension'], device, negative_slope=parameters['--negative_slope'])
    with gzip.open(path.join(saved_model_dir, 'encoder'), "rb") as input:
        encoder.load_state_dict(torch.load(input, map_location=device))

    decoder = Decoder(parameters['--latent-dimension'], activation='sigmoid', negative_slope=parameters['--negative_slope'])
    with gzip.open(path.join(saved_model_dir, 'decoder'), "rb") as input:
        decoder.load_state_dict(torch.load(input, map_location=device))

    #### Reconstruction
    batch_iter = get_dataset_iterator(batch_size=10)
    imgs, _ = next(batch_iter)

    encoder.eval()
    decoder.eval()

    z, mu, logvar = encoder(imgs)
    recon = decoder(z).detach()

    show_images_square(imgs)
    plt.show()

    show_images_square(recon)
    plt.show()

    x = torchvision.utils.make_grid(torch.cat([imgs, recon], dim=0), nrow=10, pad_value=0.5)

    print("Reconstruction")
    plt.imshow(x.numpy().transpose((1,2,0)))
    plt.axis('off')
    plt.show()

    #### Random synthesis
    print("Random Synthesis")
    noise_for_images_to_show = sample_noise(16, parameters["--latent-dimension"], dtype=torch.float, device=device)
    imgs_output = decoder(noise_for_images_to_show[:16]).data.cpu()
    fig = show_images_square(imgs_output)
    plt.show()


    #### Systematically varying the latents
    print("Varying the latents")
    def vary_latent_representation(poz, no_latents_to_vary=10, num=9):
        z, mu, logvar = encoder(imgs)
        z = z[poz].detach() # take first image
        i = torch.eye(parameters["--latent-dimension"])

        zvar = torch.stack([z + λ * i[d] for d in range(no_latents_to_vary) for λ in np.linspace(-5, 5, num)])
        rx = decoder(zvar).detach()
        rx = torchvision.utils.make_grid(rx, nrow=num, pad_value=.5)

        plt.figure(figsize=(7, 7))
        plt.imshow(rx.numpy().transpose((1, 2, 0)))
        plt.axis('off')
        plt.show()

    vary_latent_representation(0)
    vary_latent_representation(1)
    vary_latent_representation(2)

    #### Interpolation
    def clean_interpolation(num=10):
        z, mu, logvar = encoder(imgs)
        z = z.detach().numpy()
        λ = np.linspace(0, 1, num)[:, np.newaxis]
        znew = (1 - λ) * z[1] + λ * z[3]
        znew = torch.tensor(znew, dtype=torch.float32)
        recon = decoder(znew).detach()
        x = torchvision.utils.make_grid(recon, nrow=num, pad_value=.5)

        plt.imshow(x.numpy().transpose((1, 2, 0)))
        plt.show()

    clean_interpolation()



if __name__ == "__main__":
    args = docopt(__doc__)

    run_and_debug(lambda: run(args), args["--debug"])