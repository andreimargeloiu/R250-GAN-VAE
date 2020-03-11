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

import torch
import torchvision
from docopt import docopt
from dpu_utils.utils import run_and_debug
from sys import exit

from models import Encoder, Decoder, Discriminator
from utils import fix_random_seed, get_dataset_iterator


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

    # Restore models
    device = torch.device('cpu')
    encoder = Encoder(parameters['--latent-dimension'], device)
    with gzip.open(path.join(saved_model_dir, 'encoder'), "rb") as input:
        encoder.load_state_dict(torch.load(input, map_location=device))

    decoder = Decoder(parameters['--latent-dimension'], activation='sigmoid')
    with gzip.open(path.join(saved_model_dir, 'decoder'), "rb") as input:
        decoder.load_state_dict(torch.load(input, map_location=device))

    #### Reconstruction
    batch_iter = get_dataset_iterator(batch_size=5)
    imgs, _ = next(batch_iter)

    encoder.eval()
    _, mu, _ = encoder(imgs)
    decoder.eval()
    imgs_recon = decoder(mu).detach()

    x = torchvision.utils.make_grid(torch.cat([imgs, imgs_recon], dim=0), nrow=5, pad_value=0.5)

    print("Reconstruction")
    plt.imshow(x.numpy().transpose((1,2,0)))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    args = docopt(__doc__)

    run_and_debug(lambda: run(args), args["--debug"])