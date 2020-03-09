import math
import random

import IPython
import signal

import numpy as np
import torch
import torchvision

import matplotlib.pyplot as plt
import itertools

# Load MNIST dataset
from matplotlib import gridspec
from torch import nn
from torch.nn import init

mnist = torchvision.datasets.MNIST(
    root='data/',
    download=True,
    train=True,
    transform=torchvision.transforms.ToTensor()  # The data is stored in numpy. Transform it to PyTorch tensors.
)


def get_dataset_iterator(batch_size=128):
    mnist_batched = torch.utils.data.DataLoader(mnist,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True,
                                                num_workers=2)
    return iter(mnist_batched)


## Utilities

# Here are some messy Python tricks to support interactive Jupyter work on training neural networks. In once cell, run
# ```
# iter_training_data = enumerate_cycle(MYDATA)
# ```
# and in the next
# ```
# while not interrupted():
#     (epoch, batch_num), x = next(iter_training_data)
#     ... # DO THE WORK
#     if batch_num % 25 == 0:
#         IPython.display.clear_output(wait=True)
#         print(f'epoch={epoch} batch={batch_num}/{len(mnist_batched)} loss={e.item()}')
# ```
# You can use the Kernel|Interrupt menu option, and it will interrupt cleanly.
# You can resume the iteration where it left off, by re-running the second cell.


def interrupted(_interrupted=[False], _default=[None]):
    if _default[0] is None or signal.getsignal(signal.SIGINT) == _default[0]:
        _interrupted[0] = False

        def handle(signal, frame):
            if _interrupted[0] and _default[0] is not None:
                _default[0](signal, frame)
            print('Interrupt!')
            _interrupted[0] = True

        _default[0] = signal.signal(signal.SIGINT, handle)
    return _interrupted[0]


def enumerate_cycle(g):
    epoch = 0
    while True:
        for i, x in enumerate(g):
            yield (epoch, i), x
        epoch = epoch + 1


def fix_random_seed(seed_no=0):
    """
      Fix random seed to get a deterministic output
      Inputs:
      - seed_no: seed number to be fixed
    """
    torch.manual_seed(seed_no)
    torch.cuda.manual_seed(seed_no)
    random.seed(seed_no)
    np.random.seed(seed_no)


def show_images_square(images, cmap='gray'):
    """
    Input:
    - images: Tensor of images (batch_size, C, H, W)
    """
    images = torch.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(math.ceil(math.sqrt(images.shape[0])))
    sqrtimg = int(math.ceil(math.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg, sqrtimg]), cmap=cmap)
    return fig


def count_params(model):
    """Count the number of parameters in the model"""
    param_count = sum([p.numel() for p in model.parameters()])
    return param_count


def initialize_weights(param):
    if isinstance(param, nn.Linear) \
            or isinstance(param, nn.ConvTranspose2d) \
            or isinstance(param, nn.Conv2d):
        init.xavier_uniform_(param.weight.data)


if __name__ == '__main__':
    img, lbl = mnist[0]
    # (1 channel * 28 width * 28 height)
    print('Shape of a single mnist image:', img.shape)
    plt.imshow(img[0], cmap='gray')
    plt.colorbar()
    plt.show()

    imgs, lbls = next(get_dataset_iterator())
    x = torchvision.utils.make_grid(imgs, nrow=3)
    plt.imshow(x.numpy().transpose((1, 2, 0)))
    plt.show()
