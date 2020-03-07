import IPython
import signal

import torch
import torchvision

import matplotlib.pyplot as plt
import itertools

# Load MNIST dataset
mnist = torchvision.datasets.MNIST(
	root = 'data/',
	download = True,
	train = True,
	transform = torchvision.transforms.ToTensor() # The data is stored in numpy. Transform it to PyTorch tensors.
)

mnist_batched = torch.utils.data.DataLoader(mnist, batch_size = 32)

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


if __name__ == '__main__':

    img, lbl = mnist[0]
    # (1 channel * 28 width * 28 height)
    print('Shape of a single mnist image:', img.shape)
    plt.imshow(img[0], cmap='gray')
    plt.colorbar()
    plt.show()

    imgs, lbls = next(iter(mnist_batched))
    x = torchvision.utils.make_grid(imgs, nrow=3)
    plt.imshow(x.numpy().transpose((1, 2, 0)))
    plt.show()
