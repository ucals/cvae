import torch
from torchvision.transforms import Compose
from cvae.mnist import *


if __name__ == '__main__':
    dataset = CVAEMNIST(
        '../data',
        download=True,
        transform=Compose([ToTensor(), RemoveQuadrants(num=2)])
    )

    print(dataset[0])
    print(len(dataset))
