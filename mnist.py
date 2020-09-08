# First step in implementing the paper is to prepare the datasets used. The
# First dataset used is MNIST. The authors divide each digit image into four
# quadrants, and take one, two, or three quadrant(s) as an input and the
# remaining quadrants as an output. We will implement that using a custom
# transformation.

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
import torchvision.transforms.functional as F


class CVAEMNIST(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.original = MNIST(root, train=train, download=download)
        self.transform = transform

    def __getitem__(self, item):
        image, digit = self.original[item]
        sample = {'original': image, 'digit': digit}
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor:
    def __call__(self, sample):
        sample['original'] = F.to_tensor(sample['original'])
        sample['digit'] = torch.as_tensor(np.asarray(sample['digit']),
                                          dtype=torch.int64)
        return sample


class RemoveQuadrants:
    def __init__(self, num, mask_with=-1):
        if num <= 0 or num >= 4:
            raise ValueError('Number of quadrants to remove must be 1, 2 or 3')
        self.num = num
        self.mask_with = mask_with

    def __call__(self, sample):
        tensor = sample['original'].squeeze()
        inp = tensor.detach().clone()
        h, w = tensor.shape

        inp[h // 2:, :w // 2] = self.mask_with
        if self.num == 2:
            inp[:, :w // 2] = self.mask_with
        if self.num == 3:
            inp[:h // 2, :] = self.mask_with

        out = tensor.clone()
        out[inp != -1] = self.mask_with

        sample['input'] = inp
        sample['output'] = out
        return sample
