import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, z_dim, hidden_1, hidden_2):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(784, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc31 = nn.Linear(hidden_2, z_dim)
        self.fc32 = nn.Linear(hidden_2, z_dim)
        # setup the non-linearities
        self.relu = nn.ReLU()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, 784)
        # then compute the hidden units
        hidden = self.relu(self.fc1(x))
        hidden = self.relu(self.fc2(hidden))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc31(hidden)
        z_scale = torch.exp(self.fc32(hidden))
        return z_loc, z_scale


class CVAE:
    def __init__(self):
        self.recognition_net = MLP(200, 1000, 1000)
        self.conditional_net = MLP(200, 1000, 1000)
        self.generation_net = MLP(200, 1000, 1000)

    def model(self, data):
        raise NotImplemented

    def guide(self, data):
        raise NotImplemented




