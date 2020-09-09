import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F


class PriorNet(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_1, hidden_2):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(x_dim, hidden_1)
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


class GenerationNet(nn.Module):
    def __init__(self, z_dim, hidden_1, hidden_2):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 784)
        # setup the non-linearities
        self.relu = nn.ReLU()

    def forward(self, z):
        # then compute the hidden units
        y = self.relu(self.fc1(z))
        y = self.relu(self.fc2(y))
        y = self.fc3(y)
        return y


class RecognitionNet(nn.Module):
    def __init__(self, z_dim, hidden_1, hidden_2):
        super().__init__()


class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.prior_net = PriorNet(200, 1000, 1000)
        self.generation_net = GenerationNet(200, 1000, 1000)
        self.recognition_net = MLP(200, 1000, 1000)

    def model(self, xs, ys=None):
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("ss_vae", self)
        with pyro.plate("data"):

            # sample the handwriting style from the prior distribution, which is
            # modulated by the input xs.
            prior_loc, prior_scale = self.prior_net(xs)  # TODO feed y_hat
            zs = pyro.sample('z', dist.Normal(prior_loc, prior_scale).to_event(1))

            # the output y is generated from the distribution pθ(y|x, z)
            loc = self.generation_net(zs)
            # TODO remove pixels with value -1
            pyro.sample('y', dist.Bernoulli(loc).to_event(1), obs=ys)
            # return the loc so we can visualize it later
            return loc

    def guide(self, xs, ys=None):
        with pyro.plate("data"):
            # sample (and score) the latent handwriting-style with the
            # variational distribution q(z|x,y) = normal(loc(x,y),scale(x,y))
            loc, scale = self.recognition_net(xs, ys)
            pyro.sample("z", dist.Normal(loc, scale).to_event(1))











