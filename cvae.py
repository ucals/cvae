import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc31 = nn.Linear(hidden_2, z_dim)
        self.fc32 = nn.Linear(hidden_2, z_dim)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        # put x and y together in the same image for simplification
        x[x == -1] = y[x == -1]
        # then compute the hidden units
        hidden = self.relu(self.fc1(x))
        hidden = self.relu(self.fc2(hidden))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc31(hidden)
        z_scale = torch.exp(self.fc32(hidden))
        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 784)
        self.relu = nn.ReLU()

    def forward(self, z):
        y = self.relu(self.fc1(z))
        y = self.relu(self.fc2(y))
        y = self.fc3(y)
        return y


class CVAE(nn.Module):
    def __init__(self, z_dim, hidden_1, hidden_2, pre_trained_baseline_net):
        super().__init__()
        self.baseline_net = pre_trained_baseline_net
        self.prior_net = Encoder(z_dim, hidden_1, hidden_2)
        self.generation_net = Decoder(z_dim, hidden_1, hidden_2)
        self.recognition_net = Encoder(z_dim, hidden_1, hidden_2)

    def model(self, xs, ys=None):
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("generation_net", self)
        with pyro.plate("data"):

            # Prior network uses the baseline predictions as initial guess.
            # This is the generative process with recurrent connection
            with torch.no_grad():
                y_hat = self.baseline_net(xs).view(xs.shape)

            # sample the handwriting style from the prior distribution, which is
            # modulated by the input xs.
            prior_loc, prior_scale = self.prior_net(xs, y_hat)
            zs = pyro.sample('z', dist.Normal(prior_loc, prior_scale).to_event(1))

            # the output y is generated from the distribution pθ(y|x, z)
            loc = self.generation_net(zs)
            # we will only sample in the masked image
            masked_loc = loc[xs == -1]
            masked_ys = ys[xs == -1] if ys is not None else None
            pyro.sample('y', dist.Bernoulli(masked_loc).to_event(1), obs=masked_ys)
            # return the loc so we can visualize it later
            return loc

    def guide(self, xs, ys=None):
        with pyro.plate("data"):
            # sample (and score) the latent handwriting-style with the
            # variational distribution q(z|x,y) = normal(loc(x,y),scale(x,y))
            loc, scale = self.recognition_net(xs, ys)
            pyro.sample("z", dist.Normal(loc, scale).to_event(1))











