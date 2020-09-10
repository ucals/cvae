import numpy as np
from pathlib import Path
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from tqdm import tqdm
import torch
import torch.nn as nn
from baseline import BaselineNet
from mnist import get_data


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
        xc = x.clone()
        xc[x == -1] = y[x == -1]
        xc = xc.view(-1, 784)
        # then compute the hidden units
        hidden = self.relu(self.fc1(xc))
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
        y = torch.sigmoid(self.fc3(y))
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
            mask_loc = loc[(xs == -1).view(-1, 784)]
            mask_ys = ys[xs == -1] if ys is not None else None
            pyro.sample('y', dist.Bernoulli(mask_loc).to_event(1), obs=mask_ys)
            # return the loc so we can visualize it later
            return loc

    def guide(self, xs, ys=None):
        with pyro.plate("data"):
            # sample (and score) the latent handwriting-style with the
            # variational distribution q(z|x,y) = normal(loc(x,y),scale(x,y))
            loc, scale = self.recognition_net(xs, ys)
            pyro.sample("z", dist.Normal(loc, scale).to_event(1))

    def predict(self, xs, num_samples=1):
        # evaluate initial guess
        with torch.no_grad():
            y_hat = self.baseline_net(xs).view(xs.shape)
        prior_loc, prior_scale = self.prior_net(xs, y_hat)

        yss = []
        for i in range(num_samples):
            # sample in latent space
            zs = dist.Normal(prior_loc, prior_scale).sample()
            # the output y is generated from the distribution pθ(y|x, z)
            with torch.no_grad():
                ys = self.generation_net(zs).view(xs.shape)
            yss.append(ys)

        yss = torch.stack(yss, dim=0)
        return yss

    def save(self, model_path):
        parent = Path(model_path).parent
        stem = Path(model_path).stem
        torch.save(self.prior_net.state_dict(),
                   parent / f'{stem}_prior.pth')
        torch.save(self.generation_net.state_dict(),
                   parent / f'{stem}_generation.pth')
        torch.save(self.recognition_net.state_dict(),
                   parent / f'{stem}_recognition.pth')

    def load(self, model_path):
        parent = Path(model_path).parent
        stem = Path(model_path).stem
        self.prior_net.load_state_dict(
            torch.load(parent / f'{stem}_prior.pth'))
        self.generation_net.load_state_dict(
            torch.load(parent / f'{stem}_generation.pth'))
        self.recognition_net.load_state_dict(
            torch.load(parent / f'{stem}_recognition.pth'))

        self.prior_net.eval()
        self.generation_net.eval()
        self.recognition_net.eval()


def train(device, dataloaders, dataset_sizes, learning_rate, num_epochs,
          early_stop_patience, model_path):

    pre_trained_baseline_net = BaselineNet(500, 500)
    pre_trained_baseline_net.load_state_dict(
        torch.load('../data/models/baseline_net_q1.pth'))
    pre_trained_baseline_net.eval()

    cvae_net = CVAE(200, 500, 500, pre_trained_baseline_net)
    cvae_net.to(device)
    optimizer = pyro.optim.Adam({"lr": learning_rate})
    svi = SVI(cvae_net.model, cvae_net.guide, optimizer, loss=Trace_ELBO())

    best_loss = np.inf
    early_stop_count = 0

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            running_loss = 0.0
            num_preds = 0

            # Iterate over data.
            bar = tqdm(dataloaders[phase], desc=f'Epoch {epoch} {phase}'.ljust(20))
            for batch in bar:
                inputs = batch['input'].to(device)
                outputs = batch['output'].to(device)

                if phase == 'train':
                    loss = svi.step(inputs, outputs)
                else:
                    loss = svi.evaluate_loss(inputs, outputs)

                # statistics
                running_loss += loss / inputs.size(0)
                num_preds += 1
                bar.set_postfix(loss=f'{running_loss / num_preds:.2f}',
                                early_stop_count=early_stop_count)

            # epoch_loss = running_loss / dataset_sizes[phase]
            # TODO add early stopping

    # Save model weights
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    cvae_net.save(model_path)
    return cvae_net


if __name__ == '__main__':
    # Dataset
    datasets, dataloaders, dataset_sizes = get_data(
        num_quadrant_inputs=1,
        batch_size=32
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train(
        device=device,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        learning_rate=1e-3,
        num_epochs=30,
        early_stop_patience=3,
        model_path='../data/models/cvae_net_q1.pth'
    )






