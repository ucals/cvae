import numpy as np
import pandas as pd
from pathlib import Path
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
import torch
import torch.nn as nn
from tqdm import tqdm
from mnist import get_val_images


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
        # The CVAE is composed of multiple MLPs, such as recognition network
        # qφ(z|x, y), (conditional) prior network pθ(z|x), and generation
        # network pθ(y|x, z). Also, CVAE is built on top of the NN: not only
        # the direct input x, but also the initial guess y_hat made by the NN
        # are fed into the prior network.
        self.baseline_net = pre_trained_baseline_net
        self.prior_net = Encoder(z_dim, hidden_1, hidden_2)
        self.generation_net = Decoder(z_dim, hidden_1, hidden_2)
        self.recognition_net = Encoder(z_dim, hidden_1, hidden_2)

    def model(self, xs, ys=None):
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("generation_net", self)
        batch_size = xs.shape[0]
        with pyro.plate("data"):

            # Prior network uses the baseline predictions as initial guess.
            # This is the generative process with recurrent connection
            with torch.no_grad():
                # this ensures the training process does not change the
                # baseline network
                y_hat = self.baseline_net(xs).view(xs.shape)

            # sample the handwriting style from the prior distribution, which is
            # modulated by the input xs.
            prior_loc, prior_scale = self.prior_net(xs, y_hat)
            zs = pyro.sample('z', dist.Normal(prior_loc, prior_scale).to_event(1))

            # the output y is generated from the distribution pθ(y|x, z)
            loc = self.generation_net(zs)

            if ys is not None:
                # In training, we will only sample in the masked image
                mask_loc = loc[(xs == -1).view(-1, 784)].view(batch_size, -1)
                mask_ys = ys[xs == -1].view(batch_size, -1)
                pyro.sample('y', dist.Bernoulli(mask_loc).to_event(1), obs=mask_ys)
            else:
                # In testing, no need to sample: the output is already a
                # probability in [0, 1] range, which better represent pixel
                # values considering grayscale. If we sample, we will force
                # each pixel to be  either 0 or 1, killing the grayscale
                pyro.deterministic('y', loc.detach())

            # return the loc so we can visualize it later
            return loc

    def guide(self, xs, ys=None):
        with pyro.plate("data"):
            if ys is None:
                # at inference time, ys is not provided. In that case,
                # the model uses the prior network
                y_hat = self.baseline_net(xs).view(xs.shape)
                loc, scale = self.prior_net(xs, y_hat)
            else:
                # at training time, uses the variational distribution
                # q(z|x,y) = normal(loc(x,y),scale(x,y))
                loc, scale = self.recognition_net(xs, ys)

            pyro.sample("z", dist.Normal(loc, scale).to_event(1))

    def save(self, model_path):
        torch.save({
            'prior': self.prior_net.state_dict(),
            'generation': self.generation_net.state_dict(),
            'recognition': self.recognition_net.state_dict()
        }, model_path)

    def load(self, model_path, map_location=None):
        net_weights = torch.load(model_path, map_location=map_location)
        self.prior_net.load_state_dict(net_weights['prior'])
        self.generation_net.load_state_dict(net_weights['generation'])
        self.recognition_net.load_state_dict(net_weights['recognition'])
        self.prior_net.eval()
        self.generation_net.eval()
        self.recognition_net.eval()


def train(device, dataloaders, dataset_sizes, learning_rate, num_epochs,
          early_stop_patience, model_path, pre_trained_baseline_net):

    # clear param store
    pyro.clear_param_store()

    cvae_net = CVAE(200, 500, 500, pre_trained_baseline_net)
    cvae_net.to(device)
    optimizer = pyro.optim.Adam({"lr": learning_rate})
    svi = SVI(cvae_net.model, cvae_net.guide, optimizer, loss=Trace_ELBO())

    best_loss = np.inf
    early_stop_count = 0
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    # to track evolution
    val_inp, digits = get_val_images(num_quadrant_inputs=1,
                                     num_images=30, shuffle=False)
    val_inp = val_inp.to(device)
    samples = []
    losses = []

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            running_loss = 0.0

            # Iterate over data.
            bar = tqdm(dataloaders[phase],
                       desc='CVAE Epoch {} {}'.format(epoch, phase).ljust(20))
            for i, batch in enumerate(bar):
                inputs = batch['input'].to(device)
                outputs = batch['output'].to(device)

                if phase == 'train':
                    loss = svi.step(inputs, outputs) / inputs.size(0)
                else:
                    loss = svi.evaluate_loss(inputs, outputs) / inputs.size(0)

                # statistics
                running_loss += loss
                if i % 10 == 0:
                    bar.set_postfix(loss='{:.2f}'.format(loss),
                                    early_stop_count=early_stop_count)

                # track evolution
                if phase == 'train':
                    df = pd.DataFrame(columns=['epoch', 'loss'])
                    df.loc[0] = [epoch + float(i) / len(dataloaders[phase]), loss]
                    losses.append(df)
                    if i % 47 == 0:  # every 10% of training (469)
                        dfs = predict_samples(
                            val_inp, digits, cvae_net,
                            epoch + float(i) / len(dataloaders[phase]),
                        )
                        samples.append(dfs)

            epoch_loss = running_loss / dataset_sizes[phase]
            # deep copy the model
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    cvae_net.save(model_path)
                    early_stop_count = 0
                else:
                    early_stop_count += 1

        if early_stop_count >= early_stop_patience:
            break

    # Save model weights
    cvae_net.load(model_path)

    # record evolution
    samples = pd.concat(samples, axis=0, ignore_index=True)
    samples.to_csv('samples.csv', index=False)

    losses = pd.concat(losses, axis=0, ignore_index=True)
    losses.to_csv('losses.csv', index=False)

    return cvae_net


def predict_samples(inputs, digits, pre_trained_cvae, epoch_frac):
    predictive = Predictive(pre_trained_cvae.model,
                            guide=pre_trained_cvae.guide,
                            num_samples=1)
    preds = predictive(inputs)
    y_loc = preds['y'].squeeze().detach().cpu().numpy()
    dfs = pd.DataFrame(data=y_loc)
    dfs['digit'] = digits.numpy()
    dfs['epoch'] = epoch_frac
    return dfs
