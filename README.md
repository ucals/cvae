# Conditional Variational Auto-encoder

## Introduction

This tutorial implements Learning Structured Output Representation using Deep Conditional Generative Models paper, which introduced Conditional Variational Auto-encoders in 2015, using Pyro PPL.

Supervised deep learning has been successfully applied for many recognition problems in machine learning and computer vision. 
Although it can approximate a complex many-to-one function very well when large number of training data is provided, the lack of probabilistic inference of the current supervised deep learning methods makes it difficult to model a complex structured output representations. 
In this work, Kihyuk Sohn, Honglak Lee and Xinchen Yan develop a scalable deep conditional generative model for structured output variables using Gaussian latent variables. 
The model is trained efficiently in the framework of stochastic gradient variational Bayes, and allows a fast prediction using stochastic feed-forward inference. 
They called the model Conditional Variational Auto-encoder (CVAE).

The CVAE is a conditional directed graphical model whose input observations modulate the prior on Gaussian latent variables that generate the outputs. 
It is trained to maximize the conditional marginal log-likelihood. 
The authors formulate the variational learning objective of the CVAE in the framework of stochastic gradient variational Bayes (SGVB). 
In experiments, they demonstrate the effectiveness of the CVAE in comparison to the deterministic neural network counterparts in generating diverse but realistic output predictions using stochastic inference. 
Here, we will implement their proof of concept: an artificial experimental setting for structured output prediction using MNIST database.

## The problem
Let's divide each digit image into four quadrants, and take one, two, or three quadrant(s) as an input and the remaining quadrants as an output to be predicted. 
The image below shows the case where one quadrant is the input:

<img src="https://i.ibb.co/x17xFwy/image1.png" alt="image1" width="300">

Our objective is to **learn a model that can perform probabilistic inference and make diverse predictions from a single input**. 
This is because we are not simply modeling a many-to-one function as in classification tasks, but we may need to model a mapping from single input to many possible outputs. One of the limitations of deterministic neural networks is that they generate only a single prediction. 
In the example above, the input shows a small part of a digit that might be a three or a five. 

## Preparing the data
We use the MNIST dataset; the first step is to prepare it. 
Depending on how many quadrants we will use as inputs, we will build the datasets and dataloaders, removing the unused pixels with -1:

```Python
class CVAEMNIST(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.original = MNIST(root, train=train, download=download)
        self.transform = transform

    def __len__(self):
        return len(self.original)

    def __getitem__(self, item):
        image, digit = self.original[item]
        sample = {'original': image, 'digit': digit}
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor:
    def __call__(self, sample):
        sample['original'] = functional.to_tensor(sample['original'])
        sample['digit'] = torch.as_tensor(np.asarray(sample['digit']),
                                          dtype=torch.int64)
        return sample


class MaskImages:
    """This torchvision image transformation prepares the MNIST digits to be
    used in the tutorial. Depending on the number of quadrants to be used as
    inputs (1, 2, or 3), the transformation masks the remaining (3, 2, 1)
    quadrant(s) setting their pixels with -1. Additionally, the transformation
    adds the target output in the sample dict as the complementary of the input
    """
    def __init__(self, num_quadrant_inputs, mask_with=-1):
        if num_quadrant_inputs <= 0 or num_quadrant_inputs >= 4:
            raise ValueError('Number of quadrants as inputs must be 1, 2 or 3')
        self.num = num_quadrant_inputs
        self.mask_with = mask_with

    def __call__(self, sample):
        tensor = sample['original'].squeeze()
        out = tensor.detach().clone()
        h, w = tensor.shape

        # removes the bottom left quadrant from the target output
        out[h // 2:, :w // 2] = self.mask_with
        # if num of quadrants to be used as input is 2,
        # also removes the top left quadrant from the target output
        if self.num == 2:
            out[:, :w // 2] = self.mask_with
        # if num of quadrants to be used as input is 3,
        # also removes the top right quadrant from the target output
        if self.num == 3:
            out[:h // 2, :] = self.mask_with

        # now, sets the input as complementary
        inp = tensor.clone()
        inp[out != -1] = self.mask_with

        sample['input'] = inp
        sample['output'] = out
        return sample


def get_data(num_quadrant_inputs, batch_size):
    transforms = Compose([
        ToTensor(),
        MaskImages(num_quadrant_inputs=num_quadrant_inputs)
    ])
    datasets, dataloaders, dataset_sizes = {}, {}, {}
    for mode in ['train', 'val']:
        datasets[mode] = CVAEMNIST(
            '../data',
            download=True,
            transform=transforms,
            train=mode == 'train'
        )
        dataloaders[mode] = DataLoader(
            datasets[mode],
            batch_size=batch_size,
            shuffle=mode == 'train',
            num_workers=0
        )
        dataset_sizes[mode] = len(datasets[mode])

    return datasets, dataloaders, dataset_sizes
```

## Baseline: Deterministic Neural Network
Before we dive into the CVAE implementation, let's code the baseline model. 
It is a straightforward implementation:

```Python
class BaselineNet(nn.Module):
    def __init__(self, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 784)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        hidden = self.relu(self.fc1(x))
        hidden = self.relu(self.fc2(hidden))
        y = torch.sigmoid(self.fc3(hidden))
        return y
```

In the paper, the authors compare the baseline NN with the proposed CVAE by comparing the negative (Conditional) Log Likelihood (CLL), averaged by image in the validation set. Thanks to PyTorch, computing the CLL is equivalent to computing the Binary Cross Entropy Loss using as input a signal passed through a Sigmoid layer. 
The code below does a small adjustment to leverage this: it only computes the loss in the pixels not masked with -1:

```Python
class MaskedBCELoss(nn.Module):
    def __init__(self, masked_with=-1):
        super().__init__()
        self.masked_with = masked_with

    def forward(self, input, target):
        target = target.view(input.shape)
        loss = F.binary_cross_entropy(input, target, reduction='none')
        loss[target == self.masked_with] = 0
        return loss.sum()
```

The training is very straightforward. We use 500 neurons in each hidden layer, Adam optimizer with `1e-3` learning rate, and early stopping. Please check the [Github repo](https://github.com/pyro-ppl/pyro/blob/dev/examples/cvae) for the full implementation.

## Deep Conditional Generative Models for Structured Output Prediction
As illustrated in the image below, there are three types of variables in a deep conditional generative model (CGM): input variables $\bf x$, output variables $\bf y$, and latent variables $\bf z$. 
The conditional generative process of the model is given in (b) as follows: for given observation $\bf x$, $\bf z$ is drawn from the prior distribution $p_{\theta}({\bf z} | {\bf x})$, and the output $\bf y$ is generated from the distribution $p_{\theta}({\bf y} | {\bf x, z})$. 
Compared to the baseline NN (a), the latent variables $\bf z$ allow for modeling multiple modes in conditional distribution of output variables $\bf y$ given input $\bf x$, making the proposed CGM suitable for modeling one-to-many mapping.


<img src="https://i.ibb.co/0mVvkSF/image2.png" alt="image1" width="800">

Deep CGMs are trained to maximize the conditional marginal log-likelihood. 
Often the objective function is intractable, and we apply the SGVB framework to train the model. 
The empirical lower bound is written as:

<img src="https://render.githubusercontent.com/render/math?math=\tilde{\mathcal{L}}_{\text{CVAE}}(x, y; \theta, \phi) = -KL(q_{\phi}(z | x, y) || p_{\theta}(z | x)) + \frac{1}{L}\sum_{l=1}^{L}\log p_{\theta}(y | x, z^{(l)})">

where $\bf z^{(l)}$ is a Gaussian latent variable product, and $L$ is the number of samples (or particles in Pyro nomenclature).
We call this model conditional variational auto-encoder (CVAE). 
The CVAE is composed of multiple MLPs, such as **recognition network** $q_{\phi}({\bf z} | \bf{x, y})$, **(conditional) prior network** $p_{\theta}(\bf{z} | \bf{x})$, and **generation network** $p_{\theta}(\bf{y} | \bf{x, z})$. 
In designing the network architecture, we build the network components of the CVAE **on top of the baseline NN**. 
Specifically, as shown in (d) above, not only the direct input $\bf x$, but also the initial guess $\hat{y}$ made by the NN are fed into the prior network. 

Pyro makes it really easy to translate this architecture into code. 
The recognition network and the (conditional) prior network are encoders from the traditional VAE setting, while the generation network is the decoder:

```Python
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
```

## Evaluating the results
For qualitative analysis, we visualize the generated output samples in the next figure. As we can see, the baseline NNs can only make a single deterministic prediction, and as a result the output looks blurry and doesn’t look realistic in many cases. In contrast, the samples generated by the CVAE models are more realistic and diverse in shape; sometimes they can even change their identity (digit labels), such as from 3 to 5 or from 4 to 9, and vice versa.

<img src="https://i.ibb.co/Jvz9v71/cvae-q1.png" alt="image1" width="400">

We also provide a quantitative evidence by estimating the marginal conditional log-likelihoods (CLLs) in next table. 

|                    | 1 quadrant | 2 quadrants | 3 quadrants |
|--------------------|------------|-------------|-------------|
| NN (baseline)      | 100.4      | 61.9        | 25.4        |
| CVAE (Monte Carlo) | 71.8       | 51.0        | 24.2        |
| Performance gap    | 28.6       | 10.9        | 1.2         |

We achieved similar results to the ones achieved by the authors in the paper. We trained only for 50 epochs with early stopping patience of 3 epochs; to improve the results, we could leave the algorithm training for longer. Nevertheless, we can observe the same effect shown in the paper: **the estimated CLLs of the CVAE significantly outperforms the baseline NN**.

See the full code on [Github](https://github.com/ucals/cvae).

## References

[1] `Learning Structured Output Representation using Deep Conditional Generative Models`,<br/>&nbsp;&nbsp;&nbsp;&nbsp;
Kihyuk Sohn, Xinchen Yan, Honglak Lee




