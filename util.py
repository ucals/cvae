import matplotlib.pyplot as plt
import pandas as pd
from pyro.infer import Predictive, Trace_ELBO
from torchvision.utils import make_grid
from tqdm import tqdm
from baseline import BaselineNet, MaskedBCELoss
from cvae_model import CVAE
from mnist import *


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    space = np.ones((inp.shape[0], 50, inp.shape[2]))
    inp = np.concatenate([space, inp], axis=1)

    plt.imshow(inp)
    plt.text(0, 23, 'Inputs:')
    plt.text(0, 23 + 28 + 3, 'Truth:')
    plt.text(0, 23 + (28 + 3) * 2, 'NN:')
    plt.text(0, 23 + (28 + 3) * 3, 'CVAE:')
    plt.axis('off')

    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize(pre_trained_baseline, pre_trained_cvae, num_images, num_samples):
    # Load sample data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets, _, dataset_sizes = get_data(
        num_quadrant_inputs=1,
        batch_size=num_images
    )
    dataloader = DataLoader(datasets['val'], batch_size=num_images, shuffle=True)

    batch = next(iter(dataloader))
    inputs = batch['input'].to(device)
    outputs = batch['output'].to(device)
    originals = batch['original'].to(device)

    # Make predictions
    with torch.no_grad():
        baseline_preds = pre_trained_baseline(inputs).view(outputs.shape)

    predictive = Predictive(pre_trained_cvae.model,
                            guide=pre_trained_cvae.guide,
                            num_samples=num_samples)
    cvae_preds = predictive(inputs)['y'].view(num_samples, num_images, 28, 28)

    # Predictions are only made in the pixels not masked. This completes
    # the input quadrant with the prediction for the missing quadrants, for
    # visualization purpose
    baseline_preds[outputs == -1] = inputs[outputs == -1]
    for i in range(cvae_preds.shape[0]):
        cvae_preds[i][outputs == -1] = inputs[outputs == -1]

    # adjust tensor sizes
    inputs = inputs.unsqueeze(1)
    inputs[inputs == -1] = 1
    baseline_preds = baseline_preds.unsqueeze(1)
    cvae_preds = cvae_preds.view(-1, 28, 28).unsqueeze(1)

    # make grids
    inputs_tensor = make_grid(inputs, nrow=num_images, padding=0)
    originals_tensor = make_grid(originals, nrow=num_images, padding=0)
    separator_tensor = torch.ones((3, 5, originals_tensor.shape[-1]))
    baseline_tensor = make_grid(baseline_preds, nrow=num_images, padding=0)
    cvae_tensor = make_grid(cvae_preds, nrow=num_images, padding=0)

    # add vertical and horizontal lines
    for tensor in [originals_tensor, baseline_tensor, cvae_tensor]:
        for i in range(num_images - 1):
            tensor[:, :, (i + 1) * 28] = 0.3

    for i in range(num_samples - 1):
        cvae_tensor[:, (i + 1) * 28, :] = 0.3

    # concatenate all tensors
    grid_tensor = torch.cat([inputs_tensor, separator_tensor, originals_tensor,
                             separator_tensor, baseline_tensor,
                             separator_tensor, cvae_tensor], dim=1)
    # plot tensors
    imshow(grid_tensor)


def generate_table(pre_trained_baseline, pre_trained_cvae, col_name):
    # Load sample data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets, dataloaders, dataset_sizes = get_data(
        num_quadrant_inputs=1,
        batch_size=32
    )
    criterion = MaskedBCELoss(reduction='sum')
    loss_fn = Trace_ELBO(num_particles=100).differentiable_loss

    baseline_cll = 0.0
    cvae_mc_cll = 0.0
    num_preds = 0

    df = pd.DataFrame(index=['NN (baseline)', 'CVAE (Monte Carlo)'],
                      columns=[col_name])

    # Iterate over data.
    bar = tqdm(dataloaders['val'])
    for batch in bar:
        inputs = batch['input'].to(device)
        outputs = batch['output'].to(device)
        num_preds += 1

        # Compute negative log likelihood for the baseline NN
        with torch.no_grad():
            preds = pre_trained_baseline(inputs)
        baseline_cll += criterion(preds, outputs).item() / inputs.size(0)

        # Compute the negative conditional log likelihood for the CVAE
        cvae_mc_cll += loss_fn(pre_trained_cvae.model,
                               pre_trained_cvae.guide,
                               inputs, outputs).detach().item() / inputs.size(0)

    df.iloc[0, 0] = baseline_cll / num_preds
    df.iloc[1, 0] = cvae_mc_cll / num_preds
    return df


if __name__ == '__main__':
    # Load models pre-trained models
    pre_trained_baseline = BaselineNet(500, 500)
    pre_trained_baseline.load_state_dict(
        torch.load('../data/models/baseline_net_q1.pth',
                   map_location=torch.device('cpu')))
    pre_trained_baseline.eval()

    pre_trained_cvae = CVAE(200, 500, 500, pre_trained_baseline)
    pre_trained_cvae.load('../data/models/cvae_net_q1')

    # visualize(
    #     pre_trained_baseline,
    #     pre_trained_cvae,
    #     num_images=10,
    #     num_samples=10
    # )

    df = generate_table(
        pre_trained_baseline,
        pre_trained_cvae,
        col_name='1 quadrant'
    )
    print(df)









