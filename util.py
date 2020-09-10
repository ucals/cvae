import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from baseline import BaselineNet
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


def visualize(num_images, num_samples):
    # Load models pre-trained models
    pre_trained_baseline = BaselineNet(500, 500)
    pre_trained_baseline.load_state_dict(
        torch.load('../data/models/baseline_net_q1.pth'))
    pre_trained_baseline.eval()

    pre_trained_cvae = CVAE(200, 500, 500, pre_trained_baseline)
    pre_trained_cvae.load('../data/models/cvae_net_q1')

    # Load sample data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets, dataloaders, dataset_sizes = get_data(
        num_quadrant_inputs=1,
        batch_size=num_images
    )
    batch = next(iter(dataloaders['val']))
    inputs = batch['input'].to(device)
    outputs = batch['output'].to(device)
    originals = batch['original'].to(device)

    # Make predictions
    with torch.no_grad():
        baseline_preds = pre_trained_baseline(inputs).view(outputs.shape)
        cvae_preds = pre_trained_cvae.predict(inputs, num_samples=num_samples)

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


if __name__ == '__main__':
    visualize(num_images=10, num_samples=10)









