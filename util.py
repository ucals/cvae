import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.utils import make_grid
from baseline import BaselineNet
from mnist import *


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


def visualize(device, dataloaders, model, num_images):
    batch = next(iter(dataloaders['val']))
    inputs = batch['input'].to(device)
    outputs = batch['output'].to(device)
    originals = batch['original'].to(device)

    with torch.no_grad():
        preds = model(inputs).view(outputs.shape)

    # Predictions are only made in the pixels not masked. This completes
    # the input quadrant with the prediction for the missing quadrants, for
    # visualization purpose
    preds[outputs == -1] = inputs[outputs == -1]

    originals_slice = originals[:num_images]
    preds_slice = preds.unsqueeze(1)[:num_images]
    grid_tensor = torch.cat([originals_slice, preds_slice], dim=0)
    grid_tensor = make_grid(grid_tensor, nrow=num_images, padding=0)

    for i in range(num_images - 1):
        grid_tensor[:, :, (i + 1) * 28] = 0.2
    grid_tensor[:, 28, :] = 0.2

    imshow(grid_tensor)


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == '__main__':
    model = BaselineNet(500, 500)
    model.load_state_dict(torch.load('../data/models/baseline_net_q1.pth'))
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    datasets, dataloaders, dataset_sizes = get_data(
        num_quadrant_inputs=1,
        batch_size=32
    )

    visualize(device, dataloaders, model, 10)












