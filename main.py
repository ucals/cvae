import torch
import baseline as baseline
from util import get_data


if __name__ == '__main__':
    # Dataset
    datasets, dataloaders, dataset_sizes = get_data(
        num_quadrant_inputs=1,
        batch_size=32
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Train baseline
    baseline_net = baseline.train(
        device=device,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        learning_rate=1e-3,
        num_epochs=30,
        early_stop_patience=3,
        model_path='../data/models/baseline_net_q1.pth'
    )
