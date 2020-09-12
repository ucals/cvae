import pandas as pd
import torch
import baseline
import cvae_model as cvae
from util import get_data, visualize, generate_table


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = []

    for num_quadrant_inputs in [1, 2, 3]:
        maybes = 's' if num_quadrant_inputs > 1 else ''
        print(f'Training with {num_quadrant_inputs} quadrant{maybes} as input...')

        # Dataset
        datasets, dataloaders, dataset_sizes = get_data(
            num_quadrant_inputs=num_quadrant_inputs,
            batch_size=128
        )

        # Train baseline
        baseline_net = baseline.train(
            device=device,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            learning_rate=1e-3,
            num_epochs=50,
            early_stop_patience=3,
            model_path=f'baseline_net_q{num_quadrant_inputs}.pth'
        )

        # Train CVAE
        cvae_net = cvae.train(
            device=device,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            learning_rate=1e-3,
            num_epochs=100,
            early_stop_patience=3,
            model_path=f'cvae_net_q{num_quadrant_inputs}.pth',
            pre_trained_baseline_net=baseline_net
        )

        # Visualize conditional predictions
        visualize(
            device,
            baseline_net,
            cvae_net,
            num_images=10,
            num_samples=10,
            image_path=f'cvae_plot_q{num_quadrant_inputs}.png'
        )

        # Retrieve conditional log likelihood
        df = generate_table(
            baseline_net,
            cvae_net,
            col_name=f'{num_quadrant_inputs} quadrant{maybes}'
        )
        results.append(df)

    results = pd.concat(results, axis=1, ignore_index=True)
    results.to_csv('results.csv')


