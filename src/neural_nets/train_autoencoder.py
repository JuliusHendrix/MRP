import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# own modules
from dataset_utils import VulcanDataset
from autoencoder import AutoEncoder


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MRP_dir = str(Path(script_dir).parents[1])
    dataset_dir = os.path.join(MRP_dir, 'data/dataset')

    vulcan_dataset = VulcanDataset(dataset_dir)
    dataloader = DataLoader(vulcan_dataset, batch_size=4,
                            shuffle=True,
                            num_workers=0)

    # Initialize model with double precision
    model = AutoEncoder().double()

    # Create loss function
    loss_function = nn.MSELoss()

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.1)

    epochs = 20
    outputs = []
    losses = []
    for epoch in range(epochs):
        for example in dataloader:
            # extract inputs
            height_arr = example['inputs']['height_arr']
            top_flux = example['inputs']['top_flux']
            gravity = example['inputs']['gravity']

            # output of Autoencoder
            reconstructed = model(height_arr, top_flux, gravity)

            # TODO: custom loss function
            # Calculating the loss function
            loss = loss_function(reconstructed, height_arr)

            # The gradients are set to zero,
            # the the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Storing the losses in a list for plotting
            losses.append(loss)

    # for i_batch, example_batched in enumerate(dataloader):
    #
    #     vis_dict = {}
    #
    #     for key, value in example_batched.items():
    #         sub_dict = {}
    #         for k, v in value.items():
    #             sub_dict.update({k: v.size()})
    #
    #         vis_dict.update({key: sub_dict})
    #
    #     print(vis_dict)
    #
    #     break


if __name__ == "__main__":
    main()
