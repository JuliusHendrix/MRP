import os
import sys
from pathlib import Path
from torchviz import make_dot
from torch.utils.data import DataLoader

# own modules
from dataset_utils import VulcanDataset
from autoencoder_old import AutoEncoder


def main():
    # setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MRP_dir = str(Path(script_dir).parents[1])
    dataset_dir = os.path.join(MRP_dir, 'data/christmas_dataset/dataset')

    # dataset loader
    vulcan_dataset = VulcanDataset(dataset_dir)

    dataloader = DataLoader(vulcan_dataset, batch_size=1,
                              shuffle=True,
                              num_workers=0)

    model = AutoEncoder().double()

    for example in dataloader:

        # extract inputs
        y_mix_ini = example['inputs']['y_mix_ini']
        top_flux = example['inputs']['top_flux']
        Tco = example['inputs']['Tco']
        Pco = example['inputs']['Pco']
        g = example['inputs']['g']
        gravity = example['inputs']['gravity']

        # output of autoencoder
        # decoded_y_mix_ini, decoded_top_flux, decoded_Tco, decoded_Pco, decoded_g, decoded_gravity = \
        y = model(y_mix_ini, top_flux, Tco, Pco, g, gravity)

        make_dot(y, params=dict(model.named_parameters())).render("model_images/autoencoder", format="png")

        break


if __name__ == "__main__":
    main()
