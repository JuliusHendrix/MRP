from torch.utils.data import Dataset
import os
import sys
from pathlib import Path
import glob
import pickle
import torch

# own modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = str(Path(script_dir).parents[1])
sys.path.append(src_dir)

from src.neural_nets.dataset_utils import copy_output_to_input


class VulcanDataset(Dataset):
    """
    Template for VULCAN dataset loader
    """
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

        index_file = os.path.join(dataset_dir, '../index_dict.pkl')
        with open(index_file, 'rb') as f:
            self.index_dict = pickle.load(f)


class SingleVulcanDataset(VulcanDataset):
    def __init__(self, dataset_dir, time_series_evaluation=False):
        super().__init__(dataset_dir)
        self.time_series_evaluation = time_series_evaluation

    def load_example(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = f'{idx:04}.pt'
        example = torch.load(os.path.join(self.dataset_dir, filename))

        if self.time_series_evaluation:
            example['outputs']['y_mixs'] = example['outputs']['y_mixs'][-1, ...]

        return example

    def __len__(self):
        return len(self.index_dict)

    def __getitem__(self, idx):
        example = self.load_example(idx)

        return example


class DoubleVulcanDataset(VulcanDataset):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)

    def load_example(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = f'{int(idx / 2):04}.pt'
        example = torch.load(os.path.join(self.dataset_dir, filename))

        if idx % 2 != 0:
            example = copy_output_to_input(example)

        return example

    def __len__(self):
        return 2 * len(self.index_dict)

    def __getitem__(self, idx):
        example = self.load_example(idx)

        return example


class MixingRatioVulcanDataset(DoubleVulcanDataset):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)

        # get species list
        spec_file = os.path.join(dataset_dir, '../species_list.pkl')
        with open(spec_file, 'rb') as f:
            spec_list = pickle.load(f)

        self.num_species = len(spec_list)

    def __len__(self):
        return 2 * len(self.index_dict) * self.num_species

    def __getitem__(self, idx):
        # convert idx to double idx
        double_idx = int(idx / self.num_species)

        # get example
        example = self.load_example(double_idx)

        # get specific species
        sp_idx = round((idx / self.num_species - double_idx) * self.num_species)

        species_mr = example['inputs']['y_mix_ini'][:, sp_idx]

        return {'species_mr': species_mr, 'sp_idx': sp_idx}

