import os
import pickle
import torch
from torch.utils.data import Dataset


class VulcanDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

        index_file = os.path.join(dataset_dir, 'index_dict.pkl')
        with open(index_file, 'rb') as f:
            self.index_dict = pickle.load(f)

    def __len__(self):
        return len(self.index_dict)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = f'{idx:04}.pt'
        example = torch.load(os.path.join(self.dataset_dir, filename))

        return example
