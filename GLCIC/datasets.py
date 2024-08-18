import torch
import numpy as np
import os
from torch.utils.data import Dataset

class NpyDataset(Dataset):
    def __init__(self, velocity_dir, abcd_dir=None, transform=None):
        self.velocity_files = [os.path.join(velocity_dir, f) for f in os.listdir(velocity_dir) if f.endswith('.npy')]
        self.abcd_dir = abcd_dir
        self.transform = transform

    def __len__(self):
        return len(self.velocity_files)

    def __getitem__(self, idx):
        velocity = np.load(self.velocity_files[idx])
        velocity = torch.from_numpy(velocity).float()

        if self.abcd_dir:
            abcd = np.load(os.path.join(self.abcd_dir, os.path.basename(self.velocity_files[idx])))
            abcd = torch.from_numpy(abcd).float()
            return velocity, abcd
        else:
            return velocity
