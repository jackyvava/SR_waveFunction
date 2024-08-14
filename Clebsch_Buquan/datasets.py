import os
import numpy as np
import torch.utils.data as data
import torch
# import torch.utils.data as data
from torch.utils.data import DataLoader


class NpyDataset(data.Dataset):
    def __init__(self, velocity_dir, abcd_dir=None, transform=None, mask_ratio=0.2):
        super(NpyDataset, self).__init__()
        self.velocity_dir = os.path.expanduser(velocity_dir)
        self.abcd_dir = os.path.expanduser(abcd_dir) if abcd_dir else None
        self.transform = transform
        self.mask_ratio = mask_ratio
        self.velocity_files = self.__load_npyfiles_from_dir(self.velocity_dir)
        if self.abcd_dir:
            self.abcd_files = self.__load_npyfiles_from_dir(self.abcd_dir)
        else:
            self.abcd_files = None

    def __len__(self):
        return len(self.velocity_files)

    def __getitem__(self, index):
        velocity = np.load(self.velocity_files[index])

        # 生成掩码并应用
        mask = np.random.rand(*velocity.shape) < self.mask_ratio
        masked_velocity = velocity.copy()
        masked_velocity[mask] = 0

        # 转换为PyTorch张量
        masked_velocity = torch.from_numpy(masked_velocity).float()
        velocity = torch.from_numpy(velocity).float()

        if self.abcd_files:
            abcd = np.load(self.abcd_files[index])
            abcd = torch.from_numpy(abcd).float()
            if self.transform:
                masked_velocity = self.transform(masked_velocity)
                abcd = self.transform(abcd)
            return masked_velocity, abcd, velocity
        else:
            if self.transform:
                masked_velocity = self.transform(masked_velocity)
            return masked_velocity, velocity

    def __is_npyfile(self, filepath):
        filepath = os.path.expanduser(filepath)
        return os.path.isfile(filepath) and filepath.endswith('.npy')

    def __load_npyfiles_from_dir(self, dirpath):
        npyfiles = []
        dirpath = os.path.expanduser(dirpath)
        for path in os.listdir(dirpath):
            path = os.path.join(dirpath, path)
            if not self.__is_npyfile(path):
                continue
            npyfiles.append(path)
        return npyfiles
