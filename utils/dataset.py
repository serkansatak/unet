import torch
import torch.nn as nn
import os
import numpy as np
import torchvision.transforms as transforms
import glob
from torch.utils.data import Dataset
from PIL import Image


# Custom torch dataset
class BaseDataset(Dataset):
    """
    Get files from lq and gt folders
    Apply transforms to both lq and gt images

    """

    def __init__(self, folder_path, gt_folder, lq_folder, transform=None):
        self.gt = os.path.join(folder_path, "gt")
        self.lq = os.path.join(folder_path, "lq")
        self.gt_files = sorted(glob.glob(os.path.join(self.gt, "*")))
        self.lq_files = sorted(glob.glob(os.path.join(self.lq, "*")))
        self.transform = transform

    def __len__(self):
        return len(self.lq_files)

    def __getitem__(self, index):
        return self.transform(Image.open(self.lq_files[index])), self.transform(
            Image.open(self.gt_files[index])
        )
