# Purpose: Configuration file for the project
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch

from utils.dataset import BaseDataset
from utils.model import UNet


"""
The Config class is used to store all the configuration parameters for the project.

No need for extra yaml files or json files. Everything is stored in a single file.
"""


@dataclass
class DatasetConfig:
    path: str = "data"
    random_split: bool = True
    train_size: float = 0.8
    val_size: float = 0.2
    test_size: float = 0.2
    image_size: int = 384
    transform: transforms = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )
    train_dataset: BaseDataset = None
    val_dataset: BaseDataset = None
    test_dataset: BaseDataset = None
    input_channels: int = 3
    output_channels: int = 3


@dataclass
class DataLoaderConfig:
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    train: DataLoader = None
    val: DataLoader = None
    test: DataLoader = None


@dataclass
class TrainingConfig:
    criterion: str = nn.MSELoss()  # "cross_entropy"
    optimizer: str = "adam"
    lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0001
    num_epochs: int = 30


@dataclass
class GeneralConfig:
    seed: int = 42
    save_model: bool = True
    save_path: str = "./models"
    model_name: str = "unet_marine_snow"
    model: nn.Module = UNet
    checkpoint: str | None = None
    mode: str = "train"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    general: GeneralConfig = GeneralConfig()
    training: TrainingConfig = TrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    dataloader: DataLoaderConfig = DataLoaderConfig()

    def __post_init__(self):
        if self.general.mode == "train":
            self.dataset.train_dataset = BaseDataset(
                self.dataset.path, self.dataset.transform
            )

            if self.dataset.random_split:
                self.dataset.train_size = int(
                    len(self.dataset.train_dataset) * self.dataset.train_size
                )
                self.dataset.val_size = int(
                    len(self.dataset.train_dataset) * self.dataset.val_size
                )

                self.dataset.train_dataset, self.dataset.val_dataset = random_split(
                    self.dataset.train_dataset,
                    [self.dataset.train_size, self.dataset.val_size],
                )

        if self.general.mode == "test":
            self.dataset.test_dataset = BaseDataset(
                self.dataset.path, self.dataset.transform
            )

        if self.general.mode == "train":
            self.dataloader.train = DataLoader(
                dataset=self.dataset.train_dataset,
                batch_size=self.dataloader.batch_size,
                shuffle=self.dataloader.shuffle,
                num_workers=self.dataloader.num_workers,
            )
            self.dataloader.val = DataLoader(
                dataset=self.dataset.val_dataset,
                batch_size=self.dataloader.batch_size,
                shuffle=self.dataloader.shuffle,
                num_workers=self.dataloader.num_workers,
            )
        if self.general.mode == "test":
            self.dataloader.test = DataLoader(
                dataset=self.dataset.test_dataset,
                batch_size=self.dataloader.batch_size,
                shuffle=self.dataloader.shuffle,
                num_workers=self.dataloader.num_workers,
            )
