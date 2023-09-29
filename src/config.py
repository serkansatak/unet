# Purpose: Configuration file for the project
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch

from src.dataset import BaseDataset
from src.model import UNet


"""
The Config class is used to store all the configuration parameters for the project.

No need for extra yaml files or json files. Everything is stored in a single file.
"""


@dataclass
class DatasetConfig:
    path: str = "data"
    gt_folder: str = "gt"
    lq_folder: str = "lq"
    random_split: bool = True
    train_size: float = 0.8
    val_size: float = 0.2
    test_size: float = 0.2
    image_size: int = 384
    input_channels: int = 3
    output_channels: int = 3
    transform: transforms = (
        transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
            ]
        )
        if input_channels == 1
        else transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )
    )
    train_dataset: BaseDataset = None
    val_dataset: BaseDataset = None
    test_dataset: BaseDataset = None
    reference_image: str | None = "36006812.png"


@dataclass
class DataLoaderConfig:
    batch_size: int = 24
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
    num_epochs: int = 500


@dataclass
class GeneralConfig:
    seed: int = 42
    save_model: bool = True
    model_save_path: str = "./models"
    image_save_path: str = "./images"
    model_name: str = "unet_marine_snow"
    model: nn.Module = UNet
    checkpoint: str | None = None  # "./models/pretrained.pth"
    mode: str = "train"  # "train" or "test"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: str = "./logs"
    single_test_file: str | None = "./data/lq/2450682.png"


@dataclass
class Config:
    general: GeneralConfig = GeneralConfig()
    training: TrainingConfig = TrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    dataloader: DataLoaderConfig = DataLoaderConfig()

    def __post_init__(self):
        if self.general.mode == "train":
            self.dataset.train_dataset = BaseDataset(
                self.dataset.path,
                self.dataset.gt_folder,
                self.dataset.lq_folder,
                self.dataset.transform,
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
                self.dataset.path,
                self.dataset.gt_folder,
                self.dataset.lq_folder,
                self.dataset.transform,
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

    def to_dict(self):
        general_dict = self.general.__dict__
        training_dict = self.training.__dict__
        dataset_dict = self.dataset.__dict__
        dataloader_dict = self.dataloader.__dict__
        return {
            "general": general_dict,
            "training": training_dict,
            "dataset": dataset_dict,
            "dataloader": dataloader_dict,
        }
