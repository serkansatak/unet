import torch
import torch.nn as nn
import os
from PIL import Image
from datetime import datetime

from .utils import save_tensor_images
from .config import Config
from torchsummary import summary


class Tester:
    model: nn.Module
    device: str

    def __init__(self, config: Config):
        self.config = config
        self.model = self.config.general.model(
            self.config.dataset.input_channels,
            self.config.dataset.output_channels,
            self.config.dataset.batch_norm,
        )
        self.device = torch.device(self.config.general.device)
        self.model.to(self.device)

        summary(
            model=self.model,
            input_size=(3, 384, 384),
            device=self.config.general.device,
        )

        if self.config.general.checkpoint:
            self.model.load_state_dict(
                torch.load(self.config.general.checkpoint, map_location=self.device)
            )

    def test_single(self, image_path: str):
        self.model.eval()
        pth, ext = os.path.splitext(image_path)
        transform = self.config.dataset.transform
        input = transform(Image.open(image_path)).to(self.device).unsqueeze(0)
        output = self.model(input)
        target = (
            transform(Image.open(self.config.general.single_test_gt))
            .to(self.device)
            .unsqueeze(0)
        )
        save_tensor_images(
            [
                input.reshape(
                    self.config.dataset.input_channels,
                    self.config.dataset.image_size,
                    self.config.dataset.image_size,
                ),
                output.reshape(
                    self.config.dataset.input_channels,
                    self.config.dataset.image_size,
                    self.config.dataset.image_size,
                ),
                target.reshape(
                    self.config.dataset.input_channels,
                    self.config.dataset.image_size,
                    self.config.dataset.image_size,
                ),
            ],
            f"{pth}_output{ext}",
        )

    def test_batch(self):
        self.model.eval()

        self.config.dataloader.test.batch_size = 1

        test_save_path = os.path.join(
            self.config.general.image_save_path,
            f"{self.config.general.model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        )

        test_save_path = os.makedirs(
            test_save_path, exist_ok=True
        )  # Create save path if not exists

        for i, (input, target) in enumerate(self.config.dataloader.test):
            input = input.to(self.device)
            target = target.to(self.device)

            if input.shape[0] != 1:
                input.unsqueeze_(0)
                target.unsqueeze_(0)

            output = self.model(input)

            save_tensor_images(
                [
                    input.reshape(
                        self.config.dataset.input_channels,
                        self.config.dataset.image_size,
                        self.config.dataset.image_size,
                    ),
                    output.reshape(
                        self.config.dataset.input_channels,
                        self.config.dataset.image_size,
                        self.config.dataset.image_size,
                    ),
                    target.reshape(
                        self.config.dataset.input_channels,
                        self.config.dataset.image_size,
                        self.config.dataset.image_size,
                    ),
                ],
                f"{test_save_path}/{i}.png",
            )
