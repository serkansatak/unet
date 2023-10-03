import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torcheval.metrics.image import PeakSignalNoiseRatio
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import Config
from src.utils import save_tensor_images


class ModelTrainer(object):
    """
    ModelTrainer class encapsulates all the logic necessary for training the model
    """

    model: nn.Module
    criterion: nn.Module
    logger: SummaryWriter
    device: str
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler

    def __init__(self, config: Config):
        self.config = config
        self.logger = SummaryWriter(self.config.general.log_dir)

        self.criterion = self.config.training.criterion

        self.model = self.config.general.model(
            self.config.dataset.input_channels,
            self.config.dataset.output_channels,
            self.config.dataset.batch_norm,
        )

        self.device = torch.device(self.config.general.device)

        self.model.to(
            self.device
        )  # Move the model to the device specified in the config file

        if self.config.general.checkpoint:
            self.model.load_state_dict(
                torch.load(self.config.general.checkpoint, map_location=self.device)
            )
        else:
            self.model.apply(ModelTrainer.init_kaiming_normal)

        print("\nModel Summary")
        summary(
            self.model,
            (
                self.config.dataset.input_channels,
                self.config.dataset.image_size,
                self.config.dataset.image_size,
            ),
            device=self.config.general.device,
        )

        if self.config.training.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.config.training.lr
            )
        elif self.config.training.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.training.lr,
                momentum=self.config.training.momentum,
                weight_decay=self.config.training.weight_decay,
            )
        else:
            raise ValueError("Unknown optimizer")

        if self.config.training.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.config.training.num_epochs
            )
        elif self.config.training.scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.step_size,
                gamma=self.config.training.gamma,
            )
        else:
            raise ValueError("Unknown scheduler")

    def train(self):
        """
        Train the model
        :param train_loader: training data loader
        :param val_loader: validation data loader
        :param epochs: number of epochs to train
        :param save_path: path to save the model
        """

        print("Starting Training")

        psnr = PeakSignalNoiseRatio()

        for epoch in range(self.config.training.num_epochs):
            total_loss = 0
            total_psnr = 0
            val_total_loss = 0
            val_total_psnr = 0

            # Train the model
            self.model.train()
            with tqdm(
                total=len(self.config.dataloader.train),
                desc=f"Epoch {epoch+1}/{self.config.training.num_epochs}",
            ) as pbar:
                for batch in self.config.dataloader.train:
                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    # Zero the parameter gradients
                    self.optimizer.zero_grad()
                    # Forward pass
                    outputs = self.model(inputs)
                    # Compute the loss
                    loss = self.criterion(outputs, targets)
                    # Backpropagation and optimization
                    loss.backward()
                    total_loss += loss.item()
                    # Update the parameters
                    self.optimizer.step()
                    self.scheduler.step()
                    # Compute the PSNR
                    psnr_score = psnr.update(outputs, targets).compute()
                    total_psnr += psnr_score
                    # Update the progress bar
                    pbar.update(1)
                    pbar.set_postfix({"Training Loss": loss.item(), "PSNR": psnr_score})
                average_psnr = total_psnr / len(self.config.dataloader.train)
                pbar.set_postfix(
                    {"Training Loss": total_loss, "Average Training PSNR": average_psnr}
                )

            print(f"new learning rate: {self.scheduler.get_last_lr()[-1]}")

            # Validation
            self.model.eval()
            with tqdm(
                total=len(self.config.dataloader.val),
                desc=f"Epoch {epoch+1}/{self.config.training.num_epochs}",
            ) as pbar:
                for batch in self.config.dataloader.val:
                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_total_loss += loss.item()
                    psnr_score = psnr.update(outputs, targets).compute()
                    val_total_psnr += psnr_score

                    pbar.set_postfix(
                        {"Validation Loss": loss.item(), "PSNR": psnr_score}
                    )
                average_val_psnr = val_total_psnr / len(self.config.dataloader.val)
                pbar.set_postfix(
                    {
                        "Validation Loss": val_total_loss,
                        "Average Validation PSNR": average_val_psnr,
                    }
                )

            # Log scalar metrics
            self.logger.add_scalar("Loss", total_loss, global_step=epoch)
            self.logger.add_scalar("PSNR", average_psnr, global_step=epoch)
            self.logger.add_scalar("Validation Loss", val_total_loss, global_step=epoch)
            self.logger.add_scalar(
                "Validation PSNR", average_val_psnr, global_step=epoch
            )

            print(
                f"Epoch [{epoch+1}/{self.config.training.num_epochs}], Loss: {loss.item():.4f}"
            )

            if self.config.dataset.reference_image:
                # Save the reference image
                self.save_reference_image(epoch)

            # Save the model
            if self.config.general.save_model:
                torch.save(
                    self.model.state_dict(),
                    f"{os.path.join(self.config.general.model_save_path,f'{self.config.general.model_name}_{epoch+1}.pth')}",
                )

        print("Finished Training")

    @staticmethod
    def init_kaiming_normal(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def save_reference_image(self, epoch):
        if self.config.dataset.reference_image:
            transform = self.config.dataset.transform

            input = (
                transform(
                    Image.open(
                        os.path.join(
                            self.config.dataset.path,
                            self.config.dataset.lq_folder,
                            self.config.dataset.reference_image,
                        )
                    )
                )
                .to(self.device)
                .unsqueeze(0)
            )
            output = self.model(input)
            target = (
                transform(
                    Image.open(
                        os.path.join(
                            self.config.dataset.path,
                            self.config.dataset.gt_folder,
                            self.config.dataset.reference_image,
                        )
                    )
                )
                .to(self.device)
                .unsqueeze(0)
            )

            save_path = os.path.join(
                self.config.general.image_save_path, f"{epoch+1}.png"
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
                save_path,
            )
