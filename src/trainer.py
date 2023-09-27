import os
import torch
import torch.nn as nn
from src.config import Config
from src.utils import save_tensor_images
from tqdm import tqdm
from torcheval.metrics.image import PeakSignalNoiseRatio
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


class ModelTrainer(object):
    """
    ModelTrainer class encapsulates all the logic necessary for training the model
    """

    def __init__(self, config: Config):
        self.config = config
        self.model = self.config.general.model
        self.logger = SummaryWriter(self.config.general.log_dir)

    def train(self):
        """
        Train the model
        :param train_loader: training data loader
        :param val_loader: validation data loader
        :param epochs: number of epochs to train
        :param save_path: path to save the model
        """

        print("Starting Training")

        criterion = self.config.training.criterion
        psnr = PeakSignalNoiseRatio()

        model = self.config.general.model(
            self.config.dataset.input_channels, self.config.dataset.output_channels
        )

        if self.config.general.checkpoint:
            model.load_state_dict(torch.load(self.config.general.checkpoint))
        else:
            model.apply(ModelTrainer.init_kaiming_normal)

        device = torch.device(self.config.general.device)

        model.to(device)  # Move the model to the device specified in the config file

        if self.config.training.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.training.lr)
        elif self.config.training.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.config.training.lr,
                momentum=self.config.training.momentum,
                weight_decay=self.config.training.weight_decay,
            )
        else:
            raise ValueError("Unknown optimizer")

        for epoch in range(self.config.training.num_epochs):
            total_loss = 0
            total_psnr = 0
            val_total_loss = 0
            val_total_psnr = 0

            # Train the model
            model.train()
            with tqdm(
                total=len(self.config.dataloader.train),
                desc=f"Epoch {epoch+1}/{self.config.training.num_epochs}",
            ) as pbar:
                for batch in self.config.dataloader.train:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    # Forward pass
                    outputs = model(inputs)
                    # Compute the loss
                    loss = criterion(outputs, targets)
                    # Backpropagation and optimization
                    loss.backward()
                    total_loss += loss.item()
                    # Update the parameters
                    optimizer.step()
                    # Compute the PSNR
                    psnr_score = psnr.update(outputs, targets).compute()
                    total_psnr += psnr_score
                    # Update the progress bar
                    pbar.update(1)
                    pbar.set_postfix({"Training Loss": loss.item()})
                average_psnr = total_psnr / len(self.config.dataloader.train)
                pbar.set_postfix(
                    {"Training Loss": total_loss, "Average Training PSNR": average_psnr}
                )

            # Validation
            model.eval()
            with tqdm(
                total=len(self.config.dataloader.val),
                desc=f"Epoch {epoch+1}/{self.config.training.num_epochs}",
            ) as pbar:
                for batch in self.config.dataloader.val:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_total_loss += loss.item()
                    psnr_score = psnr.update(outputs, targets).compute()
                    val_total_psnr += psnr_score

                    pbar.set_postfix(
                        {"Validation Loss": loss.item(), "PSNR": psnr.value()}
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
                self.save_reference_image()

            # Save the model
            if self.config.general.save_model:
                torch.save(
                    model.state_dict(),
                    f"{os.path.join(self.config.general.model_save_path,f'{self.config.general.model_name}_{epoch+1}.pth')}",
                )

        print("Finished Training")

    @staticmethod
    def init_kaiming_normal(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

    def save_reference_image(self, epoch):
        if self.config.dataset.reference_image:
            input = transforms.ToTensor(
                Image.open(
                    os.path.join(
                        self.config.dataset.path,
                        self.config.dataset.lq_folder,
                        self.config.dataset.reference_image,
                    )
                )
            )
            output = self.model(input)
            target = transforms.ToTensor(
                Image.open(
                    os.path.join(
                        self.config.dataset.path,
                        self.config.dataset.gt_folder,
                        self.config.dataset.reference_image,
                    )
                )
            )

            save_path = os.path.join(
                self.config.general.image_save_path, f"{epoch+1}.png"
            )

            save_tensor_images(input, output, target, save_path)
