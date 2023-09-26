import os
import torch
import torch.nn as nn
from utils.config import Config
from tqdm import tqdm


class ModelTrainer(object):
    """
    ModelTrainer class encapsulates all the logic necessary for training the model
    """

    def __init__(self, config: Config):
        self.config = config
        self.model = self.config.general.model

    def train(self):
        """
        Train the model
        :param train_loader: training data loader
        :param val_loader: validation data loader
        :param epochs: number of epochs to train
        :param save_path: path to save the model
        """

        criterion = self.config.training.criterion

        model = self.config.general.model(
            self.config.dataset.input_channels, self.config.dataset.output_channels
        )

        if self.config.general.checkpoint:
            model.load_state_dict(torch.load(self.config.general.checkpoint))
        else:
            model.apply(ModelTrainer.init_kaiming_normal)

        model.to(
            self.config.general.device
        )  # Move the model to the device specified in the config file

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
            model.train()
            with tqdm(
                total=len(self.config.dataloader.train),
                desc=f"Epoch {epoch+1}/{self.config.training.num_epochs}",
            ) as pbar:
                for batch in self.config.dataloader.train:
                    inputs, targets = batch
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    # Forward pass
                    outputs = model(inputs)
                    # Compute the loss
                    loss = criterion(outputs, targets)
                    # Backpropagation and optimization
                    loss.backward()
                    # Update the parameters
                    optimizer.step()
                    # Update the progress bar
                    pbar.update(1)
                    pbar.set_postfix({"Training Loss": loss.item()})
            with tqdm(
                total=len(self.config.dataloader.val),
                desc=f"Epoch {epoch+1}/{self.config.training.num_epochs}",
            ) as pbar:
                for batch in self.config.dataloader.val:
                    inputs, targets = batch
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    pbar.set_postfix({"Validation Loss": loss.item()})

            print(
                f"Epoch [{epoch+1}/{self.config.training.num_epochs}], Loss: {loss.item():.4f}"
            )

            # Save the model
            if self.config.general.save_model:
                torch.save(
                    model.state_dict(),
                    f"{os.path.join(self.config.general.save_path,f'{self.config.general.model_name}_{epoch+1}.pth')}",
                )

        # Training loop (assuming you have a DataLoader for your dataset)
        for epoch in range(self.config.training.num_epochs):
            for data in self.config.dataloader.train:
                inputs, targets = data

                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = model(inputs)
                # Compute the loss
                loss = criterion(outputs, targets)
                print(loss)
                # Backpropagation and optimization
                loss.backward()
                optimizer.step()
            # Print the loss for this epoch
            print(
                f"Epoch [{epoch+1}/{self.config.training.num_epochs}], Loss: {loss.item():.4f}"
            )

            # Save the model
            if self.config.general.save_model:
                torch.save(
                    model.state_dict(),
                    f"{os.path.join(self.config.general.save_path,f'{self.config.general.model_name}_{epoch+1}.pth')}",
                )
        print("Finished Training")

    @staticmethod
    def init_kaiming_normal(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
