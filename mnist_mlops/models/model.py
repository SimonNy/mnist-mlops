"""Defines a basic CNN to classify the mnist numbers."""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class MyCNN(pl.LightningModule):
    """Implement a simple Convolutional Neural Network (CNN) model implemented using PyTorch Lightning.

    Attributes
    ----------
        conv1 (nn.Conv2d): First convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        conv3 (nn.Conv2d): Third convolutional layer.
        fc1 (nn.Linear): First fully connected layer.
        dropout: Uses dropout.

    """

    def __init__(
        self,
        input_dim: int,
        first_dim: int,
        second_dim: int,
        third_dim: int,
        output_dim: int,
        dropout: float,
        lr: float,
    ) -> None:
        """Initialize the CNN model with three convolutional layers and one fully connected layers.

        Args:
        ----
            input_dim (int): Input dimensions.
            first_dim (int): First layers input dimensions.
            second_dim (int): Second layers input dimensions.
            third_dim (int): Third layers input dimensions.
            output_dim (int): Fully connected layers output dimensions.
            dropout (float): Dropout rate.
            lr (float): Learningrate for the optimizer.

        """
        super().__init__()

        self.conv1 = nn.Conv2d(input_dim, first_dim, 3, 1)
        self.conv2 = nn.Conv2d(first_dim, second_dim, 3, 1)
        self.conv3 = nn.Conv2d(second_dim, third_dim, 3, 1)
        self.fc1 = nn.Linear(third_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass of the model.

        Args:
        ----
            x: input tensor expected to be of shape [N,in_features]

        Returns:
        -------
            Output tensor with shape [N,out_features]

        """
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x

    def training_step(self, batch, batch_idx):
        """Execute one training step for the model.

        Args:
        ----
            batch (tuple): A tuple containing the input data and target labels.
            batch_idx (int): Index of the batch.

        Returns:
        -------
            torch.Tensor: Computed loss for the batch.

        """
        x, y = batch
        print(f"x shape: {x.shape}, y shape: {y.shape}")
        predicts = self(x)
        loss = self.loss_fn(predicts, y)
        acc = (y == predicts.argmax(dim=-1)).float().mean()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc)
        return loss

    def configure_optimizers(self):
        """Configure the optimizer for training.

        Returns
        -------
            torch.optim.Optimizer: The optimizer to be used for training.

        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    model = MyCNN()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
