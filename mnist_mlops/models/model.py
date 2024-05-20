"""Defines a basic CNN to classify the mnist numbers."""

import torch
from torch import nn


class MyCNN(torch.nn.Module):
    """Basic Convolutional neural network class."""

    def __init__(
        self, 
        input_dim: int, 
        first_dim: int, 
        second_dim: int, 
        third_dim: int, 
        output_dim: int, 
        dropout: int
    ) -> None:
        """Initialize model."""
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
        ----
            x: input tensor expected to be of shape [N,in_features]

        Returns:
        -------
            Output tensor with shape [N,out_features]

        """
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x
