import torch
from torch import nn

class CNN(nn.Module):
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
        # dropout: float

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

        self.model = nn.Sequential(
            nn.Conv2d(input_dim, first_dim, 3, 1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Conv2d(first_dim, second_dim, 3, 1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Conv2d(second_dim, third_dim, 3, 1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Linear(third_dim, output_dim),
            # nn.Dropout(dropout),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.model(x)
