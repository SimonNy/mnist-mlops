"""Tests related to the mnist_mlops.models.model.py file."""

import torch

from mnist_mlops.models.model import MyCNN


def test_model():
    """Test if the defined CNN has the correct shape."""
    model = MyCNN()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)
