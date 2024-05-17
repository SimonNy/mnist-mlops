import torch

from mnist_mlops.models.model import MyCNN


def test_model():
    model = MyCNN()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)