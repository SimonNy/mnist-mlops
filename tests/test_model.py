"""Tests related to the mnist_mlops.models.model.py file."""

import pytest
import torch
from hydra import compose, initialize

from mnist_mlops.models.model import MyCNN


@pytest.fixture
def config():
    """Read the config file for testing."""
    with initialize(version_base=None, config_path="../config"):
        config = compose(config_name="default_config")
    return config


def test_model(config):
    """Test if the defined CNN has the correct shape."""
    model_params = config.experiment["architecture"]

    model = MyCNN(**model_params)
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)
