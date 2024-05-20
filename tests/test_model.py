"""Tests related to the mnist_mlops.models.model.py file."""
import pytest

from hydra import initialize, compose
import torch

from mnist_mlops.models.model import MyCNN

from tests import _HYDRA_CONFIG

@pytest.fixture
def config():
    # print(_HYDRA_CONFIG)
    # with initialize(version_base=None, config_path=_HYDRA_CONFIG):
    with initialize(version_base=None, config_path="../config"):
        config = compose(config_name="default_config")
    return config



def test_model(config): 
    """Test if the defined CNN has the correct shape."""
    architechture = config.experiment['architechture']

    model = MyCNN(
        input_dim = architechture["input_dim"],
        first_dim = architechture["first_dim"],
        second_dim = architechture["second_dim"],
        third_dim = architechture["third_dim"],
        output_dim = architechture["output_dim"],
        dropout = architechture["dropout"]
    )
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)
