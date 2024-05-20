"""Tests related to the mnist_mlops.models.model.py file."""
import pytest

from hydra import initialize, compose
import torch

from mnist_mlops.models.model import MyCNN

from tests import _HYDRA_CONFIG

@pytest.fixture
def hydra_config():
    # print(_HYDRA_CONFIG)
    # with initialize(version_base=None, config_path=_HYDRA_CONFIG):
    with initialize(version_base=None, config_path="config"):
        config = compose(config_name="default_config")
    return config



def test_model(hydra_config): 
    """Test if the defined CNN has the correct shape."""
    hparams = hydra_config.experiment

    model = MyCNN(
        input_dim = hparams["input_dim"],
        first_dim = hparams["first_dim"],
        second_dim = hparams["second_dim"],
        third_dim = hparams["third_dim"],
        output_dim = hparams["output_dim"],
        dropout = hparams["dropout"]
    )
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)
