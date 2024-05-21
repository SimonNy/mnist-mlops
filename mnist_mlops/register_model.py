import mlflow
import mlflow.pytorch
import torch
from omegaconf import OmegaConf

from models.model import MyCNN


def load_model_from_checkpoint(model_class, checkpoint_path, config):
    """Load the model from a checkpoint file.

    Args:
    ----
        model_class (type): The class of the model to be loaded.
        checkpoint_path (str): Path to the checkpoint file.
        config (DictConfig): Configuration dictionary containing model parameters.

    Returns:
    -------
        torch.nn.Module: The loaded model.

    """
    model = model_class(**config["architecture"])
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    return model


def register_model(cfg_path: str, model_path: str):
    """Register the trained model with MLflow.

    Args:
    ----
        cfg_path (str): Path to the configuration file.
        model_path (str): Path to the trained model file.

    """
    # Load configuration
    config = OmegaConf.load(cfg_path)
    print(config)
    # Load the model from the checkpoint
    model = load_model_from_checkpoint(MyCNN, model_path, config)

    mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run() as run:
        # Log the model
        mlflow.pytorch.log_model(
            pytorch_model=model, artifact_path="model", registered_model_name=config.mlflow.model_registry
        )
        print(f"Model registered in experiment: {config.mlflow.experiment_name} with run ID: {run.info.run_id}")


if __name__ == "__main__":
    # Example usage
    config_path = "config/experiment/experiment1.yaml"
    model_path = "models/best-checkpoint.ckpt"  # Adjust this path if necessary

    register_model(cfg_path=config_path, model_path=model_path)
