"""MLflow pipeline for orchestrating training of ML models."""

import os

import hydra
import mlflow
from hydra import utils
from train import define_data, train_model


def hydra_path(path: str) -> str:
    """Set the given path to the root of dir."""
    return os.path.join(hydra.utils.get_original_cwd(), path)


@hydra.main(version_base=None, config_path="../config", config_name="default_config.yaml")
def run_experiment(config: dict):
    """Run the training experiment with MLflow tracking.

    Args:
    ----
        config (DictConfig): Configuration dictionary containing experiment parameters.

    """
    mlflow.set_tracking_uri("file://" + utils.get_original_cwd() + "/mlruns")
    mlflow.set_experiment(config.experiment["mlflow"]["experiment_name"])

    with mlflow.start_run():
        mlflow.autolog()

        hparams = config.experiment["hyperparameters"]
        architechture = config.experiment["architechture"]
        paths = config.experiment["paths"]

        for sub_dict in config.experiment.items():
            mlflow.log_params(sub_dict[1])

        # Load data and split data
        train_data, test_data = define_data(hydra_path(paths["processed_dir"]), hparams)

        # train model
        train_model(train_data, test_data, architechture, hparams, hydra_path(paths["models_dir"]))

        # get model metrics

        # store model


if __name__ == "__main__":
    run_experiment()
