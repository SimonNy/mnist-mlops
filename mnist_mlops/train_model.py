"""Scripts for training a given model on the provided dataset."""
import logging
import os

import click
import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

from data.make_dataset import load_dataset
from models.model import MyCNN

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@click.command()
@hydra.main(version_base=None, config_path="../config", config_name="default_config.yaml")
def train(config) -> None:
    """Train a model."""
    log.info(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.experiment['hyperparameters']
    architechture = config.experiment['architechture']
    paths = config.experiment['paths']
    
    torch.manual_seed(hparams["seed"])

    model = MyCNN(
        input_dim = architechture["input_dim"],
        first_dim = architechture["first_dim"],
        second_dim = architechture["second_dim"],
        third_dim = architechture["third_dim"],
        output_dim = architechture["output_dim"],
        dropout = architechture["dropout"]
    )
    model.to(DEVICE)

    train_set, _ = load_dataset(paths['processed_dir'])

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=hparams["batch_size"])

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])

    statistics: dict[str, list] = {"train_loss": [], "train_accuracy": []}
    
    log.info("Start training model")
    model.train()
    for epoch in range(hparams["epochs"]):
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                log.info(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
        log.info(f"Epoch {epoch} complete")
 

    log.info("Training complete")
    torch.save(model.state_dict(), os.path.join(paths["models_dir"], "model.pth"))
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig(os.path.join(paths["figures_dir"], "training_statistics.png"))


@click.command()
@click.argument("model_checkpoint")
@click.option("--processed_dir", default=os.path.join("data", "processed"), help="Path to processed data directory")
def evaluate(model_checkpoint, processed_dir) -> None:
    """Evaluate a trained model."""
    print("Evaluating model")
    print(model_checkpoint)

    model = MyCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    _, test_set = load_dataset(processed_dir)

    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE, target.to(DEVICE))
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    train()
