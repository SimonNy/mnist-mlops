import os

import click
import matplotlib.pyplot as plt
import torch

from models.model import MyCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1.0e3, help="learning rate to use for training")
@click.option("--batch_size", default=32, help="batch size to use for training")
@click.option("--epochs", default=10, help="number of epochs to train for")
@click.option("--processed_dir", default=os.path.join("data", "processed"), help="Path to processed data directory")
@click.option("--models_dir", default="models", help="Path to models directory")
@click.option("--figures_dir", default=os.path.join("reports", "figures"), help="Path to figures directory")
def train(lr: float, batch_size: int, epochs: int, processed_dir: str, models_dir: str, figures_dir: str) -> None:
    """Train a model."""
    model = MyCNN()
    model.to(DEVICE)

    train_images = torch.load(os.path.join(processed_dir, "train_images.pt"))
    train_target = torch.load(os.path.join(processed_dir, "train_target.pt"))

    train_set = torch.utils.data.TensorDataset(train_images, train_target)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics: dict[str,list] = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
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
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    torch.save(model.state_dict(), os.path.join(models_dir, "model.pth"))
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig(os.path.join(figures_dir, "training_statistics.png"))


@click.command()
@click.argument("model_checkpoint")
@click.option("--processed_dir", default=os.path.join("data", "processed"), help="Path to processed data directory")
def evaluate(model_checkpoint, processed_dir) -> None:
    """Evaluate a trained model."""
    print("Evaluating model")
    print(model_checkpoint)

    model = MyCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    test_images = torch.load(os.path.join(processed_dir, "test_images.pt"))
    test_target = torch.load(os.path.join(processed_dir, "test_target.pt"))

    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE, target.to(DEVICE))
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


cli.add_command(train)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
