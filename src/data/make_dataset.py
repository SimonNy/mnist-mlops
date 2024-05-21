"""Processes raw data files and shows examples."""

import os

import click
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import ImageGrid


@click.group()
def cli():
    """Command line interface."""
    pass


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std


@click.option("--raw_dir", default=os.path.join("data", "raw"), help="Path to raw data directory")
def process_mnist(raw_dir: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return processed train and test data."""
    train_images_list, train_target_list = [], []
    for i in range(10):
        train_images_list.append(torch.load(os.path.join(raw_dir, f"train_images_{i}.pt")))
        train_target_list.append(torch.load(os.path.join(raw_dir, f"train_target_{i}.pt")))
    train_images: torch.Tensor = torch.cat(train_images_list)
    train_target: torch.Tensor = torch.cat(train_target_list)

    test_images: torch.Tensor = torch.load(os.path.join(raw_dir, "test_images.pt"))
    test_target: torch.Tensor = torch.load(os.path.join(raw_dir, "test_target.pt"))

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    return train_images, train_target, test_images, test_target


@click.command()
@click.option("--raw_dir", default=os.path.join("data", "raw"), help="Path to raw data directory")
@click.option("--processed_dir", default=os.path.join("data", "processed"), help="Path to processed data directory")
def make_dataset(raw_dir: str, processed_dir: str):
    """Process raw data and save it to processed directory."""
    train_images, train_target, test_images, test_target = process_mnist(raw_dir)

    torch.save(train_images, os.path.join(processed_dir, "train_images.pt"))
    torch.save(train_target, os.path.join(processed_dir, "train_target.pt"))
    torch.save(test_images, os.path.join(processed_dir, "test_images.pt"))
    torch.save(test_target, os.path.join(processed_dir, "test_target.pt"))


def load_dataset(
    processed_dir: str = os.path.join("data", "processed"),
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test dataloaders for corrupt MNIST."""
    train_images = torch.load(os.path.join(processed_dir, "train_images.pt"))
    train_target = torch.load(os.path.join(processed_dir, "train_target.pt"))
    train_dataset = torch.utils.data.TensorDataset(train_images, train_target)

    test_images = torch.load(os.path.join(processed_dir, "test_images.pt"))
    test_target = torch.load(os.path.join(processed_dir, "test_target.pt"))
    test_dataset = torch.utils.data.TensorDataset(test_images, test_target)
    return train_dataset, test_dataset


def show_image_and_target(images: torch.Tensor, target: torch.Tensor) -> None:
    """Plot images and their labels in a grid."""
    row_col = int(len(images) ** 0.5)
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)
    for ax, im, label in zip(grid, images, target):
        ax.imshow(im.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label.item()}")
        ax.axis("off")
    plt.show()


@click.command()
@click.option("--processed_dir", default=os.path.join("data", "processed"), help="Path to processed data directory")
def data_example(processed_dir):
    """Show data examples and provides info of the train and test set."""
    train_set, test_set = load_dataset(processed_dir)
    print(f"Size of training set: {len(train_set)}")
    print(f"Size of test set: {len(test_set)}")
    print(f"Shape of a training point {(train_set[0][0].shape, train_set[0][1].shape)}")
    print(f"Shape of a test point {(test_set[0][0].shape, test_set[0][1].shape)}")
    show_image_and_target(train_set.tensors[0][:25], train_set.tensors[1][:25])


cli.add_command(make_dataset)
cli.add_command(data_example)


if __name__ == "__main__":
    cli()
