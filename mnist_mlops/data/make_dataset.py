import os
import click
import torch


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std

@click.command()
@click.option("--raw_dir", default=os.path.join("data","raw"), help="Path to raw data directory")
@click.option("--processed_dir", default=os.path.join("data","processed"), help="Path to processed data directory")

def make_data(raw_dir: str, processed_dir: str):
    """Process raw data and save it to processed directory"""
    train_images, train_target = [], []
    for i in range(5):
        train_images.append(torch.load(os.path.join(raw_dir, f"train_images_{i}.pt")))
        train_target.append(torch.load(os.path.join(raw_dir, f"train_target_{i}.pt")))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(os.path.join(raw_dir, "test_images.pt"))
    test_target: torch.Tensor = torch.load(os.path.join(raw_dir,"test_target.pt"))

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    torch.save(train_images, os.path.join(processed_dir, "train_images.pt"))
    torch.save(train_target, os.path.join(processed_dir, "train_target.pt"))
    torch.save(test_images, os.path.join(processed_dir, "test_images.pt"))
    torch.save(test_target, os.path.join(processed_dir, "test_target.pt"))


if __name__ == "__main__":
    make_data()