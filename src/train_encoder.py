"""User defined training script for the given model."""

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from data.make_dataset import load_dataset
# from models.encoder_model import Encoder, Decoder, Model
from models.encoder_model import VAE

# from ignite.handlers import ModelCheckpoint, EarlyStopping
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def define_data(data_path, hparams):
    """Define the dataloaders."""
    train_set, test_set = load_dataset(data_path)

    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=hparams["batch_size"], num_workers=4, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=hparams["batch_size"], num_workers=4, shuffle=True
    )

    return train_dataloader, test_dataloader


def train_model(
    train_data: torch.utils.data.DataLoader,
    test_data: torch.utils.data.DataLoader,
    model_params: dict,
    hparams: dict,
    model_path: str,
) -> None:
    """Train the CNN model using the specified configuration.

    Args:
    ----
        train_data (torch.DataLoader): Training data as a torch.DataLoader.
        test_data (torch.DataLoader): Test data as a torch.DataLoader.
        model_params (dict): Dictionary containing configuration for the model.
        hparams (dict): Dictionary containing hyperparameters for the training.
        model_path (str): Path to store the final model.

    """
    print(model_params)
    model = VAE(**model_params)
    model.to(DEVICE)
    early_stopping_callback = EarlyStopping(monitor="train_loss", patience=3, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_path, filename="best-checkpoint", save_top_k=1, verbose=True, monitor="train_loss", mode="min"
    )
    trainer = Trainer(
        max_epochs=hparams["epochs"],
        limit_train_batches=0.2,
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    trainer.fit(model, train_data, test_data)
    return model
