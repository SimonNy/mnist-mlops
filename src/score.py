"""Score data with trained model."""

import mlflow.pytorch
import numpy as np
import torch
from mlflow import MlflowClient
from omegaconf import OmegaConf
from PIL import Image

client = MlflowClient()


def load_model(model_name: str):
    """Load the registered model from MLflow.

    Args:
    ----
        model_name (str): Name of the registered model.

    Returns:
    -------
        torch.nn.Module: The loaded PyTorch model.

    """
    versions = client.search_model_versions(f"name='{model_name}'")

    # Find the latest version by comparing the version numbers
    latest_version_info = max(versions, key=lambda v: int(v.version))

    model_uri = f"models:/{model_name}/{latest_version_info.version}"

    model = mlflow.pytorch.load_model(model_uri)

    return model


def preprocess_image(image_path):
    """Preprocess the input image.

    Args:
    ----
        image_path (str): Path to the input image.

    Returns:
    -------
        torch.Tensor: Preprocessed image tensor.

    """
    # Open the image file
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    # Resize the image to 28x28
    image = image.resize((28, 28))
    # Convert the image to a numpy array
    image_np = np.array(image, dtype=np.float32)
    # Normalize the image
    image_np = (image_np / 255.0 - 0.1307) / 0.3081
    # Convert to a tensor and add batch dimension
    image_tensor = torch.tensor(image_np).unsqueeze(0).unsqueeze(0)
    return image_tensor


def score_model(model, image_tensor):
    """Use the trained model to make predictions on the input image.

    Args:
    ----
        model (torch.nn.Module): The loaded PyTorch model.
        image_tensor (torch.Tensor): Preprocessed image tensor.

    Returns:
    -------
        int: Prediction made by the model.

    """
    # Use the model to make prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()


if __name__ == "__main__":
    # Path to the configuration file
    config_path = "config/experiment/experiment1.yaml"

    # Load configuration
    cfg = OmegaConf.load(config_path)

    # Load the registered model
    model = load_model(model_name=cfg.mlflow.model_registry)

    # Path to the custom input image
    image_path = "data/test/drawn_image2.jpg"

    # Preprocess the image
    image_tensor = preprocess_image(image_path)

    # Use the model to make prediction on the input image
    prediction = score_model(model, image_tensor)
    print("Prediction:", prediction)
