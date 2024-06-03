import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(pl.LightningModule):
    def __init__(
        self, 
        x_dim: int = 784, 
        hidden_dim: int = 400, 
        latent_dim: int = 20, 
        lr: float = 1e-3,
        ):
        """Initialize the VAE model.

        Parameters
        ----------
        x_dim : int, optional
            Dimensionality of the input data (default is 784 for MNIST).
        hidden_dim : int, optional
            Dimensionality of the hidden layer (default is 400).
        latent_dim : int, optional
            Dimensionality of the latent space (default is 20).

        """
        super(VAE, self).__init__()
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr

        # Encoder
        self.fc1 = nn.Linear(x_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, x_dim)


    def encode(self, x):
        """Encode the input data into latent space.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        tuple
            Mean and log variance of the encoded input.

        """
        h1 = F.relu(self.fc1(x))
        return self.fc2_mu(h1), self.fc2_logvar(h1)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from a normal distribution.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent distribution.
        logvar : torch.Tensor
            Log variance of the latent distribution.

        Returns
        -------
        torch.Tensor
            Sampled latent vector.

        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode the latent vector back into the input space.

        Parameters
        ----------
        z : torch.Tensor
            Latent vector.

        Returns
        -------
        torch.Tensor
            Reconstructed input.

        """
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        """Perform the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        tuple
            Reconstructed input, mean, and log variance.

        """
        mu, logvar = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        """Compute the loss function for VAE.

        Parameters
        ----------
        recon_x : torch.Tensor
            Reconstructed input.
        x : torch.Tensor
            Original input.
        mu : torch.Tensor
            Mean of the latent distribution.
        logvar : torch.Tensor
            Log variance of the latent distribution.

        Returns
        -------
        torch.Tensor
            Loss value.

        """
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.x_dim), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def training_step(self, batch, batch_idx):
        """Perform a single training step.

        Parameters
        ----------
        batch : tuple
            Batch of data.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Training loss.

        """
        x, _ = batch
        recon_x, mu, logvar = self(x)
        loss = self.loss_function(recon_x, x, mu, logvar)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a single validation step.

        Parameters
        ----------
        batch : tuple
            Batch of data.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Validation loss.

        """
        x, _ = batch
        recon_x, mu, logvar = self(x)
        loss = self.loss_function(recon_x, x, mu, logvar)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        """Configure the optimizer for training.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer to be used for training.

        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)

if __name__ == "__main__":
    model = VAE()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")