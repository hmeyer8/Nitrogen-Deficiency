# src/features/autoencoder.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import GPU_ENABLED

class DenseAutoencoder(nn.Module):
    """
    Nonlinear feature extractor.
    Encoder:    X -> z (bottleneck)
    Decoder:    z -> X_hat

    Compared to PCA:
    - No orthogonality constraint
    - Learns nonlinear manifolds
    - Can model higher-order spectral interactions
    """
    def __init__(self, input_dim: int, latent_dim: int = 64, hidden_dims=(512, 256)):
        super().__init__()
        self.hidden_dims = tuple(hidden_dims)
        enc_layers = []
        prev = input_dim
        for h in self.hidden_dims:
            enc_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        prev = latent_dim
        for h in reversed(self.hidden_dims):
            dec_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

def train_autoencoder(
    X_train,
    input_dim,
    latent_dim=64,
    hidden_dims=(512, 256),
    batch_size=256,
    epochs=50,
    lr=1e-3,
    device=None,
):
    if device is None:
        use_cuda = GPU_ENABLED and torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
    model = DenseAutoencoder(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    ds = TensorDataset(torch.from_numpy(X_train).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            x_hat, _ = model(batch)
            loss = criterion(x_hat, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)

        epoch_loss = total_loss / len(ds)
        print(f"[AE] Epoch {epoch+1}/{epochs} - Recon Loss: {epoch_loss:.6f}")

    return model

def encode_with_autoencoder(model: DenseAutoencoder, X: np.ndarray, device=None):
    if device is None:
        use_cuda = GPU_ENABLED and torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
    model.eval()
    ds = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    zs = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            _, z = model(batch)
            zs.append(z.cpu().numpy())
    return np.concatenate(zs, axis=0)


def reconstruct_with_autoencoder(model: DenseAutoencoder, X: np.ndarray, device=None):
    if device is None:
        use_cuda = GPU_ENABLED and torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
    model.eval()
    ds = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    xs = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            x_hat, _ = model(batch)
            xs.append(x_hat.cpu().numpy())
    return np.concatenate(xs, axis=0)


def save_autoencoder(model: DenseAutoencoder, path):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": model.encoder[0].in_features,
            "latent_dim": model.decoder[0].in_features,
            "hidden_dims": getattr(model, "hidden_dims", (512, 256)),
        },
        path,
    )


def load_autoencoder(path, device=None) -> DenseAutoencoder:
    checkpoint = torch.load(path, map_location=device or "cpu")
    hidden_dims = checkpoint.get("hidden_dims", (512, 256))
    model = DenseAutoencoder(
        input_dim=checkpoint["input_dim"],
        latent_dim=checkpoint["latent_dim"],
        hidden_dims=hidden_dims,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device or "cpu")
    model.eval()
    return model


class AutoencoderFeatureExtractor:
    """
    Convenience wrapper for using a trained DenseAutoencoder as a feature encoder/decoder.
    """

    def __init__(self, model: DenseAutoencoder, device=None):
        if device is None:
            use_cuda = GPU_ENABLED and torch.cuda.is_available()
            device = "cuda" if use_cuda else "cpu"
        self.model = model.to(device)
        self.device = device

    def transform(self, X: np.ndarray):
        return encode_with_autoencoder(self.model, X, device=self.device)

    def reconstruct(self, X: np.ndarray):
        return reconstruct_with_autoencoder(self.model, X, device=self.device)
