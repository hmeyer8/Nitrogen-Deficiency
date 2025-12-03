# src/features/autoencoder.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
    def __init__(self, input_dim: int, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

def train_autoencoder(X_train, input_dim, latent_dim=64,
                      batch_size=256, epochs=50, lr=1e-3, device="cuda"):
    model = DenseAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
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

def encode_with_autoencoder(model: DenseAutoencoder, X: np.ndarray, device="cuda"):
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
