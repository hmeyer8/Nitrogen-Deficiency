# Train a simple temporal forecaster to predict end-of-season NDRE (or deficiency score)
# from early-season NDRE time series.
import os
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

from src.config import INTERIM_DIR, GPU_ENABLED


class GRURegressor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)


def evaluate_regression(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
    corr, _ = pearsonr(y_true, y_pred)
    return dict(r2=r2, rmse=rmse, corr=corr)


def run():
    device = "cuda" if (GPU_ENABLED and torch.cuda.is_available()) else "cpu"
    target_mode = os.getenv("TARGET_MODE", "ndre")  # ndre | deficit_score

    # Inputs: early-season NDRE time series (exclude last window)
    ndre_ts_train = np.load(INTERIM_DIR / "ndre_ts_train.npy")
    ndre_ts_test = np.load(INTERIM_DIR / "ndre_ts_test.npy")
    X_train = ndre_ts_train[:, :-1]  # shape: (N, T-1)
    X_test = ndre_ts_test[:, :-1]

    target_files = {
        "ndre": ("y_future_train.npy", "y_future_test.npy"),
        "deficit_score": ("y_future_train_deficit_score.npy", "y_future_test_deficit_score.npy"),
    }
    if target_mode not in target_files:
        raise ValueError(f"Unknown TARGET_MODE={target_mode}; choose ndre or deficit_score")
    y_train = np.load(INTERIM_DIR / target_files[target_mode][0])
    y_test = np.load(INTERIM_DIR / target_files[target_mode][1])

    # Scale inputs and targets
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_test_scaled = x_scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)

    # Dataloaders
    train_ds = TensorDataset(
        torch.from_numpy(X_train_scaled).float().unsqueeze(-1),
        torch.from_numpy(y_train_scaled).float(),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test_scaled).float().unsqueeze(-1),
        torch.from_numpy(y_test_scaled).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    model = GRURegressor(input_dim=1, hidden_dim=64, num_layers=1).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * xb.size(0)
        epoch_loss = total_loss / len(train_ds)
        print(f"[Temporal GRU] Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}")

    # Evaluate
    model.eval()
    def predict(loader):
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                pred = model(xb).squeeze(-1).cpu().numpy()
                preds.append(pred)
        return np.concatenate(preds, axis=0)

    y_pred_train_scaled = predict(train_loader)
    y_pred_test_scaled = predict(test_loader)
    y_pred_train = y_scaler.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).reshape(-1)
    y_pred_test = y_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).reshape(-1)

    metrics_train = evaluate_regression(y_train, y_pred_train)
    metrics_test = evaluate_regression(y_test, y_pred_test)

    print("\nTemporal forecasting metrics (target:", target_mode, ")")
    for split, m in [("Train", metrics_train), ("Test", metrics_test)]:
        print(f"{split}: R^2={m['r2']:.4f}, RMSE={m['rmse']:.4f}, r={m['corr']:.4f}")

    np.save(INTERIM_DIR / "y_pred_temporal_train.npy", y_pred_train)
    np.save(INTERIM_DIR / "y_pred_temporal_test.npy", y_pred_test)
    with open(INTERIM_DIR / "temporal_metrics.json", "w") as f:
        json.dump(
            {
                "target_mode": target_mode,
                "train": {k: float(v) for k, v in metrics_train.items()},
                "test": {k: float(v) for k, v in metrics_test.items()},
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    run()
