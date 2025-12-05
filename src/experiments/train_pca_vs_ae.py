# src/experiments/train_pca_vs_ae.py

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import torch

from src.features.autoencoder import train_autoencoder, encode_with_autoencoder
from src.features.pca_svd import PCAFeatureExtractor
from src.config import INTERIM_DIR, GPU_ENABLED

def evaluate_regression(y_true, y_pred, label: str):
    r2 = r2_score(y_true, y_pred)
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
    corr, _ = pearsonr(y_true, y_pred)
    print(f"\n[{label}] R^2: {r2:.4f}, RMSE: {rmse:.4f}, Pearson r: {corr:.4f}")
    return dict(r2=r2, rmse=rmse, corr=corr)

def plot_predictions(y_true, preds: dict, out_path: Path):
    plt.figure(figsize=(6, 6))
    min_y = min(y_true.min(), *(p.min() for p in preds.values()))
    max_y = max(y_true.max(), *(p.max() for p in preds.values()))
    diag = np.linspace(min_y, max_y, 100)

    for label, y_pred in preds.items():
        plt.scatter(y_true, y_pred, alpha=0.5, s=18, label=label)

    plt.plot(diag, diag, "k--", linewidth=1, label="1:1")
    plt.xlabel("True NDRE")
    plt.ylabel("Predicted NDRE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def run_experiment():
    X_train = np.load(INTERIM_DIR / "X_train.npy")
    y_train = np.load(INTERIM_DIR / "y_train.npy")
    X_test = np.load(INTERIM_DIR / "X_test.npy")
    y_test = np.load(INTERIM_DIR / "y_test.npy")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    input_dim = X_train.shape[1]
    latent_dim = 64
    device = "cuda" if (GPU_ENABLED and torch.cuda.is_available()) else "cpu"

    pca = PCAFeatureExtractor(n_components=latent_dim)
    Zp_train = pca.fit_transform(X_train)
    Zp_test = pca.transform(X_test)

    reg_pca = Ridge(alpha=1.0)
    reg_pca.fit(Zp_train, y_train)
    y_pred_pca = reg_pca.predict(Zp_test)
    metrics_pca = evaluate_regression(y_test, y_pred_pca, "PCA")

    ae_model = train_autoencoder(
        X_train, input_dim=input_dim, latent_dim=latent_dim,
        epochs=50, device=device
    )

    Za_train = encode_with_autoencoder(ae_model, X_train, device=device)
    Za_test = encode_with_autoencoder(ae_model, X_test, device=device)

    reg_ae = Ridge(alpha=1.0)
    reg_ae.fit(Za_train, y_train)
    y_pred_ae = reg_ae.predict(Za_test)
    metrics_ae = evaluate_regression(y_test, y_pred_ae, "Autoencoder")

    try:
        from catboost import CatBoostRegressor, Pool
    except ImportError as e:
        raise ImportError("catboost is required for this comparison (pip install catboost)") from e

    # CatBoost operates directly on the standardized features
    cb_train = Pool(X_train, y_train)
    cb_test = Pool(X_test, y_test)
    cb_model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=600,
        depth=8,
        learning_rate=0.05,
        eval_metric="RMSE",
        random_seed=0,
        verbose=100,
    )
    cb_model.fit(cb_train, eval_set=cb_test, use_best_model=True)
    y_pred_cb = cb_model.predict(X_test)
    metrics_cb = evaluate_regression(y_test, y_pred_cb, "CatBoost")

    np.save(INTERIM_DIR / "y_pred_pca.npy", y_pred_pca)
    np.save(INTERIM_DIR / "y_pred_ae.npy", y_pred_ae)
    np.save(INTERIM_DIR / "y_pred_catboost.npy", y_pred_cb)

    metrics = {
        "pca": metrics_pca,
        "autoencoder": metrics_ae,
        "catboost": metrics_cb,
    }
    with open(INTERIM_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_predictions(
        y_test,
        {"PCA": y_pred_pca, "Autoencoder": y_pred_ae, "CatBoost": y_pred_cb},
        INTERIM_DIR / "pred_vs_true.png",
    )

    # Console comparison table
    print("\nModel comparison:")
    print(f"{'Model':<12} | {'R^2':>6} | {'RMSE':>8} | {'Pearson r':>10}")
    print("-" * 44)
    for label, m in [
        ("PCA", metrics_pca),
        ("Autoencoder", metrics_ae),
        ("CatBoost", metrics_cb),
    ]:
        print(f"{label:<12} | {m['r2']:6.4f} | {m['rmse']:8.4f} | {m['corr']:10.4f}")

if __name__ == "__main__":
    run_experiment()
