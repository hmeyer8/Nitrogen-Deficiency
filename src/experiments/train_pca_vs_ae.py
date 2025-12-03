# src/experiments/train_pca_vs_ae.py
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

from src.config import INTERIM_DIR
from src.features.pca_svd import PCAFeatureExtractor
from src.features.autoencoder import train_autoencoder, encode_with_autoencoder

def evaluate_regression(y_true, y_pred, label: str):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    corr, _ = pearsonr(y_true, y_pred)
    print(f"\n[{label}] R^2: {r2:.4f}, RMSE: {rmse:.4f}, Pearson r: {corr:.4f}")
    return dict(r2=r2, rmse=rmse, corr=corr)

def run_experiment():
    X_train = np.load(INTERIM_DIR / "X_train.npy")
    y_train = np.load(INTERIM_DIR / "y_train_ndre_mean.npy")
    X_test = np.load(INTERIM_DIR / "X_test.npy")
    y_test = np.load(INTERIM_DIR / "y_test_ndre_mean.npy")

    input_dim = X_train.shape[1]
    latent_dim = 64

    # ---- PCA branch ----
    pca = PCAFeatureExtractor(n_components=latent_dim)
    Zp_train = pca.fit_transform(X_train)
    Zp_test = pca.transform(X_test)

    reg_pca = Ridge(alpha=1.0)
    reg_pca.fit(Zp_train, y_train)
    y_pred_pca = reg_pca.predict(Zp_test)
    metrics_pca = evaluate_regression(y_test, y_pred_pca, "PCA")

    # ---- Autoencoder branch ----
    ae_model = train_autoencoder(
        X_train, input_dim=input_dim, latent_dim=latent_dim,
        epochs=50, device="cuda"
    )
    Za_train = encode_with_autoencoder(ae_model, X_train, device="cuda")
    Za_test = encode_with_autoencoder(ae_model, X_test, device="cuda")

    reg_ae = Ridge(alpha=1.0)
    reg_ae.fit(Za_train, y_train)
    y_pred_ae = reg_ae.predict(Za_test)
    metrics_ae = evaluate_regression(y_test, y_pred_ae, "Autoencoder")

    # Save for later plotting / report
    np.save(INTERIM_DIR / "y_pred_pca.npy", y_pred_pca)
    np.save(INTERIM_DIR / "y_pred_ae.npy", y_pred_ae)

if __name__ == "__main__":
    run_experiment()
