import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

from src.config import INTERIM_DIR


def report_metrics(y_true, y_pred, label):
    r2 = r2_score(y_true, y_pred)
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r, _ = pearsonr(y_true, y_pred)

    print(f"\n[{label}]")
    print(f"R^2     = {r2:.4f}")
    print(f"RMSE    = {rmse:.4f}")
    print(f"Pearson r = {r:.4f}")


def main():
    y_test = np.load(INTERIM_DIR / "y_test.npy")
    y_pred_pca = np.load(INTERIM_DIR / "y_pred_pca.npy")
    y_pred_ae = np.load(INTERIM_DIR / "y_pred_ae.npy")

    report_metrics(y_test, y_pred_pca, "PCA / SVD")
    report_metrics(y_test, y_pred_ae, "Autoencoder")


if __name__ == "__main__":
    main()
