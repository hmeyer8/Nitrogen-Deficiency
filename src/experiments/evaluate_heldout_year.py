import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

from src.config import INTERIM_DIR, TARGET_MODE


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
    target_mode = TARGET_MODE  # ndre | deficit_score
    target_files = {
        "ndre": "y_test.npy",
        "deficit_score": "y_test_deficit_score.npy",
    }
    if target_mode not in target_files:
        raise ValueError(f"Unknown TARGET_MODE={target_mode}; choose ndre or deficit_score")

    y_test = np.load(INTERIM_DIR / target_files[target_mode])
    y_pred_pca = np.load(INTERIM_DIR / "y_pred_pca.npy")
    y_pred_ae = np.load(INTERIM_DIR / "y_pred_ae.npy")
    y_pred_cb = np.load(INTERIM_DIR / "y_pred_catboost.npy")

    report_metrics(y_test, y_pred_pca, f"PCA / SVD ({target_mode})")
    report_metrics(y_test, y_pred_ae, f"Autoencoder ({target_mode})")
    report_metrics(y_test, y_pred_cb, f"CatBoost Gradient Booster ({target_mode})")


if __name__ == "__main__":
    main()
