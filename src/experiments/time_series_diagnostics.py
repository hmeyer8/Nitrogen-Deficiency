# Quick diagnostics for time-series NDRE stacks:
# - Plot mean NDRE trajectory (train/test)
# - Compute first-difference (derivative) and flag outlier drops
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

from src.config import INTERIM_DIR, TARGET_CROP, get_season_windows_for_crop


def load_time_series():
    ndre_ts_train = np.load(INTERIM_DIR / "ndre_ts_train.npy")
    ndre_ts_test = np.load(INTERIM_DIR / "ndre_ts_test.npy")
    return ndre_ts_train, ndre_ts_test


def get_window_labels(year: int):
    labels = []
    for (m1, d1, m2, d2) in get_season_windows_for_crop(TARGET_CROP):
        labels.append(f"{year}-{m1:02d}/{d1:02d}–{m2:02d}/{d2:02d}")
    return labels


def plot_mean_curves(train_ts, test_ts, out_path: Path, year: int):
    labels = get_window_labels(year)
    plt.figure(figsize=(6, 4))
    plt.plot(train_ts.mean(axis=0), marker="o", label="Train mean")
    plt.plot(test_ts.mean(axis=0), marker="o", label="Test mean")
    plt.xticks(ticks=range(len(labels)), labels=[f"T{i}" for i in range(len(labels))])
    plt.xlabel("Time window")
    plt.ylabel("Mean NDRE")
    plt.title(f"NDRE Trajectory ({TARGET_CROP}, {year})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def compute_derivative_outliers(train_ts, k: int = 20):
    # First difference along time axis
    deriv = np.diff(train_ts, axis=1)  # shape (N, T-1)
    # Use z-score to find steep negative drops
    mean = deriv.mean()
    std = deriv.std() + 1e-8
    z = (deriv - mean) / std
    # Flatten to find global extremes
    flat_indices = np.argpartition(z.ravel(), k)[:k]
    outliers = []
    T_minus_1 = deriv.shape[1]
    for idx in flat_indices:
        row = idx // T_minus_1
        t = idx % T_minus_1
        outliers.append(
            {
                "tile_index": int(row),
                "time_step": int(t),
                "derivative": float(deriv[row, t]),
                "z_score": float(z[row, t]),
            }
        )
    return outliers


def main():
    ndre_ts_train, ndre_ts_test = load_time_series()
    year_for_labels = 2024  # labels are generic; year is for title context
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    # Plot mean trajectories
    plot_mean_curves(
        ndre_ts_train,
        ndre_ts_test,
        INTERIM_DIR / "ndre_time_series_mean.png",
        year_for_labels,
    )

    # Derivative-based outlier detection
    outliers = compute_derivative_outliers(ndre_ts_train, k=20)
    with open(INTERIM_DIR / "ndre_derivative_outliers.json", "w") as f:
        json.dump({"target_crop": TARGET_CROP, "outliers": outliers}, f, indent=2)

    print("Saved NDRE mean trajectory plot and derivative outlier report to:", INTERIM_DIR)


if __name__ == "__main__":
    main()
