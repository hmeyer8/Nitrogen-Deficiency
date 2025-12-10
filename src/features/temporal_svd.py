# src/features/temporal_svd.py
#
# Temporal SVD utilities for 5-step phenology sequences.
# - Standardizes each time step
# - Chooses rank k to hit variance threshold
# - Returns PC scores, low-rank reconstructions, residuals, and residual statistics

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class TemporalSVDFit:
    mean: np.ndarray
    std: np.ndarray
    Vt: np.ndarray
    k: int
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray


def _standardize_columns(X: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    std_safe = np.where(std < 1e-8, 1.0, std)
    X_std = (X - mean) / std_safe
    return X_std, mean, std_safe


def _choose_rank(singular_values: np.ndarray, var_threshold: float, max_rank: int) -> int:
    total_var = (singular_values ** 2).sum()
    if total_var <= 0:
        return 1
    cumulative = np.cumsum(singular_values ** 2) / total_var
    k = int(np.searchsorted(cumulative, var_threshold) + 1)
    return min(max_rank, max(1, k))


def fit_temporal_svd(X: np.ndarray, var_threshold: float = 0.95, max_rank: int = 5) -> TemporalSVDFit:
    """
    Fit truncated SVD to standardized time-series matrix X (N, T).
    """
    X_std, mean, std = _standardize_columns(X)
    U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
    k = _choose_rank(S, var_threshold, max_rank=min(max_rank, Vt.shape[0]))
    singular_values = S[:k]
    explained_variance_ratio = (singular_values ** 2) / max((S ** 2).sum(), 1e-12)
    return TemporalSVDFit(
        mean=mean,
        std=std,
        Vt=Vt[:k],
        k=k,
        singular_values=singular_values,
        explained_variance_ratio=explained_variance_ratio,
    )


def transform_with_svd(
    X: np.ndarray,
    svd_fit: TemporalSVDFit,
    early_len: int = 2,
    late_len: int = 2,
):
    """
    Project X using a fitted TemporalSVDFit and compute residual-based summaries.
    Returns:
        scores: (N, k) PC scores
        x_hat: (N, T) low-rank reconstruction (standardized space)
        residuals: (N, T)
        residual_norm: (N,)
        feature_vector: (N, k + 4) concatenated features for CatBoost
        stacked_channels: (N, 3T) = [x, x_hat, residual]
    """
    X_std, _, _ = _standardize_columns(X, mean=svd_fit.mean, std=svd_fit.std)
    scores = X_std @ svd_fit.Vt.T  # (N, k)
    x_hat = scores @ svd_fit.Vt    # (N, T)
    residuals = X_std - x_hat
    residual_norm = np.linalg.norm(residuals, axis=1)

    T = X_std.shape[1]
    early_len = min(early_len, T)
    late_len = min(late_len, T)
    residual_mean = residuals.mean(axis=1)
    residual_max_abs = np.max(np.abs(residuals), axis=1)
    early_mean = residuals[:, :early_len].mean(axis=1)
    late_mean = residuals[:, -late_len:].mean(axis=1)

    feature_vector = np.column_stack([
        scores,
        residual_norm,
        residual_mean,
        residual_max_abs,
        early_mean,
        late_mean,
    ])

    stacked_channels = np.concatenate([X_std, x_hat, residuals], axis=1)

    return {
        "scores": scores,
        "x_hat": x_hat,
        "residuals": residuals,
        "residual_norm": residual_norm,
        "features": feature_vector,
        "stacked_channels": stacked_channels,
    }
