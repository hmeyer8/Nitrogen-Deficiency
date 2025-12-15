# Hybrid nitrogen risk pipeline:
# 1) Fit temporal SVD on 5-step NDRE sequences
# 2) Build CatBoost classifier on PC scores + residual statistics
# 3) Train autoencoder on stacked SVD channels (healthy-only) for unsupervised anomaly
# 4) Fuse CatBoost probability, SVD residual norm, AE anomaly into a single risk score

import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score

from src.config import INTERIM_DIR, GPU_ENABLED
from src.features.temporal_svd import fit_temporal_svd, transform_with_svd
from src.features.catboost_model import train_catboost_classifier, predict_catboost_proba
from src.features.autoencoder import train_autoencoder, save_autoencoder, reconstruct_with_autoencoder


def _safe_metric(fn, y_true, y_score):
    try:
        return float(fn(y_true, y_score))
    except Exception:
        return float("nan")


def _min_max_scale(arr, ref):
    lo = ref.min()
    hi = ref.max()
    denom = (hi - lo) + 1e-8
    return (arr - lo) / denom, float(lo), float(hi)


def _best_f1_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return 0.5, float("nan")
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-8)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])


def run():
    device = "cuda" if (GPU_ENABLED and torch.cuda.is_available()) else "cpu"

    ndre_ts_train = np.load(INTERIM_DIR / "ndre_ts_train.npy")
    ndre_ts_val = np.load(INTERIM_DIR / "ndre_ts_val.npy")
    ndre_ts_test = np.load(INTERIM_DIR / "ndre_ts_test.npy")
    y_train = np.load(INTERIM_DIR / "y_train_deficit_label.npy")
    y_val = np.load(INTERIM_DIR / "y_val_deficit_label.npy")
    y_test = np.load(INTERIM_DIR / "y_test_deficit_label.npy")

    # --- Temporal SVD fit ---
    svd = fit_temporal_svd(
        ndre_ts_train,
        var_threshold=0.95,
        max_rank=min(5, ndre_ts_train.shape[1]),
    )
    proj_train = transform_with_svd(ndre_ts_train, svd, early_len=2, late_len=2)
    proj_val = transform_with_svd(ndre_ts_val, svd, early_len=2, late_len=2)
    proj_test = transform_with_svd(ndre_ts_test, svd, early_len=2, late_len=2)

    np.save(INTERIM_DIR / "svd_mean.npy", svd.mean)
    np.save(INTERIM_DIR / "svd_std.npy", svd.std)
    np.save(INTERIM_DIR / "svd_components.npy", svd.Vt)
    with open(INTERIM_DIR / "svd_meta.json", "w") as f:
        json.dump(
            {
                "rank": svd.k,
                "singular_values": svd.singular_values.tolist(),
                "explained_variance_ratio": svd.explained_variance_ratio.tolist(),
            },
            f,
            indent=2,
        )

    np.save(INTERIM_DIR / "svd_scores_train.npy", proj_train["scores"])
    np.save(INTERIM_DIR / "svd_scores_val.npy", proj_val["scores"])
    np.save(INTERIM_DIR / "svd_scores_test.npy", proj_test["scores"])
    np.save(INTERIM_DIR / "svd_residual_norm_train.npy", proj_train["residual_norm"])
    np.save(INTERIM_DIR / "svd_residual_norm_val.npy", proj_val["residual_norm"])
    np.save(INTERIM_DIR / "svd_residual_norm_test.npy", proj_test["residual_norm"])

    # --- CatBoost on SVD features ---
    cb_features_train = proj_train["features"].astype(np.float32)
    cb_features_val = proj_val["features"].astype(np.float32)
    cb_features_test = proj_test["features"].astype(np.float32)
    use_gpu = GPU_ENABLED and torch.cuda.is_available()
    try:
        cb_model = train_catboost_classifier(
            cb_features_train,
            y_train,
            X_valid=cb_features_val,
            y_valid=y_val,
            iterations=400,
            depth=6,
            learning_rate=0.07,
            random_seed=0,
            use_gpu=use_gpu,
            gpu_ram_part=0.2,
            max_bin=32,
            subsample=0.85,
            rsm=None,
            bootstrap_type="Poisson",
            bagging_temperature=1.0,
            l2_leaf_reg=3.0,
        )
    except Exception as e:
        msg = str(e)
        if "Out of memory" in msg or "NCuda" in msg:
            print("CatBoost GPU OOM; falling back to CPU with lighter settings.")
            cb_model = train_catboost_classifier(
                cb_features_train,
                y_train,
                X_valid=cb_features_val,
                y_valid=y_val,
                iterations=300,
                depth=6,
                learning_rate=0.06,
                random_seed=0,
                use_gpu=False,
                max_bin=32,
                subsample=0.8,
                bootstrap_type="Poisson",
                l2_leaf_reg=4.0,
            )
        else:
            raise

    cb_train_prob = predict_catboost_proba(cb_model, cb_features_train).astype(np.float32)
    cb_val_prob = predict_catboost_proba(cb_model, cb_features_val).astype(np.float32)
    cb_test_prob = predict_catboost_proba(cb_model, cb_features_test).astype(np.float32)
    cb_model.save_model(INTERIM_DIR / "catboost_classifier.cbm")
    np.save(INTERIM_DIR / "catboost_prob_train.npy", cb_train_prob)
    np.save(INTERIM_DIR / "catboost_prob_val.npy", cb_val_prob)
    np.save(INTERIM_DIR / "catboost_prob_test.npy", cb_test_prob)

    # --- Autoencoder on stacked SVD channels (healthy-only) ---
    stack_train = proj_train["stacked_channels"].astype(np.float32)
    stack_val = proj_val["stacked_channels"].astype(np.float32)
    stack_test = proj_test["stacked_channels"].astype(np.float32)
    healthy_mask = y_train == 0
    ae_train_data = stack_train[healthy_mask]
    if ae_train_data.shape[0] < 10:
        print("Warning: too few healthy samples; training AE on full train set.")
        ae_train_data = stack_train

    # Keep AE small for 15-dim inputs
    ae_hidden = (64, 32) if stack_train.shape[1] <= 64 else (128, 64)
    ae_latent = min(16, max(4, stack_train.shape[1] // 2))
    ae_model = train_autoencoder(
        ae_train_data,
        input_dim=stack_train.shape[1],
        latent_dim=ae_latent,
        hidden_dims=ae_hidden,
        batch_size=128,
        epochs=80,
        lr=1e-3,
        device=device,
    )
    save_autoencoder(ae_model, INTERIM_DIR / "temporal_ae.pt")

    recon_train = reconstruct_with_autoencoder(ae_model, stack_train, device=device)
    recon_val = reconstruct_with_autoencoder(ae_model, stack_val, device=device)
    recon_test = reconstruct_with_autoencoder(ae_model, stack_test, device=device)
    ae_train_anomaly = np.sum((stack_train - recon_train) ** 2, axis=1)
    ae_val_anomaly = np.sum((stack_val - recon_val) ** 2, axis=1)
    ae_test_anomaly = np.sum((stack_test - recon_test) ** 2, axis=1)
    np.save(INTERIM_DIR / "ae_anomaly_train.npy", ae_train_anomaly)
    np.save(INTERIM_DIR / "ae_anomaly_val.npy", ae_val_anomaly)
    np.save(INTERIM_DIR / "ae_anomaly_test.npy", ae_test_anomaly)

    # --- Hybrid risk fusion ---
    res_norm_train = proj_train["residual_norm"]
    res_norm_val = proj_val["residual_norm"]
    res_norm_test = proj_test["residual_norm"]
    res_norm_train_norm, res_lo, res_hi = _min_max_scale(res_norm_train, res_norm_train)
    res_norm_val_norm = (res_norm_val - res_lo) / ((res_hi - res_lo) + 1e-8)
    res_norm_test_norm = (res_norm_test - res_lo) / ((res_hi - res_lo) + 1e-8)

    ae_train_norm, ae_lo, ae_hi = _min_max_scale(ae_train_anomaly, ae_train_anomaly)
    ae_val_norm = (ae_val_anomaly - ae_lo) / ((ae_hi - ae_lo) + 1e-8)
    ae_test_norm = (ae_test_anomaly - ae_lo) / ((ae_hi - ae_lo) + 1e-8)

    alpha = float(os.getenv("RISK_ALPHA", "0.5"))
    beta = float(os.getenv("RISK_BETA", "0.25"))
    gamma = float(os.getenv("RISK_GAMMA", "0.25"))
    hybrid_train = alpha * cb_train_prob + beta * res_norm_train_norm + gamma * ae_train_norm
    hybrid_val = alpha * cb_val_prob + beta * res_norm_val_norm + gamma * ae_val_norm
    hybrid_test = alpha * cb_test_prob + beta * res_norm_test_norm + gamma * ae_test_norm

    # Choose threshold on validation (protocol: tune on val, report once on test)
    tau, best_f1_val = _best_f1_threshold(y_val, hybrid_val)
    hybrid_train_pred = (hybrid_train > tau).astype(np.int8)
    hybrid_val_pred = (hybrid_val > tau).astype(np.int8)
    hybrid_test_pred = (hybrid_test > tau).astype(np.int8)

    np.save(INTERIM_DIR / "hybrid_risk_train.npy", hybrid_train)
    np.save(INTERIM_DIR / "hybrid_risk_val.npy", hybrid_val)
    np.save(INTERIM_DIR / "hybrid_risk_test.npy", hybrid_test)
    with open(INTERIM_DIR / "hybrid_threshold.txt", "w") as f:
        f.write(f"alpha={alpha}\nbeta={beta}\ngamma={gamma}\ntau={tau}\n")

    metrics = {
        "svd": {
            "rank": int(svd.k),
            "explained_variance_ratio": [float(x) for x in svd.explained_variance_ratio],
        },
        "catboost": {
            "roc_auc_train": _safe_metric(roc_auc_score, y_train, cb_train_prob),
            "roc_auc_val": _safe_metric(roc_auc_score, y_val, cb_val_prob),
            "roc_auc_test": _safe_metric(roc_auc_score, y_test, cb_test_prob),
            "average_precision_val": _safe_metric(average_precision_score, y_val, cb_val_prob),
            "average_precision_test": _safe_metric(average_precision_score, y_test, cb_test_prob),
            "f1_at_0_5_val": _safe_metric(f1_score, y_val, (cb_val_prob > 0.5).astype(np.int8)),
            "f1_at_0_5_test": _safe_metric(f1_score, y_test, (cb_test_prob > 0.5).astype(np.int8)),
        },
        "autoencoder": {
            "train_recon_mse": float(np.mean((stack_train - recon_train) ** 2)),
            "healthy_train_samples": int(ae_train_data.shape[0]),
        },
        "hybrid": {
            "roc_auc_val": _safe_metric(roc_auc_score, y_val, hybrid_val),
            "roc_auc_test": _safe_metric(roc_auc_score, y_test, hybrid_test),
            "average_precision_val": _safe_metric(average_precision_score, y_val, hybrid_val),
            "average_precision_test": _safe_metric(average_precision_score, y_test, hybrid_test),
            "f1_at_tau_val": _safe_metric(f1_score, y_val, hybrid_val_pred),
            "f1_at_tau_test": _safe_metric(f1_score, y_test, hybrid_test_pred),
            "tau": tau,
            "tau_source": "val",
            "val_f1_opt": best_f1_val,
            "train_f1_at_tau": _safe_metric(f1_score, y_train, hybrid_train_pred),
        },
    }

    with open(INTERIM_DIR / "hybrid_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nHybrid nitrogen risk metrics")
    print("SVD rank:", svd.k, "| EVR:", metrics["svd"]["explained_variance_ratio"])
    print(
        "CatBoost AUC train/val/test:",
        metrics["catboost"]["roc_auc_train"],
        metrics["catboost"]["roc_auc_val"],
        metrics["catboost"]["roc_auc_test"],
    )
    print(
        "Hybrid AUC val/test:",
        metrics["hybrid"]["roc_auc_val"],
        metrics["hybrid"]["roc_auc_test"],
        "tau:", tau,
    )


if __name__ == "__main__":
    run()
