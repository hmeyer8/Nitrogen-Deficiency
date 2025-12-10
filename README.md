# Hybrid Nitrogen Risk: Temporal SVD + CatBoost + Autoencoder

End-to-end phenology modeling on Nebraska corn fields with temporal SVD features, CatBoost classification, AE anomaly detection, and fusion.

---

## Current Status
- Stable-corn mask and Sentinel-2 NDRE/NDVI cubes built for 2019–2024; NDRE-based deficit proxy stored in `data/interim/deficit_threshold.txt`.
- Canonical split locked: train 2019–2020, validation 2021–2022, test 2023–2024 (see `docs/eval_protocol.md`); test remains untouched during tuning.
- Active path: hybrid pipeline (temporal SVD backbone + CatBoost classifier + AE anomaly + fusion). Legacy PCA/AE/CB regression scripts were removed.
- Visualization/debug: `notebooks/03_feature_visualization.ipynb` for SVD loadings, CatBoost importance, and AE reconstruction error checks.

---

## Data Sources & Protocol
- **USDA Cropland Data Layer (CDL):** annual 30 m crop classification (NE) to build the stable-corn mask.
- **Sentinel-2 L2A NDRE/NDVI:** Copernicus Data Space (default) or Planetary Computer COGs.
- Labeling: NDRE minima quantile proxy (`NDRE_DEFICIT_Q`, default 50) recorded in `data/interim/deficit_threshold.txt`.
- Evaluation: follow `docs/eval_protocol.md` for fixed splits, environment defaults, and reporting guardrails.

---

## Setup & Data (Ignition Protocol)
See `docs/ignition_protocol.md` for full environment setup, Copernicus credentials, raw data download, dataset build, and training. Quick start after creating/activating `.venv` and `.env`:

```
python -m pip install -r requirements.txt
python -m src.experiments.prepare_dataset
python -m src.experiments.train_temporal_hybrid
python -m src.experiments.time_series_diagnostics  # optional diagnostics
```

Env toggles:
- `S2_SOURCE=cdse` (default) or `S2_SOURCE=pc` for Planetary Computer.
- Fusion weights: `RISK_ALPHA`, `RISK_BETA`, `RISK_GAMMA`.

---

## Hybrid Phenology Math (GitHub-safe sketch)
We work on standardized 5-step NDRE matrix $X \in \mathbb{R}^{N \times 5}$. Core transforms:
$$
\begin{aligned}
X &= U \Sigma V^{\top},\\
s_i &= x_i V_k,\\
\hat{x}_i &= s_i V_k^{\top},\\
r_i &= x_i - \hat{x}_i,\\
a_i &= \lVert r_i \rVert_2
\end{aligned}
$$

Fusion of supervised probability $\hat{p}_i$ with normalized anomalies $\tilde{a}_i, \tilde{b}_i$:
$$
\text{Risk}_i = \alpha \hat{p}_i + \beta \tilde{a}_i + \gamma \tilde{b}_i
$$

Full derivation and interpretations: `docs/hybrid_temporal_svd_math.md`.

---

## Key Artifacts & CLI
Artifacts in `data/interim` after the pipeline:
- SVD: `svd_components.npy`, `svd_mean.npy`, `svd_std.npy`, `svd_meta.json`, `svd_scores_{split}.npy`, `svd_residual_norm_{split}.npy`
- CatBoost: `catboost_classifier.cbm`, `catboost_prob_{split}.npy`
- Autoencoder: `temporal_ae.pt`, `ae_anomaly_{split}.npy`
- Fusion: `hybrid_risk_{split}.npy`, `hybrid_threshold.txt`, `hybrid_metrics.json`
- Time series + labels: `ndre_ts_*`, `ndvi_ts_*`, `y_min_*`, `y_*_deficit_label.npy`, `deficit_threshold.txt`

Supported CLI workflows:
- Core: `python -m src.experiments.prepare_dataset` -> `python -m src.experiments.train_temporal_hybrid`
- Diagnostics: `python -m src.experiments.time_series_diagnostics`
- Early forecast (optional): `python -m src.experiments.train_temporal_forecaster`
- Data source toggle example: `python -m src.datasources.sentinel_download_pc --year 2023 --verbose`

---

## Nitrogen-Focused Evaluation Tips
- Focus on **late vegetative/pre-tassel NDRE** windows where red-edge tracks nitrogen status.
- Stratify by moisture/drought to reduce confounding; compare relative risk within nearby fields.
- Inspect **SVD loadings** (components) and **CatBoost feature importance** to confirm reliance on red-edge/NIR.
- Review **risk components**: high residual norm or AE anomaly with low $\hat{p}$ may suggest unlabeled stress.
- Use **time-series diagnostics** (`ndre_time_series_mean.png`, `ndre_derivative_outliers.json`) to spot data glitches or abrupt drops.

---

## Repository Structure

```text
Nitrogen-Deficiency/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py
│   ├── datasources/
│   │   └── sentinel_download_pc.py
│   ├── geo/
│   ├── features/
│   │   ├── indices.py
│   │   ├── autoencoder.py
│   │   ├── catboost_model.py
│   │   └── temporal_svd.py
│   └── experiments/
│       ├── prepare_dataset.py
│       ├── train_temporal_hybrid.py
│       ├── train_temporal_forecaster.py
│       └── time_series_diagnostics.py
├── data/
│   ├── raw/
│   └── interim/
└── docs/
    ├── ignition_protocol.md
    └── hybrid_temporal_svd_math.md
```
