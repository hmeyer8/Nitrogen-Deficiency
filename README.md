# Hybrid Nitrogen Risk: Temporal SVD + CatBoost + Autoencoder

End-to-end, time-series phenology modeling on Nebraska corn fields using:
- **Temporal SVD** to learn a low-rank phenology manifold and residual stress signal.
- **CatBoost classifier** on SVD-derived features (PC scores + residual statistics).
- **Autoencoder anomaly** on stacked SVD channels \([x, \hat{x}, r]\).
- **Hybrid risk fusion** combining supervised probability with two unsupervised anomalies.

Core question: can low-rank phenology plus residual-aware anomalies robustly flag nitrogen deficiency when labels are weak?

---

## 1. Scientific Objectives
- Enforce **stable-corn pixels** across 2019–2024 using CDL intersections.
- Extract **Sentinel-2 L2A** stacks over phenology-aligned windows; compute NDVI/NDRE.
- Build **5-step NDRE time series** per tile; derive weak nitrogen labels from NDRE minima.
- Learn phenology coordinates via **temporal SVD**, capture deviations via residuals.
- Combine **supervised CatBoost** and **unsupervised AE** anomalies into a **single risk score**.

---

## 2. Data Sources
- **USDA Cropland Data Layer (CDL)**: annual 30 m crop classification (NE).
- **Sentinel-2 L2A**: Copernicus Data Space or Planetary Computer COGs (11 bands incl. SCL).

---

## 3. Core Experimental Design
### 3.1 Temporal Split
- Train: **2019–2023**
- Held-out test: **2024**
- Tile coordinates fixed from CDL masks; 2024 NDRE not used for training.

### 3.2 Stable Corn Masking
- Intersect CDL corn across all five years before tiling; discard tiles with low stable coverage.

### 3.3 Weak Nitrogen Labels
- Per tile, take **min NDRE over the 5 windows** as a weak nitrogen proxy.
- Deficiency flag = bottom quantile (default 25%) of train NDRE minima; also store z-scores.

---

## 4. Hybrid Phenology Math (summary)
- Standardize columns of \(X \in \mathbb{R}^{N \times 5}\).
- SVD: \(X = U \Sigma V^\top\); pick smallest \(k \le 5\) with \(\sum_{i=1}^k \sigma_i^2 / \sum_{i=1}^5 \sigma_i^2 \ge 0.95\).
- Per field \(i\):
  - PC scores \(s_i = x_i V_k\); reconstruction \(\hat{x}_i = s_i V_k^\top\); residual \(r_i = x_i - \hat{x}_i\); SVD anomaly \(a_i = \|r_i\|_2\).
- CatBoost features: \([s_i, a_i, \text{mean}(r_i), \max |r_i|, \text{early\_mean}(r_i), \text{late\_mean}(r_i)] \to \hat{p}_i\).
- AE channels: \(u_i = [x_i, \hat{x}_i, r_i] \in \mathbb{R}^{15}\); train AE on healthy; AE anomaly \(b_i = \|u_i - \tilde{u}_i\|_2\).
- Hybrid risk: \(\text{Risk}_i = \alpha \hat{p}_i + \beta \tilde{a}_i + \gamma \tilde{b}_i\) with min–max normalized \(\tilde{a}, \tilde{b}\); default \((\alpha,\beta,\gamma) = (0.5, 0.25, 0.25)\); deficient if \(\text{Risk}_i > \tau\) (PR/F1-chosen).
- Full derivation: `docs/hybrid_temporal_svd_math.md`.

---

## 5. Running the Pipeline (minimal)
1. Install deps: `python -m pip install -r requirements.txt`
2. Build datasets (NDRE/NDVI time series + labels):  
   `python -m src.experiments.prepare_dataset`
3. Train hybrid model (SVD + CatBoost + AE + fusion):  
   `python -m src.experiments.train_temporal_hybrid`  
   - Optional envs: `RISK_ALPHA`, `RISK_BETA`, `RISK_GAMMA` to tune fusion weights.
4. Optional diagnostics: `python -m src.experiments.time_series_diagnostics`  
   Optional early-forecast GRU: `python -m src.experiments.train_temporal_forecaster`

Key artifacts in `data/interim`:
- SVD: `svd_components.npy`, `svd_mean.npy`, `svd_std.npy`, `svd_meta.json`, `svd_scores_{split}.npy`, `svd_residual_norm_{split}.npy`
- CatBoost: `catboost_classifier.cbm`, `catboost_prob_{split}.npy`
- Autoencoder: `temporal_ae.pt`, `ae_anomaly_{split}.npy`
- Fusion: `hybrid_risk_{split}.npy`, `hybrid_threshold.txt`, `hybrid_metrics.json`
- Time series + labels: `ndre_ts_*`, `ndvi_ts_*`, `y_min_*`, `y_*_deficit_label.npy`, `deficit_threshold.txt`

Supported workflows:
- Core: `prepare_dataset` -> `train_temporal_hybrid`
- Optional: `time_series_diagnostics`, `train_temporal_forecaster`
(Legacy PCA/AE/CatBoost regression scripts were removed to avoid confusion; use the hybrid pipeline.)

Data source toggle:
- `S2_SOURCE=cdse` (default): Copernicus Data Space via sentinelhub.
- `S2_SOURCE=pc`: free Sentinel-2 L2A COGs via Planetary Computer/AWS.  
  CLI: `python -m src.datasources.sentinel_download_pc --year 2023 --verbose`

See `docs/ignition_protocol.md` for setup and data download.

---

## 6. Nitrogen-Focused Evaluation Tips
- Focus on **late vegetative/pre-tassel NDRE** windows where red-edge tracks nitrogen status.
- Stratify by moisture/drought to reduce confounding; compare relative risk within nearby fields.
- Inspect **SVD loadings** (components) and **CatBoost feature importance** to confirm reliance on red-edge/NIR.
- Review **risk components**: high residual norm or AE anomaly with low \(\hat{p}\) may suggest unlabeled stress.
- Use **time-series diagnostics** (`ndre_time_series_mean.png`, `ndre_derivative_outliers.json`) to spot data glitches or abrupt drops.

---

## 7. Repository Structure

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
