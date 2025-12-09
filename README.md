# Nitrogen Deficiency Feature Learning  
### PCA / SVD vs Autoencoders on Stable Nebraska Cropland

This repository implements a **end-to-end machine learning pipeline** to rigorously compare:

- **Linear spectral feature extraction** using **PCA / SVD**
- **Nonlinear representation learning** using **Autoencoders**
- **Tree-based gradient boosting** using **CatBoost** directly on standardized pixels

for the task of modeling **nitrogen-related spectral trends in crops** using **Sentinel-2 multispectral satellite imagery** and the **USDA Cropland Data Layer (CDL)**.

The central scientific question is:

> *Are nitrogen deficiency patterns in crops fundamentally linear and spectrally interpretable, or nonlinear and spatiotemporal in nature?*

---

## 1. Scientific Objectives

This project is designed to:

- Identify **Nebraska cropland that remains the same crop across multiple consecutive years**
- Pull **Sentinel-2 Level-2A multispectral imagery** from the **Copernicus Data Space**
- Tile stable cropland regions and compute:
  - **NDVI (Normalized Difference Vegetation Index)**
  - **NDRE (Normalized Difference Red-Edge Index)**
- Train and compare:
  - **PCA / SVD feature encoders**
  - **Neural autoencoder feature encoders**
  - **CatBoost gradient boosting baseline** on the raw (standardized) tiles
- Quantitatively evaluate feature quality by predicting:
  - **Held-out future-year NDRE trends**
- Evaluate using:
  - **Coefficient of determination (R²)**
  - **Root mean squared error (RMSE)**
  - **Pearson correlation (r)**

---

## 2. Data Sources

This project integrates two authoritative remote-sensing data products:

- **USDA Cropland Data Layer (CDL)**  
  Annual 30-meter crop classification maps published by USDA NASS.

- **Sentinel-2 Level-2A imagery**  
  Surface-reflectance multispectral imagery provided via the **Copernicus Data Space**.

---

## 3. Core Experimental Design

### 3.1 Temporal Data-Leakage Control

The pipeline enforces a **strict, leakage-free temporal split**:

- **Training Years:** 2019–2023  
- **Held-Out Test Year:** 2024  

Key guarantees:

- **Tile coordinates are generated from CDL-based stability masks** (intersection of corn across 2019–2024).
- **The test year is used only for CDL masking and final evaluation**; NDRE values from 2024 are never used during training or feature learning.
- **No NDRE leakage into tiling**: only class labels guide spatial selection.

---

### 3.2 Corn-Only Sampling Across 2019–2024

- We intersect CDL corn masks across **all years 2019–2024** before tiling; only pixels that are corn in every year are sampled.
- This intersection is applied in both the CDL exploration notebook and `src/experiments/prepare_dataset.py`, ensuring training and test tiles are drawn from stable corn fields.
- The intersection uses only CDL class labels (not NDRE values) to avoid label leakage into the regression target.

---

### 3.3 Weak Nitrogen Supervision via NDRE

Because chemically measured tissue nitrogen is not publicly available at scale, this project uses:

- **Seasonal NDRE** (min across intra-season snapshots) as a **weak proxy for nitrogen status**
- Optional **deficiency proxy**: NDRE z-score relative to the training mean (lower = more deficient) and a bottom-quartile flag saved alongside NDRE.

This allows **feature-quality comparison without requiring direct nitrogen labels**, while preserving physiological interpretability through the red-edge spectral response.

---

## 4. Mathematical Description of the Feature Models

This project compares **two fundamentally different feature-generation mechanisms**:

---

### 4.1 PCA / SVD Feature Extraction (Linear Subspace Models)

Let  
\[
X \in \mathbb{R}^{N \times D}
\]
represent vectorized Sentinel-2 tiles.

PCA is obtained through the **singular value decomposition**:

\[
X = U \Sigma V^\top
\]

Truncated rank-\(k\) latent features are given by:

\[
Z_{\text{PCA}} = X V_k
\]

**Properties:**
- Linear mapping only
- Orthogonal feature axes
- Globally optimal under ℓ₂ reconstruction error
- Features represent:
  - Dominant spectral contrast
  - Chlorophyll absorption gradients
  - Biomass and canopy reflectance structure

PCA answers the question:

> *Can nitrogen stress be expressed as linear spectral energy modes?*

---

### 4.2 Autoencoder Feature Learning (Nonlinear Manifold Models)

An autoencoder learns the nonlinear transformation:

\[
z = f_\theta(x), \quad \hat{x} = g_\phi(z)
\]

where:

- \(f_\theta\) is the **encoder network**
- \(g_\phi\) is the **decoder network**
- \(z \in \mathbb{R}^k\) is the learned latent representation

The objective minimizes:

\[
\min_{\theta,\phi} \|x - g_\phi(f_\theta(x))\|^2
\]

**Properties:**
- Nonlinear spectral-spatial compression
- No orthogonality constraints
- Captures:
  - Band interactions
  - Red-edge curvature
  - Subtle absorption nonlinearities
  - Texture–spectral coupling

Autoencoders answer the question:

> *Is nitrogen stress encoded through nonlinear spectral interactions that linear PCA cannot represent?*

---

### 4.3 CatBoost Gradient Boosting (Tree Ensembles)

CatBoost fits an ensemble of **oblivious decision trees** to minimize squared error on NDRE. In this project:

- Inputs are the **standardized raw pixel vectors** (no latent encoding).
- The model learns **piecewise-constant nonlinear interactions** across bands and spatial pixels.
- Gradient boosting iteratively fits residuals, enabling flexible fits without manual feature engineering.
- CatBoost bakes in regularization (learning-rate decay, depth limits) to reduce overfitting on tabular data.

This baseline asks:

> *Can a high-capacity tree ensemble match or exceed learned latent features without explicit representation learning?*

---

### 4.4 Fair Model Comparison

To ensure fairness:

- Both feature types output **equal-dimensional latent vectors**
- Both use the **same downstream regression head**
- Both predict **future-year NDRE exclusively**
- Evaluation metrics are applied identically

---

## 5. Running the Pipeline (minimal)

1. Install dependencies: `python -m pip install -r requirements.txt`
2. Build datasets: `python -m src.experiments.prepare_dataset` (uses time-series Sentinel stacks downloaded for June/July/August windows)
3. Train models + save artifacts: `python -m src.experiments.train_pca_ae_catboost`
   - Optional: set `TARGET_MODE=deficit_score` to train/evaluate on the NDRE deficiency z-score instead of raw NDRE.
4. Optional temporal forecaster (predict end-of-season NDRE from early windows): `python -m src.experiments.train_temporal_forecaster`
5. Evaluate held-out year: `python -m src.experiments.evaluate_heldout_year`

Key artifacts in `data/interim`:
- `scaler.joblib`, `pca_model.joblib`, `autoencoder.pt`, `catboost_model.cbm`
- Latents: `Zp_train.npy`, `Zp_test.npy`, `Za_train.npy`, `Za_test.npy`
- Targets: `y_train.npy`, `y_test.npy` (NDRE); `y_*_deficit_score.npy`, `y_*_deficit_label.npy`, `deficit_threshold.txt`
- Time series: `ndre_ts_train.npy`, `ndre_ts_test.npy`, `ndvi_ts_train.npy`, `ndvi_ts_test.npy`
- Future targets for forecasting: `y_future_*` (NDRE last window) and deficit variants
- Predictions: `y_pred_pca.npy`, `y_pred_ae.npy`, `y_pred_catboost.npy`
- Temporal predictions/metrics: `y_pred_temporal_test.npy`, `temporal_metrics.json`
- Metrics/table: `model_metrics.json`, `pred_vs_true.png`

Data source toggle:
- `S2_SOURCE=cdse` (default): Copernicus Data Space via sentinelhub (requires processing units).
- `S2_SOURCE=pc`: free Sentinel-2 L2A COGs via Planetary Computer/AWS (no processing units; uses pystac-client/stackstac).
Or call the PC downloader directly: `python -m src.datasources.sentinel_download_pc --year 2019 --verbose`.

See `docs/ignition_protocol.md` for full setup, data download, and validation steps.

---

## 6. Nitrogen-Focused Evaluation (beyond general vigor)

- Use **NDRE in late vegetative to pre-tassel stages** (V6–VT) where red-edge response correlates with nitrogen status; avoid early/late phenology windows dominated by emergence or senescence.
- Filter or stratify by **soil moisture/drought flags** to reduce confounding from water stress (e.g., Sentinel-1 backscatter or precipitation reanalysis).
- Compare model signals against **known N-rate trial plots** when available (e.g., University of Nebraska–Lincoln, Iowa State, Purdue Extension studies on NDRE-based N sufficiency).
- Use **paired-field controls**: high-vigor reference strips vs. candidate deficiency areas to see if models capture relative N gaps rather than generic health.
- Track **per-band/region importance** (CatBoost) and **component loadings** (PCA) to confirm reliance on red-edge/NIR bands rather than purely broadband brightness.

---

## 7. Repository Structure

```text
nd-nitrogen-features/
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py
│   ├── datasources/
│   │   ├── copernicus_client.py
│   │   ├── sentinel_download.py
│   │   └── cdl_loader.py
│   ├── geo/
│   │   ├── aoi_nebraska.py
│   │   └── tiling.py
│   ├── features/
│   │   ├── indices.py
│   │   ├── pca_svd.py
│   │   └── autoencoder.py
│   ├── experiments/
│   │   ├── prepare_dataset.py
│   │   ├── train_pca_ae_catboost.py
│   │   └── evaluate_heldout_year.py
│   └── utils/
│       └── io.py
├── data/
│   ├── raw/
│   ├── interim/
│   └── features/
└── docs/
    ├── feature_models.md
    └── ignition_protocol.md
