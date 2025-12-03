# Nitrogen Deficiency Feature Learning  
### PCA / SVD vs Autoencoders on Stable Nebraska Cropland

This repository implements a **end-to-end machine learning pipeline** to rigorously compare:

- **Linear spectral feature extraction** using **PCA / SVD**
- **Nonlinear representation learning** using **Autoencoders**

for the task of modeling **nitrogen-related spectral trends in crops** using **Sentinel-2 multispectral satellite imagery** and the **USDA Cropland Data Layer (CDL)**.

The central scientific question is:

> *Are nitrogen deficiency patterns in crops fundamentally linear and spectrally interpretable, or nonlinear and spatiotemporal in nature?*

---

## 1. Scientific Objectives

This project is designed to:

- Identify **Nebraska cropland that remains the same crop across multiple consecutive years**
- Pull **Sentinel-2 Level-2A multispectral imagery** from the **:contentReference[oaicite:0]{index=0} Data Space**
- Tile stable cropland regions and compute:
  - **NDVI (Normalized Difference Vegetation Index)**
  - **NDRE (Normalized Difference Red-Edge Index)**
- Train and compare:
  - **PCA / SVD feature encoders**
  - **Neural autoencoder feature encoders**
- Quantitatively evaluate feature quality by predicting:
  - **Held-out future-year NDRE trends**
- Evaluate using:
  - **Coefficient of determination (R²)**
  - **Root mean squared error (RMSE)**
  - **Pearson correlation (r)**

---

## 2. Data Sources

This project integrates two authoritative remote-sensing data products:

- **:contentReference[oaicite:1]{index=1} (CDL)**  
  Annual 30-meter crop classification maps published by the **:contentReference[oaicite:2]{index=2}** and the **:contentReference[oaicite:3]{index=3}**.

- **:contentReference[oaicite:4]{index=4} Level-2A imagery**  
  Surface-reflectance multispectral imagery provided by the **:contentReference[oaicite:5]{index=5}** via the **Copernicus Data Space**.

---

## 3. Core Experimental Design

### 3.1 Temporal Data-Leakage Control

The pipeline enforces a **strict, leakage-free temporal split**:

- **Training Years:** 2019–2023  
- **Held-Out Test Year:** 2024  

Key guarantees:

- **Crop stability detection uses training years only**
- **Tile coordinates are generated only from training stability masks**
- **The test year never influences feature learning or spatial selection**
- **The test year is used only once, at final evaluation time**

This ensures **true forward generalization**, not interpolation.

---

### 3.2 Weak Nitrogen Supervision via NDRE

Because chemically measured tissue nitrogen is not publicly available at scale, this project uses:

- **Mean tile-level NDRE** as a **weak proxy for nitrogen status**

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

### 4.3 Fair Model Comparison

To ensure fairness:

- Both feature types output **equal-dimensional latent vectors**
- Both use the **same downstream regression head**
- Both predict **future-year NDRE exclusively**
- Evaluation metrics are applied identically

---

## 5. Repository Structure

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
│   │   ├── train_pca_vs_ae.py
│   │   └── evaluate_heldout_year.py
│   └── utils/
│       └── io.py
├── data/
│   ├── raw/
│   ├── interim/
│   └── features/
└── docs/
    ├── experiment_design.md
    ├── feature_models.md
    └── data_lineage.md
