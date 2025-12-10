# Temporal SVD Hybrid Nitrogen Risk

> Math-rendering fallback: equations are embedded as GitHub-rendered images to avoid Markdown/KaTeX quirks across viewers.

This note explains the hybrid phenology model in plain linear-algebra terms. It shows how 5-step NDRE time series are decomposed with SVD, how residuals highlight stress, how we turn those signals into supervised and unsupervised detectors, and how we fuse them into one nitrogen-risk score.

## Data: 5-step NDRE time series
- Each field/tile has NDRE at 5 phenology windows (rows are fields, columns are time steps).
- Build a matrix $X \in \mathbb{R}^{N \times 5}$; row $x_i$ is the 5-step series for field $i$.
- Standardize each column so every time step is on the same scale:

  ![standardization](https://render.githubusercontent.com/render/math?math=X_%7B%5Ctext%7Bstd%7D%7D%5B%3A%2C%20j%5D%20%3D%20%5Cfrac%7BX%5B%3A%2C%20j%5D%20-%20%5Cmu_j%7D%7B%5Csigma_j%7D)

  where $\mu_j, \sigma_j$ are the mean and std of column $j$.

## 1) Temporal SVD: the low-rank phenology backbone
Think of SVD as finding the main “shapes” of healthy growth across time.
- Compute SVD on standardized data:

  ![svd](https://render.githubusercontent.com/render/math?math=X_%7B%5Ctext%7Bstd%7D%7D%20%3D%20U%20%5CSigma%20V%5E%5Ctop)

- Keep the smallest $k \le 5$ singular values that explain at least 95% of the variance:

  ![evr](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7Bk%7D%20%5Csigma_i%5E2%7D%7B%5Csum_%7Bi%3D1%7D%5E%7B5%7D%20%5Csigma_i%5E2%7D%20%5Cge%200.95)

- For each field $i$:
  - Phenology coordinates (PC scores): ![score](https://render.githubusercontent.com/render/math?math=s_i%20%3D%20x_i%20V_k%20%5Cin%20%5Cmathbb%7BR%7D%5Ek)
  - Expected low-rank trajectory: ![xhat](https://render.githubusercontent.com/render/math?math=%5Chat%7Bx%7D_i%20%3D%20s_i%20V_k%5E%5Ctop%20%5Cin%20%5Cmathbb%7BR%7D%5E5)
  - Residual (what does not fit the backbone): ![resid](https://render.githubusercontent.com/render/math?math=r_i%20%3D%20x_i%20-%20%5Chat%7Bx%7D_i)
  - Residual size (SVD anomaly): ![anom](https://render.githubusercontent.com/render/math?math=a_i%20%3D%20%5C%7Cr_i%5C%7C_2)

Interpretation:
- $V_k$ holds the dominant time patterns (e.g., normal rise and fall of NDRE).
- $s_i$ says where the field sits on those patterns.
- $r_i$ shows how the field deviates from expected growth; large $r_i$ can mean stress.

## 2) Supervised branch: CatBoost on SVD features
Goal: use labels (healthy vs deficient) to learn decision rules on the SVD signals.
- Labels $y_i \in \{0,1\}$: 1 = likely nitrogen deficient (derived from low NDRE minima).
- Features per field:

  ![features](https://render.githubusercontent.com/render/math?math=%5Ctext%7Bfeat%7D_i%20%3D%20%5B%5C%2C%20s_%7Bi%2C1%7D%2C%20%5Cldots%2C%20s_%7Bi%2Ck%7D%2C%5C%2C%20a_i%2C%5C%2C%20%5Ctext%7Bmean%7D%28r_i%29%2C%5C%2C%20%5Cmax%20%7Cr_i%7C%2C%5C%2C%20%5Ctext%7Bearly%5C_mean%7D%28r_i%29%2C%5C%2C%20%5Ctext%7Blate%5C_mean%7D%28r_i%29%20%5C%2C%5D)

  - early_mean: average of the first time steps (t1-t2)
  - late_mean: average of the last time steps (t4-t5)

- Train a CatBoost classifier:

  ![hatp](https://render.githubusercontent.com/render/math?math=%5Chat%7Bp%7D_i%20%3D%20f_%7B%5Ctext%7BCB%7D%7D%28%5Ctext%7Bfeat%7D_i%29%20%5Cin%20%5B0%2C1%5D)

  where $\hat{p}_i$ is the probability of nitrogen deficiency.

## 3) Unsupervised branch: autoencoder on stacked channels
Goal: detect “unhealthy-looking” trajectories without labels by comparing to healthy patterns.
- Stack three channels for each field:

  ![stack](https://render.githubusercontent.com/render/math?math=u_i%20%3D%20%5B%5C%2C%20x_i%2C%5C%2C%20%5Chat%7Bx%7D_i%2C%5C%2C%20r_i%20%5C%2C%5D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7B15%7D)

  (what happened, what SVD expects, and where they disagree).

- Train an autoencoder only on healthy fields ($y_i = 0$):

  ![ae objective](https://render.githubusercontent.com/render/math?math=%5Cbegin%7Baligned%7D%20z_i%20%26%3D%20f_%5Ctheta%28u_i%29%2C%20%5C%5C%20%5Ctilde%7Bu%7D_i%20%26%3D%20g_%5Cphi%28z_i%29%2C%20%5C%5C%20%5Cmin_%7B%5Ctheta%2C%5Cphi%7D%20%26%5Csum_i%20%5C%7Cu_i%20-%20%5Ctilde%7Bu%7D_i%5C%7C_2%5E2%20%5Cend%7Baligned%7D)

- AE anomaly score (how “unhealthy” the stacked channels look):

  ![ae anomaly](https://render.githubusercontent.com/render/math?math=b_i%20%3D%20%5C%7Cu_i%20-%20%5Ctilde%7Bu%7D_i%5C%7C_2)

Interpretation:
- If a field’s stacked channels cannot be reconstructed well by a model trained on healthy fields, it likely deviates from normal phenology (stress, including nitrogen).

## 4) Fusion: one nitrogen risk score
Normalize the two unsupervised signals using train min-max:

![norm](https://render.githubusercontent.com/render/math?math=%5Ctilde%7Ba%7D_i%20%3D%20%5Ctext%7Bnorm%7D%28a_i%29%2C%20%5Cquad%20%5Ctilde%7Bb%7D_i%20%3D%20%5Ctext%7Bnorm%7D%28b_i%29)

Blend supervised probability with unsupervised stress signals:

![risk](https://render.githubusercontent.com/render/math?math=%5Ctext%7BRisk%7D_i%20%3D%20%5Calpha%20%5Chat%7Bp%7D_i%20%2B%20%5Cbeta%20%5Ctilde%7Ba%7D_i%20%2B%20%5Cgamma%20%5Ctilde%7Bb%7D_i%2C%20%5Cquad%20%5Calpha%2C%5Cbeta%2C%5Cgamma%20%5Cge%200)

Defaults: ![defaults](https://render.githubusercontent.com/render/math?math=%5Calpha%20%3D%200.5%2C%5C%3B%20%5Cbeta%20%3D%200.25%2C%5C%3B%20%5Cgamma%20%3D%200.25).

Decision: mark field $i$ as nitrogen-deficient if $\text{Risk}_i > \tau$, with $\tau$ chosen from precision-recall/F1 on validation data.

## How this flags nitrogen issues
- Low-rank fit catches the expected growth curve; residuals capture “shape errors.”
- CatBoost learns label-driven rules on both the location on the phenology manifold (scores) and how the field deviates (residual summaries).
- The AE adds a label-free anomaly score that highlights fields whose stacked channels (actual, expected, residual) look unlike healthy examples.
- Fusion keeps the supervised signal in the lead while adding two independent stress checks, improving robustness when labels are weak or noisy.

## Saved artifacts (produced by `train_temporal_hybrid`)
- SVD: `svd_components.npy`, `svd_mean.npy`, `svd_std.npy`, `svd_meta.json`, `svd_scores_{split}.npy`, `svd_residual_norm_{split}.npy`
- CatBoost: `catboost_classifier.cbm`, `catboost_prob_{split}.npy`
- Autoencoder: `temporal_ae.pt`, `ae_anomaly_{split}.npy`
- Fusion: `hybrid_risk_{split}.npy`, `hybrid_threshold.txt`, `hybrid_metrics.json`
