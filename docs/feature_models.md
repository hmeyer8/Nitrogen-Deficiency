# Feature Models: Math, Assumptions, and Suitability

## PCA / SVD (Linear Subspaces)

Data matrix \(X \in \mathbb{R}^{N \times D}\) (flattened tiles) is decomposed as:
\[
X = U \Sigma V^\top,\quad Z_{\text{PCA}} = X V_k
\]
where \(V_k \in \mathbb{R}^{D \times k}\) holds the top-\(k\) right singular vectors.

Properties:
- Linear, orthogonal axes; globally optimal for squared reconstruction error.
- Reconstruction: \(\hat{x} = z V_k^\top\).
- Inductive bias: favors global variance directions (brightness, red-edge slope, moisture).
- Regularization: truncation \(k \ll D\) + standardized inputs.

When it helps:
- Strong linear structure; low noise; need interpretability and fast training.
- Limited data volume where high-capacity models would overfit.

Limits:
- Cannot model nonlinear band interactions or texture–spectral coupling.

## Autoencoder (Nonlinear Manifolds)

Encoder/decoder \(f_\theta, g_\phi\):
\[
z = f_\theta(x),\quad \hat{x} = g_\phi(z),\quad
\min_{\theta,\phi} \|x - g_\phi(f_\theta(x))\|_2^2
\]
Implementation: fully connected encoder/decoder with ReLU, bottleneck \(k=64\).

Properties:
- Learns curved manifolds; no orthogonality constraint.
- Captures higher-order band interactions and local textures.
- Regularization via architecture depth, bottleneck size, and early stopping/epochs.

When it helps:
- Sufficient samples to learn nonlinearities; access to GPU; tolerance for less interpretability.

Limits:
- Can overfit with small datasets; harder to attribute features directly to physical bands.

## CatBoost Gradient Boosting (Tree Ensembles)

Objective (squared loss) optimized by boosting oblivious trees:
\[
F_{t}(x) = F_{t-1}(x) + \eta \cdot h_t(x)
\]
where each tree \(h_t\) fits residuals of \(y - F_{t-1}(x)\).

Properties:
- Nonlinear, piecewise-constant interactions across pixels/bands.
- Built-in regularization: learning rate, depth, ordered boosting, early stopping.
- Operates directly on standardized raw pixels (no learned latent needed).

When it helps:
- Tabular, high-dimensional features with moderate sample size.
- Need strong performance without heavy feature learning.

Limits:
- Feature importance is less spatially interpretable; very high-dimensional inputs can slow training.

## Temporal GRU Forecaster (Sequence Model)

We fit a GRU-based regressor to forecast end-of-season NDRE (or a deficiency z-score) from early-season NDRE trajectories:

- Inputs: \(X \in \mathbb{R}^{N \times T \times 1}\) with \(T=5\) phenology-aligned windows (e.g., V4–V6, pre-tassel, tassel, early grain fill).
- Model: single-layer GRU (hidden size 64); take the last hidden state and a linear head to predict \(\hat{y}\).
- Target: \(y_{\text{future}}\) = mean NDRE in the last window (or its z-score for deficiency).
- Loss: MSE; inputs/targets are standardized for stable training.

Why it helps:
- Uses trajectory shape (growth, plateaus, drops) to predict late-season status earlier.
- Can flag emerging N issues before the final window.

Limits:
- Needs consistent phenology windows and clean time series; sensitive to missing/NaN spikes.
- Less interpretable; pair with diagnostics to sanity check trajectories.

## Time-Series Diagnostics (Derivatives and Outliers)

We store NDRE time series per tile (`ndre_ts_train/test.npy`) and provide diagnostics:
- Mean trajectories: `ndre_time_series_mean.png` to confirm seasonal shape.
- First-difference outlier report: `ndre_derivative_outliers.json` flags tiles/windows with extreme negative \(\Delta\) NDRE.

How to spot outliers:
- Compute \(\Delta \text{NDRE}_t = \text{NDRE}_{t+1} - \text{NDRE}_t\).
- Flag large negative \(\Delta\) (z-score based) to inspect potential stress or data issues.

## Model Selection Guidance for This Problem

- **Data regime**: tiles are high-dimensional; sample sizes are modest. This favors **CatBoost** (robust tabular booster) and **PCA** (low-variance linear baseline). Autoencoders benefit if many clear tiles are available and GPU training is feasible.
- **Interpretability**: PCA components are most transparent; CatBoost offers global/importances; autoencoder latents are least interpretable.
- **Generalization risk**: PCA has lowest variance; CatBoost balances bias/variance; autoencoder has highest variance without careful tuning.
- **Recommended default**: run all three and select via held-out 2024 metrics (`model_metrics.json`). If training data is small or noisy, start with CatBoost + PCA; if abundant clean data and GPU are available, include the autoencoder to capture nonlinearities.
