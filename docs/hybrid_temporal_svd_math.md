# Temporal SVD Hybrid Nitrogen Risk

This note explains the hybrid phenology model in plain linear-algebra terms, with enough words around the symbols that a non-specialist can follow. It shows how 5-step NDRE time series are decomposed with SVD, how residuals highlight stress, how we turn those signals into supervised and unsupervised detectors, and how we fuse them into one nitrogen-risk score.

## Data: 5-step NDRE time series
- Each field/tile has NDRE at 5 phenology windows (rows are fields, columns are time steps).
- Build a matrix $X \in \mathbb{R}^{N \times 5}$; row $\mathbf{x}_i$ is the 5-step series for field $i$.
- Standardize each column so every time step is on the same scale (subtract the mean, divide by the standard deviation so all columns are apples-to-apples):

$$
X_{\text{std}}[:, j] = \frac{X[:, j] - \mu_j}{\sigma_j}
$$

  where $\mu_j, \sigma_j$ are the mean and std of column $j$. (If a column were constant, we would add a tiny $\epsilon$ to avoid division by zero.)

## 1) Temporal SVD: the low-rank phenology backbone
Think of SVD as finding the main “shapes” of healthy growth across time.
- Compute SVD on standardized data (here $U$ is $N \times 5$, $\Sigma$ is $5 \times 5$ diagonal, $V$ is $5 \times 5$ and its columns are the time patterns):

$$
X_{\text{std}} = U \Sigma V^\top
$$

- Keep the smallest $k \le 5$ singular values that explain at least 95% of the variance (i.e., the leading $k$ “shapes” carry almost all of the signal):

$$
\frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{5} \sigma_i^2} \ge 0.95
$$

- For each field $i$ (take $V_k$ as the first $k$ columns of $V$):
  - Phenology coordinates (PC scores): $\mathbf{s}_i = \mathbf{x}_i V_k \in \mathbb{R}^k$ (how much of each pattern is present)
  - Expected low-rank trajectory: $\hat{\mathbf{x}}_i = \mathbf{s}_i V_k^\top \in \mathbb{R}^5$ (what SVD thinks the curve should look like)
  - Residual (what does not fit the backbone): $\mathbf{r}_i = \mathbf{x}_i - \hat{\mathbf{x}}_i$
  - Residual size (SVD anomaly): $a_i = \|\mathbf{r}_i\|_2$ (single stress scalar: bigger = more off-curve)

Interpretation:
- $V_k$ holds the dominant time patterns (e.g., normal rise and fall of NDRE).
- $s_i$ says where the field sits on those patterns.
- $r_i$ shows how the field deviates from expected growth; large $r_i$ can mean stress.

## 2) Supervised branch: CatBoost on SVD features
Goal: use labels (healthy vs deficient) to learn decision rules on the SVD signals.
- Labels $y_i \in \{0,1\}$: 1 = likely nitrogen deficient (derived from low NDRE minima quantile), 0 = healthy phenology.
- Features per field (all derived from SVD outputs and residual statistics):

$$
\text{feat}_i = \big[\, s_{i,1}, \ldots, s_{i,k},\, a_i,\, \text{mean}(\mathbf{r}_i),\, \|\mathbf{r}_i\|_\infty,\, \text{early-mean}(\mathbf{r}_i),\, \text{late-mean}(\mathbf{r}_i) \,\big]
$$

  - `early_mean`: average residual over early windows (t1–t2)
  - `late_mean`: average residual over late windows (t4–t5)

- Train a CatBoost classifier:

$$
\hat{p}_i = f_{\text{CB}}(\text{feat}_i) \in [0,1]
$$

  where $\hat{p}_i$ is the probability of nitrogen deficiency.

## 3) Unsupervised branch: autoencoder on stacked channels
Goal: detect “unhealthy-looking” trajectories without labels by comparing to healthy patterns.
- Stack three channels for each field:

$$
u_i = [\, \mathbf{x}_i,\, \hat{\mathbf{x}}_i,\, \mathbf{r}_i \,] \in \mathbb{R}^{15}
$$

  (what happened, what SVD expects, and where they disagree).

- Train an autoencoder only on healthy fields ($y_i = 0$):

$$
\begin{aligned}
z_i &= f_\theta(u_i), \\
\tilde{u}_i &= g_\phi(z_i), \\
\min_{\theta,\phi} &\sum_i \|u_i - \tilde{u}_i\|_2^2
\end{aligned}
$$

- AE anomaly score (how “unhealthy” the stacked channels look):

$$
b_i = \|u_i - \tilde{u}_i\|_2^2
$$

Interpretation:
- If a field’s stacked channels cannot be reconstructed well by a model trained on healthy fields, it likely deviates from normal phenology (stress, including nitrogen). (In code we use squared error; any monotone variant works as an anomaly score.)

## 4) Fusion: one nitrogen risk score
Normalize the two unsupervised signals using train min-max (so they live on the same 0–1 scale as probabilities):

$$
\tilde{a}_i = \text{norm}(a_i), \quad \tilde{b}_i = \text{norm}(b_i)
$$

Blend supervised probability with unsupervised stress signals:

$$
\text{Risk}_i = \alpha \hat{p}_i + \beta \tilde{a}_i + \gamma \tilde{b}_i, \quad \alpha,\beta,\gamma \ge 0
$$

Defaults: $\alpha = 0.5,\; \beta = 0.25,\; \gamma = 0.25$.

Decision: mark field $i$ as nitrogen-deficient if $\text{Risk}_i > \tau$, with $\tau$ chosen from precision-recall/F1 on validation data (so precision/recall trade-offs are explicit).

## How this flags nitrogen issues
- Low-rank fit catches the expected growth curve; residuals capture “shape errors” (timing/height mismatches).
- CatBoost learns label-driven rules on both the location on the phenology manifold (scores) and how the field deviates (residual summaries).
- The AE adds a label-free anomaly score that highlights fields whose stacked channels (actual, expected, residual) look unlike healthy examples.
- Fusion keeps the supervised signal in the lead while adding two independent stress checks, improving robustness when labels are weak or noisy.

## Saved artifacts (produced by `train_temporal_hybrid`)
- SVD: `svd_components.npy`, `svd_mean.npy`, `svd_std.npy`, `svd_meta.json`, `svd_scores_{split}.npy`, `svd_residual_norm_{split}.npy`
- CatBoost: `catboost_classifier.cbm`, `catboost_prob_{split}.npy`
- Autoencoder: `temporal_ae.pt`, `ae_anomaly_{split}.npy`
- Fusion: `hybrid_risk_{split}.npy`, `hybrid_threshold.txt`, `hybrid_metrics.json`

## Conclusion
The hybrid phenology model is built to handle both the known signal and the unknown unknowns in crop stress. Temporal SVD gives a compact, interpretable backbone of how a healthy field should evolve, and CatBoost learns sharp, label-driven rules on those scores and residual summaries to spot nitrogen deficits. But field conditions drift: seasons shift, sensing noise creeps in, and labels are imperfect. That’s where the hybrid matters. Residual norms and the healthy-trained autoencoder add a label-free stress detector that lights up when a field’s trajectory moves off the manifold, even if the supervised model stays confident. By blending the supervised probability with these unsupervised checks under calibrated weights, the final risk score keeps CatBoost’s edge when data match training, yet remains resilient when labels are noisy or new patterns appear. In practice this means: high precision/recall on held-out years, plus a built-in guardrail against distribution shift—making the hybrid a safer, more reliable choice for real-world deployment.
