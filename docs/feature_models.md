# PCA vs Autoencoder: What These Features Mean

## PCA / SVD

- Learns **global linear directions** in the original spectral-pixel space.
- Each PCA feature is a **weighted sum of original bands and pixels**.
- Components are:
  - Orthogonal
  - Sorted by variance explained
- Reconstruction is constrained to be **linear**:
  \[
  \hat{x} = \sum_{k=1}^K z_k v_k
  \]

This gives us **physically interpretable** axes: brightness, red-edge strength, moisture, etc.

## Autoencoder

- Learns a nonlinear mapping:
  \[
  z = f_\theta(x), \quad \hat{x} = g_\phi(z)
  \]
- Encoder and decoder are multi-layer neural networks with ReLU nonlinearities.
- No orthogonality constraints, no requirement that features be linear.
- Can represent **curved manifolds** in the spectral space; i.e., it can capture:
  - Interactions between bands (e.g., combinations of red-edge curvature and NIR plateau)
  - Subtle nonlinear patterns that may signal early nitrogen stress.

The autoencoder’s features are therefore:
- Less directly interpretable,
- But potentially more powerful if nitrogen deficiency is expressed through **nonlinear spectral signatures** over space and time.
