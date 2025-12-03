# src/features/pca_svd.py
import numpy as np
from sklearn.decomposition import PCA

class PCAFeatureExtractor:
    """
    Linear feature extractor using PCA.
    - Learns orthogonal directions that explain variance in the spectral tiles.
    - Features are linear combinations of original bands and pixels.
    """
    def __init__(self, n_components=64):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, whiten=False)

    def fit(self, X: np.ndarray):
        self.pca.fit(X)
        return self

    def transform(self, X: np.ndarray):
        return self.pca.transform(X)

    def fit_transform(self, X: np.ndarray):
        return self.pca.fit_transform(X)

    @property
    def components_(self):
        # Each row: principal direction in original feature space
        return self.pca.components_

    @property
    def explained_variance_ratio_(self):
        return self.pca.explained_variance_ratio_
