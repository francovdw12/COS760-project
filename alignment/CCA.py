import numpy as np
from sklearn.cross_decomposition import CCA as SklearnCCA

class CCAAligner:
    """
    CCA alignment: finds two linear projections
    W_src and W_tgt that maximize cross-correlation.
    """
    def __init__(self, n_components: int = 100):
        self.n_components = n_components
        self.cca = SklearnCCA(n_components=n_components, max_iter=1000)
        self.W_src = None
        self.W_tgt = None

    def fit(self, X_src: np.ndarray, X_tgt: np.ndarray):
        """
        X_src, X_tgt: anchor matrices (n_pairs, dim)
        """
        self.cca.fit(X_src, X_tgt)
        # Retrieve loadings to project the full vocabulary
        self.W_src = self.cca.x_rotations_   # (dim, n_components)
        self.W_tgt = self.cca.y_rotations_   # (dim, n_components)
        return self

    def transform_src(self, matrix: np.ndarray) -> np.ndarray:
        """Project the full source vocabulary into the CCA space."""
        projected = matrix @ self.W_src
        # Re-normalize with L2
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        return projected / np.maximum(norms, 1e-8)

    def transform_tgt(self, matrix: np.ndarray) -> np.ndarray:
        projected = matrix @ self.W_tgt
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        return projected / np.maximum(norms, 1e-8)