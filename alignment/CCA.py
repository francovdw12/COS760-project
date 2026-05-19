# CCA alignment (RQ1).
# Finds two linear projections that maximise cross-lingual correlation
# between the source-language and English anchor matrices.
import numpy as np
from sklearn.cross_decomposition import CCA as SklearnCCA


def cca_fit(X_src, X_tgt, n_components=100):
    """Fit CCA on aligned anchor matrices.

    X_src, X_tgt: arrays (n_pairs, dim) built from the seed lexicon.
    Returns a dict holding the two rotation matrices.
    """
    cca = SklearnCCA(n_components=n_components, max_iter=1000)
    cca.fit(X_src, X_tgt)
    return {
        "n_components": n_components,
        "W_src": cca.x_rotations_,   # (dim, n_components)
        "W_tgt": cca.y_rotations_,   # (dim, n_components)
    }


def _project(matrix, W):
    projected = matrix @ W
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    return projected / np.maximum(norms, 1e-8)


def cca_transform_src(model, matrix):
    """Project source-language vectors into the shared CCA space."""
    return _project(matrix, model["W_src"])


def cca_transform_tgt(model, matrix):
    """Project English vectors into the shared CCA space."""
    return _project(matrix, model["W_tgt"])


class CCAAligner:
    """Class-based wrapper around cca_fit / _project for use by run_rq1 and run_rq2."""

    def __init__(self, n_components: int = 100):
        self.n_components = n_components
        self.W_src: np.ndarray | None = None
        self.W_tgt: np.ndarray | None = None

    def fit(self, X_src: np.ndarray, X_tgt: np.ndarray) -> "CCAAligner":
        model = cca_fit(X_src, X_tgt, n_components=self.n_components)
        self.n_components = model["n_components"]
        self.W_src = model["W_src"]
        self.W_tgt = model["W_tgt"]
        return self

    def transform_src(self, matrix: np.ndarray) -> np.ndarray:
        return _project(matrix, self.W_src)

    def transform_tgt(self, matrix: np.ndarray) -> np.ndarray:
        return _project(matrix, self.W_tgt)
