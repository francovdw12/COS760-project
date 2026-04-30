import numpy as np
from scipy.linalg import eigh

class KCCAAligner:
    """
    Kernel CCA with an RBF kernel.
    Captures non-linear structure, a key hypothesis for isiZulu.

    Reference: Hardoon et al. (2004)
    """
    def __init__(self, n_components: int = 50, 
                 gamma: float = 1.0, 
                 reg: float = 1e-4):
        self.n_components = n_components
        self.gamma = gamma          # RBF kernel parameter
        self.reg = reg              # Tikhonov regularization
        self.alpha = None           # source coefficients
        self.beta = None            # target coefficients
        self.X_src_train = None
        self.X_tgt_train = None

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute the Gram matrix K(X, Y) with an RBF kernel."""
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x dot y
        sq_X = np.sum(X**2, axis=1, keepdims=True)
        sq_Y = np.sum(Y**2, axis=1, keepdims=True)
        dist_sq = sq_X + sq_Y.T - 2 * X @ Y.T
        return np.exp(-self.gamma * dist_sq)

    def _center_kernel(self, K: np.ndarray) -> np.ndarray:
        """Center the Gram matrix (required for KCCA)."""
        n = K.shape[0]
        one = np.ones((n, n)) / n
        return K - one @ K - K @ one + one @ K @ one

    def fit(self, X_src: np.ndarray, X_tgt: np.ndarray):
        n = X_src.shape[0]
        self.X_src_train = X_src
        self.X_tgt_train = X_tgt

        Kx = self._center_kernel(self._rbf_kernel(X_src, X_src))
        Ky = self._center_kernel(self._rbf_kernel(X_tgt, X_tgt))

        # Regularization
        reg_x = Kx + self.reg * np.eye(n)
        reg_y = Ky + self.reg * np.eye(n)

        # Generalized KCCA eigenvalue problem
        # Solve: Kx @ Ky @ alpha = lambda * reg_x @ reg_y @ alpha
        M = np.linalg.solve(reg_x, Kx) @ np.linalg.solve(reg_y, Ky)
        eigenvalues, eigenvectors = eigh(M)

        # Keep the largest n_components eigenvalues
        idx = np.argsort(eigenvalues)[::-1][:self.n_components]
        self.alpha = eigenvectors[:, idx]  # (n_pairs, n_components)

        # Compute beta symmetrically
        M2 = np.linalg.solve(reg_y, Ky) @ np.linalg.solve(reg_x, Kx)
        _, eigenvectors2 = eigh(M2)
        idx2 = np.argsort(_)[::-1][:self.n_components]
        self.beta = eigenvectors2[:, idx2]

        return self

    def transform_src(self, X_new: np.ndarray) -> np.ndarray:
        """Project new source vectors into the KCCA space."""
        K = self._rbf_kernel(X_new, self.X_src_train)
        # Center relative to the training data
        n_train = self.X_src_train.shape[0]
        K -= K.mean(axis=1, keepdims=True)
        projected = K @ self.alpha
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        return projected / np.maximum(norms, 1e-8)

    def transform_tgt(self, X_new: np.ndarray) -> np.ndarray:
        K = self._rbf_kernel(X_new, self.X_tgt_train)
        K -= K.mean(axis=1, keepdims=True)
        projected = K @ self.beta
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        return projected / np.maximum(norms, 1e-8)