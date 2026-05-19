# Kernel CCA alignment (RQ1).
# Non-linear extension of CCA using an RBF kernel. The hypothesis is that
# isiZulu's conjunctive morphology produces a non-linear embedding geometry
# that linear CCA/VecMap cannot fully capture.
# Reference: Hardoon et al. (2004).
import numpy as np


def _rbf_kernel(X, Y, gamma):
    """RBF Gram matrix K(X, Y) = exp(-gamma * ||x - y||^2)."""
    sq_X = np.sum(X ** 2, axis=1, keepdims=True)
    sq_Y = np.sum(Y ** 2, axis=1, keepdims=True)
    dist_sq = sq_X + sq_Y.T - 2.0 * (X @ Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    return np.exp(-gamma * dist_sq)


def _center_train_kernel(K):
    """Center a square training Gram matrix."""
    n = K.shape[0]
    one = np.ones((n, n)) / n
    return K - one @ K - K @ one + one @ K @ one


def _center_test_kernel(K_test, K_train):
    """Center a test Gram matrix (n_test, n_train) using training statistics.

    K_train is the *uncentered* training Gram matrix.
    """
    n = K_train.shape[0]
    n_test = K_test.shape[0]
    ones_test = np.ones((n_test, n)) / n
    ones_train = np.ones((n, n)) / n
    return (
        K_test
        - ones_test @ K_train
        - K_test @ ones_train
        + ones_test @ K_train @ ones_train
    )


def kcca_fit(X_src, X_tgt, n_components=50, gamma=1.0, reg=1e-4):
    """Fit Kernel CCA on aligned anchor matrices.

    Solves the generalised eigenvalue problem. The matrix M is not symmetric,
    so np.linalg.eig is used (eigh would silently read only the lower triangle).
    """
    n = X_src.shape[0]
    n_components = min(n_components, n)

    Kx_raw = _rbf_kernel(X_src, X_src, gamma)
    Ky_raw = _rbf_kernel(X_tgt, X_tgt, gamma)
    Kx = _center_train_kernel(Kx_raw)
    Ky = _center_train_kernel(Ky_raw)

    reg_x = Kx + reg * np.eye(n)
    reg_y = Ky + reg * np.eye(n)

    M_src = np.linalg.solve(reg_x, Kx) @ np.linalg.solve(reg_y, Ky)
    eigvals_src, eigvecs_src = np.linalg.eig(M_src)
    order_src = np.argsort(np.real(eigvals_src))[::-1][:n_components]
    alpha = np.real(eigvecs_src[:, order_src])

    M_tgt = np.linalg.solve(reg_y, Ky) @ np.linalg.solve(reg_x, Kx)
    eigvals_tgt, eigvecs_tgt = np.linalg.eig(M_tgt)
    order_tgt = np.argsort(np.real(eigvals_tgt))[::-1][:n_components]
    beta = np.real(eigvecs_tgt[:, order_tgt])

    return {
        "n_components": n_components,
        "gamma": gamma,
        "reg": reg,
        "alpha": alpha,
        "beta": beta,
        "X_src_train": X_src,
        "X_tgt_train": X_tgt,
        "Kx_train": Kx_raw,   # uncentered, needed to center test kernels
        "Ky_train": Ky_raw,
    }


def _transform(X_new, X_train, K_train, coeff, gamma, batch_size=2000):
    """Project new vectors into the KCCA space, batched to bound memory."""
    out = []
    for start in range(0, X_new.shape[0], batch_size):
        chunk = X_new[start:start + batch_size]
        K = _rbf_kernel(chunk, X_train, gamma)
        K = _center_test_kernel(K, K_train)
        projected = K @ coeff
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        out.append(projected / np.maximum(norms, 1e-8))
    return np.vstack(out)


def kcca_transform_src(model, X_new):
    """Project source-language vectors into the shared KCCA space."""
    return _transform(
        X_new, model["X_src_train"], model["Kx_train"],
        model["alpha"], model["gamma"],
    )


def kcca_transform_tgt(model, X_new):
    """Project English vectors into the shared KCCA space."""
    return _transform(
        X_new, model["X_tgt_train"], model["Ky_train"],
        model["beta"], model["gamma"],
    )


class KCCAAligner:
    """Class-based wrapper around kcca_fit / _transform for use by run_rq1 and run_rq2.

    Supports lazy recomputation of the training kernel matrices when an aligner
    is reconstructed from a saved artifact (which omits the kernel arrays).
    """

    def __init__(self, n_components: int = 50, gamma: float = 1.0, reg: float = 1e-4):
        self.n_components = n_components
        self.gamma = gamma
        self.reg = reg
        self.alpha: np.ndarray | None = None
        self.beta: np.ndarray | None = None
        self.X_src_train: np.ndarray | None = None
        self.X_tgt_train: np.ndarray | None = None
        self._Kx_train: np.ndarray | None = None
        self._Ky_train: np.ndarray | None = None

    def fit(self, X_src: np.ndarray, X_tgt: np.ndarray) -> "KCCAAligner":
        model = kcca_fit(X_src, X_tgt, n_components=self.n_components, gamma=self.gamma, reg=self.reg)
        self.n_components = model["n_components"]
        self.alpha = model["alpha"]
        self.beta = model["beta"]
        self.X_src_train = model["X_src_train"]
        self.X_tgt_train = model["X_tgt_train"]
        self._Kx_train = model["Kx_train"]
        self._Ky_train = model["Ky_train"]
        return self

    def _get_kx_train(self) -> np.ndarray:
        if self._Kx_train is None:
            self._Kx_train = _rbf_kernel(self.X_src_train, self.X_src_train, self.gamma)
        return self._Kx_train

    def _get_ky_train(self) -> np.ndarray:
        if self._Ky_train is None:
            self._Ky_train = _rbf_kernel(self.X_tgt_train, self.X_tgt_train, self.gamma)
        return self._Ky_train

    def transform_src(self, X_new: np.ndarray) -> np.ndarray:
        return _transform(X_new, self.X_src_train, self._get_kx_train(), self.alpha, self.gamma)

    def transform_tgt(self, X_new: np.ndarray) -> np.ndarray:
        return _transform(X_new, self.X_tgt_train, self._get_ky_train(), self.beta, self.gamma)
