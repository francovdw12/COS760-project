from __future__ import annotations

import numpy as np


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Centered Kernel Alignment between two matrices (using a linear kernel).

    Returns a scalar in [0, 1]; 1 means the spaces geometrically 
    are identical up to a rotation.
    """
    def centre(K: np.ndarray) -> np.ndarray:
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    Kx = centre(X @ X.T)
    Ky = centre(Y @ Y.T)

    hsic_xy = np.sum(Kx * Ky)
    hsic_xx = np.sum(Kx * Kx)
    hsic_yy = np.sum(Ky * Ky)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)