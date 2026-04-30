# evaluation.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -- Precision at k --
def precision_at_k(src_aligned: np.ndarray, tgt_matrix: np.ndarray,
                   test_pairs, src_words, tgt_words, k: int = 1) -> float:
    """
    For each test pair (sw, tw):
    - find the k nearest neighbors of sw in tgt_matrix
    - success if tw is among them
    """
    src_idx = {w: i for i, w in enumerate(src_words)}
    tgt_idx = {w: i for i, w in enumerate(tgt_words)}

    hits = 0
    total = 0
    for sw, tw in test_pairs:
        if sw not in src_idx or tw not in tgt_idx:
            continue
        query = src_aligned[src_idx[sw]].reshape(1, -1)
        sims = cosine_similarity(query, tgt_matrix)[0]
        top_k = np.argsort(sims)[::-1][:k]
        if tgt_idx[tw] in top_k:
            hits += 1
        total += 1

    return hits / total if total > 0 else 0.0

# -- Mean cosine similarity --
def mean_cosine_similarity(src_aligned: np.ndarray, tgt_matrix: np.ndarray,
                            test_pairs, src_idx, tgt_idx) -> float:
    sims = []
    for sw, tw in test_pairs:
        if sw in src_idx and tw in tgt_idx:
            v1 = src_aligned[src_idx[sw]]
            v2 = tgt_matrix[tgt_idx[tw]]
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            sims.append(cos)
    return float(np.mean(sims)) if sims else 0.0

# -- CKA (Centered Kernel Alignment) --
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA (Kornblith et al., 2019).
    Measures geometric similarity between two embedding spaces.
    Invariant to orthogonal transforms and isotropic scaling.

    Interpretation:
        - CKA near 1 -> spaces are very similar (favors linear alignment)
        - CKA low -> geometric divergence (favors KCCA)
    """
    # Center the matrices
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Linear Gram matrices
    K = X @ X.T
    L = Y @ Y.T

    hsic_kl = np.sum(K * L)
    hsic_kk = np.sqrt(np.sum(K * K))
    hsic_ll = np.sqrt(np.sum(L * L))

    return hsic_kl / (hsic_kk * hsic_ll + 1e-8)