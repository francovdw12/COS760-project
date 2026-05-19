# evaluation.py - intrinsic metrics for RQ1 alignment quality.
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def precision_at_k(src_aligned, tgt_matrix, test_pairs, src_words, tgt_words, k=1):
    """Word-translation precision at k.

    For each test pair (src_word, tgt_word):
    - find the k nearest English neighbours of src_word in tgt_matrix,
    - count a hit if tgt_word is among them.

    src_aligned rows must correspond to src_words; tgt_matrix rows to tgt_words.
    """
    src_idx = {w: i for i, w in enumerate(src_words)}
    tgt_idx = {w: i for i, w in enumerate(tgt_words)}

    hits = 0
    total = 0
    for sw, tw in test_pairs:
        if sw not in src_idx or tw not in tgt_idx:
            continue
        i = src_idx[sw]
        j = tgt_idx[tw]
        if i >= src_aligned.shape[0] or j >= tgt_matrix.shape[0]:
            continue
        query = src_aligned[i].reshape(1, -1)
        sims = cosine_similarity(query, tgt_matrix)[0]
        top_k = np.argsort(sims)[::-1][:k]
        if j in top_k:
            hits += 1
        total += 1

    return hits / total if total > 0 else 0.0


def mean_cosine_similarity(src_aligned, tgt_matrix, test_pairs, src_idx, tgt_idx):
    """Mean cosine similarity between aligned translation pairs."""
    sims = []
    for sw, tw in test_pairs:
        if sw not in src_idx or tw not in tgt_idx:
            continue
        i = src_idx[sw]
        j = tgt_idx[tw]
        if i >= src_aligned.shape[0] or j >= tgt_matrix.shape[0]:
            continue
        v1 = src_aligned[i]
        v2 = tgt_matrix[j]
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        sims.append(cos)
    return float(np.mean(sims)) if sims else 0.0


def linear_cka(X, Y):
    """Linear Centered Kernel Alignment (Kornblith et al., 2019).

    Measures geometric similarity between two embedding spaces; invariant to
    orthogonal transforms and isotropic scaling.
        - CKA near 1 -> spaces are very similar
        - CKA low     -> geometric divergence

    Uses the equivalent formulation tr(XX^T · YY^T) = ||X^T Y||_F^2
    which operates on (dim x dim) matrices instead of (n x n), making it
    tractable for large vocabularies.
    """
    X = (X - X.mean(axis=0)).astype(np.float32)
    Y = (Y - Y.mean(axis=0)).astype(np.float32)

    # All three products are (dim, dim) — memory cost is O(dim^2), not O(n^2)
    XtY = X.T @ Y
    XtX = X.T @ X
    YtY = Y.T @ Y

    hsic_kl = float(np.sum(XtY ** 2))
    hsic_kk = float(np.sqrt(np.sum(XtX ** 2)))
    hsic_ll = float(np.sqrt(np.sum(YtY ** 2)))

    return hsic_kl / (hsic_kk * hsic_ll + 1e-8)
