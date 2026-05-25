from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

def bli_precision_at_k(
    src_words: List[str],
    src_matrix: np.ndarray,
    tgt_words: List[str],
    tgt_matrix: np.ndarray,
    lexicon_pairs: List[Tuple[str, str]],
    *,
    k: int = 5,
    max_pairs: int = 2000,
) -> float:
    """Bilingual Lexicon Induction precision@k.

    For each (src, tgt) pair in the lexicon that appears in both
    vocabularies, projects the src vector and checks whether the
    correct English translation appears among the top-k nearest
    neighbours in English space.

    Returns the fraction of pairs where the correct translation
    is in the top-k retrievals.
    """
    src_index = {w: i for i, w in enumerate(src_words)}
    tgt_index = {w: i for i, w in enumerate(tgt_words)}

    valid = [
        (s, t) for s, t in lexicon_pairs
        if s in src_index and t in tgt_index
    ][:max_pairs]

    print(f"[BLI] {len(valid)} valid pairs (of {len(lexicon_pairs)} in lexicon, cap={max_pairs})")

    if not valid:
        return 0.0

    src_vecs = np.stack([src_matrix[src_index[s]] for s, _ in valid])

    scores = src_vecs @ tgt_matrix.T
    # top-k indices per pair, sorted descending
    topk_indices = np.argsort(scores, axis=1)[:, -k:]

    correct_indices = np.array([tgt_index[t] for _, t in valid])

    hits = sum(
        correct_indices[i] in topk_indices[i]
        for i in range(len(valid))
    )
    return hits / len(valid)