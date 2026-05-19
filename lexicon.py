# lexicon.py - bilingual seed lexicon loading and anchor-matrix building.
import random
import numpy as np


def load_lexicon(path):
    """Load a bilingual lexicon (one "src_word<TAB>tgt_word" pair per line).

    Falls back to whitespace splitting if no tab is present.
    """
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                parts = line.split()
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def split_lexicon(pairs, held_out=1000, seed=42):
    """Split into train (seed lexicon) and test (held-out) pairs."""
    pairs = list(pairs)
    random.seed(seed)
    random.shuffle(pairs)
    if len(pairs) <= 1:
        return pairs[:0], pairs[:]

    held_out = min(held_out, len(pairs) - 1)
    return pairs[held_out:], pairs[:held_out]  # train, test


def build_anchor_matrices(train_pairs, src_words, src_matrix, tgt_words, tgt_matrix):
    """Build aligned anchor matrices (X_src, X_tgt) from training lexicon pairs.

    Only pairs whose words are present in both embedding vocabularies are kept.
    """
    src_idx = {w: i for i, w in enumerate(src_words)}
    tgt_idx = {w: i for i, w in enumerate(tgt_words)}

    X_src, X_tgt = [], []
    for sw, tw in train_pairs:
        if sw in src_idx and tw in tgt_idx:
            X_src.append(src_matrix[src_idx[sw]])
            X_tgt.append(tgt_matrix[tgt_idx[tw]])

    if not X_src:
        return np.empty((0, src_matrix.shape[1])), np.empty((0, tgt_matrix.shape[1]))

    return np.array(X_src), np.array(X_tgt)
