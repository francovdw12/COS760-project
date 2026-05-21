# lexicon.py - bilingual seed lexicon loading and anchor-matrix building.
import random
import numpy as np


def _clean_token(tok):
    """Strip NCHLT verb-infinitive prefix dash and whitespace."""
    return tok.strip().lstrip("-").strip()


def _expand_slash_entry(src_raw, tgt_raw):
    """Expand 'a / b → x / y' into [(a,x),(a,y),(b,x),(b,y)], single-word only."""
    src_parts = [_clean_token(s) for s in src_raw.split("/")]
    tgt_parts = [_clean_token(t) for t in tgt_raw.split("/")]
    pairs = []
    for s in src_parts:
        for t in tgt_parts:
            if s and t and " " not in s and " " not in t:
                pairs.append((s, t))
    return pairs


def load_lexicon(path):
    """Load a bilingual lexicon (one "src_word<TAB>tgt_word" pair per line).

    Handles three entry formats found in NCHLT lexicons:
      - Simple single-word pairs:  hamba\\tgo
      - Slash-separated synonyms:  -ahlula / -mangaza\\toverwhelm
      - Multi-word entries (skipped — not usable for word-level alignment)

    Leading dashes on verb stems (NCHLT infinitive convention) are stripped.
    """
    pairs = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            src_raw, tgt_raw = parts

            expanded = _expand_slash_entry(src_raw, tgt_raw)
            for pair in expanded:
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
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
