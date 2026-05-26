from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from alignment.VecMap import load_txt_embeddings, run_vecmap
from rq2.diagnostics.bli import bli_precision_at_k
from rq2.diagnostics.cka import compute_cka
from rq2.aligners.base import AlignerBase


def _l2(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def _l2_rows(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    return (M / np.maximum(norms, 1e-8)).astype(np.float32)


class VecMapWrapper(AlignerBase):

    def __init__(self, lang: str, fraction: float, lex_path: Path, force: bool = False):
        self.lang = lang
        self.fraction = fraction
        self.lex_path = lex_path
        self.force = force
        self._aligned_dict: Dict[str, np.ndarray] = {}
        # VecMap maps BOTH languages into a shared space, so it also transforms
        # the English side. We must keep this aligned-English dict and use it as
        # the reference for BLI/CKA — comparing against the original English
        # space (a different frame) collapses retrieval to ~0.
        self._en_aligned_dict: Dict[str, np.ndarray] = {}
        self._stats: Dict[str, int] = {"hit": 0, "total": 0}

    def fit(self, src_words, src_matrix, en_words, en_matrix, lexicon_pairs,
            en_matrix_orig=None):
        from config import OUTPUTS_ROOT, get_fraction_text_embeddings_path, get_text_embeddings_path, ENGLISH

        src_txt = get_fraction_text_embeddings_path(self.lang, self.fraction)
        en_txt = get_text_embeddings_path(ENGLISH)
        vecmap_dir = Path(OUTPUTS_ROOT) / f"vecmap_{self.lang}_{self.fraction:.2f}"
        src_aligned_path = vecmap_dir / "src_aligned.txt"
        tgt_aligned_path = vecmap_dir / "tgt_aligned.txt"

        if src_aligned_path.exists() and tgt_aligned_path.exists() and not self.force:
            self._aligned_dict = load_txt_embeddings(str(src_aligned_path))
            self._en_aligned_dict = load_txt_embeddings(str(tgt_aligned_path))
        else:
            self._aligned_dict, self._en_aligned_dict = run_vecmap(
                str(src_txt), str(en_txt), str(self.lex_path),
                output_dir=str(vecmap_dir),
            )
        return self

    def project(self, tokens: List[str], ft_model) -> np.ndarray:
        vecs = []
        for tok in tokens:
            self._stats["total"] += 1
            v = self._aligned_dict.get(tok)
            if v is None and tok.lower() != tok:
                v = self._aligned_dict.get(tok.lower())
            if v is not None:
                self._stats["hit"] += 1
                vecs.append(_l2(v))
            else:
                vecs.append(_l2(ft_model.get_word_vector(tok)))
        return np.stack(vecs, axis=0).astype(np.float32)

    @property
    def coverage(self) -> float:
        total = self._stats["total"]
        return self._stats["hit"] / total if total else 0.0

    def reset_stats(self):
        self._stats = {"hit": 0, "total": 0}

    def alignment_quality(self, src_words, src_matrix, en_words, en_matrix, lexicon_pairs):
        # English reference = VecMap's ALIGNED target embeddings (shared space),
        # NOT the original English space passed in en_words/en_matrix.
        aligned_words = list(self._aligned_dict.keys())
        aligned_matrix = _l2_rows(
            np.stack(list(self._aligned_dict.values())).astype(np.float32)
        )
        en_aligned_words = list(self._en_aligned_dict.keys())
        en_aligned_matrix = _l2_rows(
            np.stack(list(self._en_aligned_dict.values())).astype(np.float32)
        )

        bli = bli_precision_at_k(
            aligned_words, aligned_matrix,
            en_aligned_words, en_aligned_matrix,
            lexicon_pairs,
        )

        # CKA on the post-alignment geometry — both sides in VecMap's shared space.
        en_idx = {w: i for i, w in enumerate(en_aligned_words)}
        aligned_src, aligned_tgt = [], []
        for s, t in lexicon_pairs:
            v = self._aligned_dict.get(s)
            if v is None and s.lower() != s:
                v = self._aligned_dict.get(s.lower())
            if v is not None and t in en_idx:
                aligned_src.append(v)
                aligned_tgt.append(self._en_aligned_dict[t])

        if len(aligned_src) >= 10:
            A = _l2_rows(np.stack(aligned_src).astype(np.float32))
            B = _l2_rows(np.stack(aligned_tgt).astype(np.float32))
            cka = compute_cka(A, B)
        else:
            cka = float("nan")

        return {
            "bli_p5": bli,
            "cka": cka,
            "n_anchors": len(aligned_words),
            "coverage": self.coverage,
        }
        