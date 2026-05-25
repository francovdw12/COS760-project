from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from alignment.VecMap import load_txt_embeddings, run_vecmap
from rq2.diagnostics.bli import bli_precision_at_k
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
        self._stats: Dict[str, int] = {"hit": 0, "total": 0}

    def fit(self, src_words, src_matrix, en_words, en_matrix, lexicon_pairs):
        from config import OUTPUTS_ROOT, get_fraction_text_embeddings_path, get_text_embeddings_path, ENGLISH

        src_txt = get_fraction_text_embeddings_path(self.lang, self.fraction)
        en_txt = get_text_embeddings_path(ENGLISH)
        vecmap_dir = Path(OUTPUTS_ROOT) / f"vecmap_{self.lang}_{self.fraction:.2f}"
        src_aligned_path = vecmap_dir / "src_aligned.txt"

        if src_aligned_path.exists() and not self.force:
            self._aligned_dict = load_txt_embeddings(str(src_aligned_path))
        else:
            self._aligned_dict, _ = run_vecmap(
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
        aligned_words = list(self._aligned_dict.keys())
        aligned_matrix = _l2_rows(
            np.stack(list(self._aligned_dict.values())).astype(np.float32)
        )
        en_norm = _l2_rows(en_matrix)

        bli = bli_precision_at_k(
            aligned_words, aligned_matrix,
            en_words, en_norm,
            lexicon_pairs,
        )
        return {
            "bli_p5": bli,
            "cka": float("nan"),
            "n_anchors": len(aligned_words),
            "coverage": self.coverage,
        }
        