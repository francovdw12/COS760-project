from __future__ import annotations


from pathlib import Path
from typing import Dict, List, Tuple


import numpy as np


from alignment.CCA import CCAAligner as _CCAAligner
from lexicon import build_anchor_matrices
from rq2.diagnostics.bli import bli_precision_at_k
from rq2.diagnostics.cka import compute_cka
from rq2.aligners.base import AlignerBase


def _l2(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def _l2_rows(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    return (M / np.maximum(norms, 1e-8)).astype(np.float32)


class CCAWrapper(AlignerBase):

    def __init__(self, lang: str, fraction: float, force: bool = False,
    n_components: int = 100):
        self.lang = lang
        self.fraction = fraction
        self.force = force # forcing recomputation of existing artefact
        self.n_components = n_components
        self._aligner: _CCAAligner | None = None
        self._R: np.ndarray | None = None # regression back to the english space

    def fit(self, src_words, src_matrix, en_words, en_matrix, lexicon_pairs):
        from config import get_alignment_artifact_path

        artifact = get_alignment_artifact_path(self.lang, self.fraction, "CCA")

        if artifact.exists() and not self.force:
            data = np.load(artifact, allow_pickle=False)
            aligner = _CCAAligner(n_components=int(data["n_components"]))
            aligner.W_src = data["W_src"]
            aligner.W_tgt = data["W_tgt"]
            self._aligner = aligner
            self._R = data["R"]
            return self

        X_src, X_tgt = build_anchor_matrices(
            lexicon_pairs, src_words, src_matrix, en_words, en_matrix
        )
        if len(X_src) < 2:
            raise ValueError(
                f"Not enough anchor pairs for {self.lang} at fraction={self.fraction}"
            )

        # 100 -- tune this cap?
        n_components = min(self.n_components, X_src.shape[0], X_src.shape[1], X_tgt.shape[1])
        print(f"CCA components used = {n_components}\n")
        aligner = _CCAAligner(n_components=n_components)
        aligner.fit(X_src, X_tgt)

        # Normalisation collapses magnitude variation that lstsq needs
        Z_en_full = (en_matrix @ aligner.W_tgt).astype(np.float32)
        R = np.linalg.lstsq(Z_en_full, en_matrix, rcond=None)[0].astype(np.float32)

        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            artifact,
            n_components=n_components,
            W_src=aligner.W_src.astype(np.float32),
            W_tgt=aligner.W_tgt.astype(np.float32),
            R=R,
        )
        self._aligner = aligner
        self._R = R
        return self

    def project(self, tokens: List[str], ft_model) -> np.ndarray:
        vecs = np.stack([_l2(ft_model.get_word_vector(tok)) for tok in tokens])
        # Raw CCA projection without normalisation before applying R
        Z = (vecs @ self._aligner.W_src).astype(np.float32)
        Y = (Z @ self._R).astype(np.float32)
        return _l2_rows(Y)
        
    def alignment_quality(self, src_words, src_matrix, en_words, en_matrix, lexicon_pairs):
        X_src, X_tgt = build_anchor_matrices(
            lexicon_pairs, src_words, src_matrix, en_words, en_matrix
        )
        projected = self._aligner.transform_src(src_matrix)
        projected_in_en = _l2_rows((projected @ self._R).astype(np.float32))

        bli = bli_precision_at_k(
            src_words, projected_in_en,
            en_words, _l2_rows(en_matrix),
            lexicon_pairs,
        )
        cka = compute_cka(X_src, X_tgt) if X_src.shape[0] >= 10 else float("nan")
        return {"bli_p5": bli, "cka": cka, "n_anchors": len(X_src)}