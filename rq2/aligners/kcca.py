from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


from alignment.KCCA import KCCAAligner as _KCCAAligner
from lexicon import build_anchor_matrices
from rq2.diagnostics.bli import bli_precision_at_k
from rq2.diagnostics.cka import compute_cka
from rq2.aligners.base import AlignerBase


def _l2(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def _l2_rows(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    return (M / np.maximum(norms, 1e-8)).astype(np.float32)


class KCCAWrapper(AlignerBase):

    def __init__(
        self,
        lang: str,
        fraction: float,
        gamma: float = 0.1, # controls localization of RBF similarity
        reg: float = 1e-3, 
        max_anchors: int = 1000,
        force: bool = False,
        n_components: int = 50,
    ):
        self.lang = lang
        self.fraction = fraction
        self.gamma = gamma
        self.reg = reg
        self.max_anchors = max_anchors
        self.force = force
        self._aligner: _KCCAAligner | None = None
        self._R: np.ndarray | None = None
        self.n_components = n_components

    def fit(self, src_words, src_matrix, en_words, en_matrix, lexicon_pairs):
        from config import get_alignment_artifact_path

        artifact = get_alignment_artifact_path(self.lang, self.fraction, "KCCA")

        if artifact.exists() and not self.force:
            data = np.load(artifact, allow_pickle=False)
            aligner = _KCCAAligner(
                n_components=int(data["n_components"]),
                gamma=float(data["gamma"]),
                reg=float(data["reg"]),
            )
            aligner.alpha = data["alpha"]
            aligner.beta = data["beta"]
            aligner.X_src_train = data["X_src_train"]
            aligner.X_tgt_train = data["X_tgt_train"]
            self._aligner = aligner
            self._R = data["R"]
            return self

        X_src, X_tgt = build_anchor_matrices(
            lexicon_pairs, src_words, src_matrix, en_words, en_matrix
        )
        if len(X_src) < 5:
            raise ValueError(
                f"Not enough anchor pairs for {self.lang} at fraction={self.fraction}"
            )

        n = min(self.max_anchors, len(X_src))
        n_components = min(self.n_components, n)
        aligner = _KCCAAligner(n_components=n_components, gamma=self.gamma, reg=self.reg)
        aligner.fit(X_src[:n], X_tgt[:n])

        # Use raw KCCA projections without L2 normalisation
        Z_tgt = (aligner._get_ky_train() @ aligner.beta).astype(np.float32)
        R = np.linalg.lstsq(Z_tgt, X_tgt[:n], rcond=None)[0].astype(np.float32)

        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            artifact,
            n_components=n_components,
            gamma=self.gamma,
            reg=self.reg,
            alpha=aligner.alpha.astype(np.float32),
            beta=aligner.beta.astype(np.float32),
            X_src_train=aligner.X_src_train.astype(np.float32),
            X_tgt_train=aligner.X_tgt_train.astype(np.float32),
            R=R,
        )
        self._aligner = aligner
        self._R = R
        return self

    def project(self, tokens: List[str], ft_model) -> np.ndarray:
        vecs = np.stack([_l2(ft_model.get_word_vector(tok)) for tok in tokens])
        Z = self._aligner.transform_src(vecs)
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
        