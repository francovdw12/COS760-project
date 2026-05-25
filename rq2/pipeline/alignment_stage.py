from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from rq2.aligners.base import AlignerBase
from rq2.aligners.cca import CCAWrapper
from rq2.aligners.kcca import KCCAWrapper
from rq2.aligners.vecmap import VecMapWrapper
from rq2.pipeline.config import RQ2Config


def make_aligner(
    method: str,
    lang: str,
    fraction: float,
    lex_path: Path,
    config: RQ2Config,
) -> AlignerBase:
    """Alignment Factory — returns the correct AlignerBase 
    subclass for a given method.

    Accepts the full RQ2Config so each wrapper can read its own
    hyperparameters without the factory needing to know about them.
    """
    if method == "CCA":
        return CCAWrapper(
            lang=lang,
            fraction=fraction,
            force=config.force,
            n_components=config.cca_n_components,
        )
    if method == "KCCA":
        return KCCAWrapper(
            lang=lang,
            fraction=fraction,
            force=config.force,
            n_components=config.kcca_n_components,
            gamma=config.kcca_gamma,
            reg=config.kcca_reg,
            max_anchors=config.kcca_max_anchors,
        )
    if method == "VecMap":
        return VecMapWrapper(
            lang=lang,
            fraction=fraction,
            lex_path=lex_path,
            force=config.force,
        )
    raise ValueError(f"Unknown alignment method: {method}")


def fit_aligner(
    aligner: AlignerBase,
    src_words: List[str],
    src_matrix: np.ndarray,
    en_words: List[str],
    en_matrix: np.ndarray,
    lexicon_pairs: List[Tuple[str, str]],
) -> AlignerBase:
    """Stage 1 — fit the aligner on source and English anchor matrices.

    Separated from make_aligner so that construction and fitting
    are explicit distinct steps in the experiment runner.
    """
    aligner.fit(src_words, src_matrix, en_words, en_matrix, lexicon_pairs)
    return aligner


def run_diagnostics(
    aligner: AlignerBase,
    src_words: List[str],
    src_matrix: np.ndarray,
    en_words: List[str],
    en_matrix: np.ndarray,
    lexicon_pairs: List[Tuple[str, str]],
    method: str,
) -> Dict:
    """Diagnostic layer — compute BLI p@5 and CKA independently of NER.

    Returns a dict with bli_p5, cka, and n_anchors.
    Falls back to NaN values if diagnostics fail so the pipeline
    continues should a single language/fraction combination fail.
    """
    try:
        diag = aligner.alignment_quality(
            src_words, src_matrix, en_words, en_matrix, lexicon_pairs
        )
        print(
            f"  [{method}] BLI p@5={diag.get('bli_p5', float('nan')):.4f} "
            f"CKA={diag.get('cka', float('nan')):.4f} "
            f"anchors={diag.get('n_anchors', '?')}"
        )
        return diag
    except Exception as e:
        print(f"  [{method}] diagnostics failed: {e}")
        return {"bli_p5": float("nan"), "cka": float("nan"), "n_anchors": -1}
