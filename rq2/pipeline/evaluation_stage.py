from __future__ import annotations


from typing import Dict, List, Tuple

import numpy as np

from rq2.aligners.base import AlignerBase
from rq2.aligners.vecmap import VecMapWrapper
from rq2.zero_shot_eval import evaluate_sentences, Sentence


def run_evaluation(
    ner_model, 
    idx_to_tag: List[str],
    eval_sents: List[Sentence],
    aligner: AlignerBase,
    ft_model,
    allowed_types: List[str],
    device: str,
) -> Dict:
    """Stage 3 — evaluate the frozen English NER model on aligned source sentences..
    """

    # coverage statistics are only applicable to VecMap as only it has no provision
    # for OOV cross-lingual inference 
    if isinstance(aligner, VecMapWrapper):
        aligner.reset_stats()

    def embedder(tokens: List[str]) -> np.ndarray:
        return aligner.project(tokens, ft_model)

    return evaluate_sentences(
        ner_model,
        idx_to_tag,
        eval_sents,
        embedder=embedder,
        device=device,
        allowed_entity_types=allowed_types,
        unknown_type_strategy="to_misc",
    )


def assemble_result_row(
    *,
    lang: str,
    fraction: float,
    subset_tokens: int,
    method: str,
    metrics: Dict,
    diag: Dict,
    aligner: AlignerBase,
    en_baseline_f1: float = float("nan"), # default when validate_ner = False
    n_train_anchors: int = -1,
    cka_pre: float = float("nan"),
) -> Dict:
    """Assemble a single result row for the output CSV.
    """
    cka = diag.get("cka", float("nan"))

    return {
        "language": lang,
        "fraction": float(fraction),
        "subset_tokens": int(subset_tokens),
        "method": method,
        "en_baseline_f1": round(en_baseline_f1, 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1": round(metrics["f1"], 4),
        "bli_p5": round(diag.get("bli_p5", float("nan")), 4),
        "cka": round(cka, 4) if not np.isnan(cka) else "",
        "n_anchors": diag.get("n_anchors", -1),
        "n_train_anchors": n_train_anchors,
        "cka_pre": round(cka_pre, 4) if not np.isnan(cka_pre) else "",
        "vecmap_coverage": (
            round(aligner.coverage, 4)
            if isinstance(aligner, VecMapWrapper) else ""
        ),
    }