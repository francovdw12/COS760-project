from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from config import EMBEDDING_DIM, OUTPUTS_ROOT
from rq2.zero_shot_eval import (
    train_or_load_english_ner_model,
    evaluate_sentences,
    find_conll2003_splits,
    read_conll_sentences,
    BiLSTMCRF,
    _embed_sentence_fasttext,
)


def load_ner_model(
    *,
    force: bool = False,
    device: str = "cpu",
    epochs: int = 5,
) -> Tuple[BiLSTMCRF, dict, List[str]]:
    from config import get_conll2003_root, get_embeddings_path, ENGLISH

    en_bin = get_embeddings_path(ENGLISH)
    ner_ckpt = Path(OUTPUTS_ROOT) / "ner" / "bilstm_crf_conll2003.pt"

    return train_or_load_english_ner_model(
        conll_root=get_conll2003_root(),
        english_fasttext_bin=en_bin,
        out_path=ner_ckpt,
        embedding_dim=EMBEDDING_DIM,
        epochs=epochs,
        batch_size=16,
        seed=42,
        force_retrain=force,
        device=device,
    )


def validate_ner_model(
    model: BiLSTMCRF,
    idx_to_tag: List[str],
    allowed_types: List[str],
    *,
    device: str = "cpu",
    f1_threshold: float = 0.75,
) -> Dict:
    import fasttext
    from config import get_conll2003_root, get_embeddings_path, ENGLISH

    conll_root = get_conll2003_root()
    splits = find_conll2003_splits(conll_root)

    if "test" not in splits:
        print("[NER validation] CoNLL-2003 test split not found — skipping.")
        return {}

    test_sents = read_conll_sentences(splits["test"])
    en_bin = get_embeddings_path(ENGLISH)
    ft_en = fasttext.load_model(str(en_bin))
    cache = {}

    metrics = evaluate_sentences(
        model,
        idx_to_tag,
        test_sents,
        embedder=lambda toks: _embed_sentence_fasttext(ft_en, toks, cache),
        device=device,
        allowed_entity_types=allowed_types,
    )

    print(
        f"[NER validation] CoNLL-2003 test — "
        f"P={metrics['precision']:.4f} "
        f"R={metrics['recall']:.4f} "
        f"F1={metrics['f1']:.4f}"
    )

    if metrics["f1"] < f1_threshold:
        print(
            f"[NER validation] WARNING — F1 {metrics['f1']:.4f} is below threshold "
            f"{f1_threshold}. Zero-shot results may be unreliable."
        )

    return metrics


def entity_types_from_tagset(idx_to_tag: List[str]) -> List[str]:
    types = set()
    for tag in idx_to_tag:
        # exclude outside tag and malformed tags lacking PREFIX-TYPE structure
        if tag != "O" and "-" in tag:
            types.add(tag.split("-", 1)[1])
    return sorted(types)

