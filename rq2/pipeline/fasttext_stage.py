from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import numpy as np

from config import EMBEDDING_DIM
from embeddings import load_embeddings_as_matrix, save_embeddings_as_txt, train_fasttext

def ensure_fasttext_model(
    corpus_path: Path,
    bin_path: Path,
    txt_path: Path | None = None,
) -> None:
    """Trains a FastText model if it does not already exist.

    Also exports a text embedding file if txt_path is provided and
    does not already exist — required by VecMap which reads text format.
    """
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus: {corpus_path}")

    bin_path.parent.mkdir(parents=True, exist_ok=True)
    if not bin_path.exists():
        print(f"[FastText] training {bin_path.name} from {corpus_path}")
        train_fasttext(str(corpus_path), str(bin_path), dim=EMBEDDING_DIM)

    if txt_path is not None:
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        if not txt_path.exists():
            print(f"[FastText] exporting text embeddings to {txt_path.name}")
            save_embeddings_as_txt(str(bin_path), str(txt_path))


def load_source_embeddings(bin_path: Path) -> Tuple[List[str], np.ndarray]:
    """Load a FastText binary model as a (words, matrix) pair.

    Returns:
        words  — list of vocabulary strings, one per row
        matrix — (vocab_size, dim) float32 array
    """
    return load_embeddings_as_matrix(str(bin_path))


def supplement_source_embeddings(
    lang: str,
    words: List[str],
    matrix: np.ndarray,
    lexicon_pairs: List[Tuple[str, str]],
) -> Tuple[List[str], np.ndarray]:
    """Supplement source vocabulary with OOV lexicon words via FastText subword inference
    """
    from embeddings import supplement_with_oov
    from config import get_embeddings_path

    src_words = [sw for sw, _ in lexicon_pairs]
    return supplement_with_oov(get_embeddings_path(lang), words, matrix, src_words)


def supplement_english_embeddings(
    words: List[str],
    matrix: np.ndarray,
    all_lexicon_pairs: List[Tuple[str, str]],
) -> Tuple[List[str], np.ndarray]:
    """Supplement English vocabulary with OOV target words from all lexicons.
    """
    from embeddings import supplement_with_oov
    from config import get_embeddings_path, ENGLISH

    en_targets = [tw for _, tw in all_lexicon_pairs]
    return supplement_with_oov(get_embeddings_path(ENGLISH), words, matrix, en_targets)