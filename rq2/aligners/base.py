from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np


class AlignerBase(ABC):

    @abstractmethod
    def fit(
        self,
        src_words: List[str],
        src_matrix: np.ndarray,
        en_words: List[str],
        en_matrix: np.ndarray,
        lexicon_pairs: List[Tuple[str, str]],

    ) -> "AlignerBase": """Learn the mapping from source language space into English embedding space."""

    @abstractmethod
    def project(self, tokens: List[str], ft_model) -> np.ndarray:
        """Map a list of tokens to a (n_tokens, dim) float32 array in English embedding space."""


    @abstractmethod
    def alignment_quality(
        self,
        src_words: List[str],
        src_matrix: np.ndarray,
        en_words: List[str],
        en_matrix: np.ndarray,
        lexicon_pairs: List[Tuple[str, str]],
    ) -> Dict:
        """Return diagnostic metrics — BLI p@1, CKA, anchor count — independently of NER."""
