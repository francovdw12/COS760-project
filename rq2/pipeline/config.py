from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from config import CORPUS_FRACTIONS, LANGUAGES


@dataclass
class RQ2Config:
    # Run identity
    run_name: str = "default"

    # Scope
    langs: List[str] = field(default_factory=lambda: list(LANGUAGES))
    fractions: List[float] = field(default_factory=lambda: list(CORPUS_FRACTIONS))
    methods: List[str] = field(default_factory=lambda: ["CCA", "KCCA", "VecMap"])
    masakha_split: str = "test"

    # Runtime
    force: bool = False
    device: str = "cpu"
    validate_ner : bool = True
    ner_epochs: int = 5

    # Reproducibility — vary across runs (with --force + a distinct --run-name)
    # to obtain independent replicates for error bars.
    seed: int = 42

    # CCA
    cca_n_components: int = 100

    # KCCA
    kcca_n_components: int = 50
    kcca_gamma: float = 0.1
    kcca_reg: float = 1e-3
    kcca_max_anchors: int = 1000