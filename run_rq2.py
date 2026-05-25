from __future__ import annotations

import argparse
from typing import Sequence

from config import CORPUS_FRACTIONS, LANGUAGES
from rq2.pipeline.config import RQ2Config
from rq2.pipeline.experiment_runner import run_rq2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="RQ2: corpus fractions + zero-shot NER"
    )
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--langs", nargs="*", default=LANGUAGES)
    parser.add_argument("--fractions", nargs="*", type=float, default=CORPUS_FRACTIONS)
    parser.add_argument("--methods", nargs="*", default=["CCA", "KCCA", "VecMap"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cca-n-components", type=int, default=100)
    parser.add_argument("--kcca-n-components", type=int, default=50)
    parser.add_argument("--kcca-gamma", type=float, default=0.1)
    parser.add_argument("--kcca-reg", type=float, default=1e-3)
    parser.add_argument("--kcca-max-anchors", type=int, default=1000)
    parser.add_argument("--no-validate-ner", action="store_false", dest="validate_ner")
    parser.add_argument("--ner-epochs", type=int, default=10)
    args = parser.parse_args(argv)

    config = RQ2Config(
        run_name=args.run_name,
        langs=args.langs,
        fractions=args.fractions,
        methods=args.methods,
        ner_epochs=args.ner_epochs,
        masakha_split=args.split,
        force=args.force,
        device=args.device,
        cca_n_components=args.cca_n_components,
        kcca_n_components=args.kcca_n_components,
        kcca_gamma=args.kcca_gamma,
        kcca_reg=args.kcca_reg,
        kcca_max_anchors=args.kcca_max_anchors,
        validate_ner=args.validate_ner,
    )

    run_rq2(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())