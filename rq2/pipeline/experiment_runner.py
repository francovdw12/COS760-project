from __future__ import annotations


import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from config import (
    ENGLISH,
    OUTPUTS_ROOT,
    RESULTS_ROOT,
    get_corpus_path,
    get_display_name,
    get_embeddings_path,
    get_fraction_embeddings_path,
    get_fraction_text_embeddings_path,
    get_ner_path,
    get_seed_lexicon_path,
    get_subset_corpus_path,
    get_text_embeddings_path,
)
from lexicon import load_lexicon
from rq2.corpus_subsets import ensure_language_subsets
from rq2.zero_shot_eval import load_masakhaner_split, Sentence
from rq2.pipeline.config import RQ2Config
from rq2.pipeline.fasttext_stage import ensure_fasttext_model, load_source_embeddings
from rq2.pipeline.ner_stage import load_ner_model, validate_ner_model, entity_types_from_tagset
from rq2.pipeline.alignment_stage import make_aligner, fit_aligner, run_diagnostics
from rq2.pipeline.evaluation_stage import run_evaluation, assemble_result_row

SEP_CHAR_NUM = 100 # separation characters for serial results display

def run_rq2(config: RQ2Config) -> None:
    """Main pipeline entry point.

    Accepts a single RQ2Config object so that multiple experimental
    runs differ only in their config — no code changes required.
    """
    results: List[Dict] = []

    # Output directories
    results_dir = Path(RESULTS_ROOT) / "rq2" / config.run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    Path(OUTPUTS_ROOT).mkdir(parents=True, exist_ok=True)

    # Configuration parameters serialization to save hyperparameters alongside results
    config_path = results_dir / "config_params.json"
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=2)
    print(f"[RQ2] config saved to {config_path}")

    # English FastText embeddings
    en_corpus = Path(get_corpus_path(ENGLISH))
    en_bin = get_embeddings_path(ENGLISH)
    en_txt = get_text_embeddings_path(ENGLISH)
    try:
        ensure_fasttext_model(en_corpus, en_bin, en_txt)
    except FileNotFoundError as e:
        print(f"[RQ2] {e}")
        return

    # English NER model (trained once and frozen) 
    try:
        ner_model, _tag_to_idx, idx_to_tag = load_ner_model(
            force=config.force,
            device=config.device,
            epochs=config.ner_epochs,
        )
    except FileNotFoundError as e:
        print(f"[RQ2] {e}")
        return

    allowed_types = entity_types_from_tagset(idx_to_tag)
    print(f"[NER] tag types: {allowed_types}")

    # English NER validation
    en_baseline_f1 = float("nan") # default when baseline evaluatio is not run
    if config.validate_ner:
        val_metrics = validate_ner_model(
            ner_model,
            idx_to_tag,
            allowed_types,
            device=config.device,
        )
        en_baseline_f1 = val_metrics.get("f1", float("nan"))

        en_words, en_matrix = load_source_embeddings(en_bin)

        # Collect all English target words from all language lexicons for supplementation
        all_en_targets = []
        for _lang in config.langs:
            _lex = get_seed_lexicon_path(_lang)
            if _lex and Path(_lex).exists():
                all_en_targets.extend(load_lexicon(str(_lex)))

        from rq2.pipeline.fasttext_stage import supplement_english_embeddings
        en_words, en_matrix = supplement_english_embeddings(en_words, en_matrix, all_en_targets)
        print(f"[RQ2] English vocab after supplementation: {len(en_words)} words")

    # Language evaluation loop
    for lang in config.langs:
        display = get_display_name(lang)
        print(f"\n{'=' * SEP_CHAR_NUM}\nRQ2 language: {display} ({lang})\n{'=' * SEP_CHAR_NUM}")

        lex_path = get_seed_lexicon_path(lang)
        if lex_path is None or not lex_path.exists():
            print(f"[RQ2] Missing seed lexicon for {lang} (skipping)")
            continue

        lexicon_pairs = load_lexicon(str(lex_path))
        if not lexicon_pairs:
            print(f"[RQ2] Empty seed lexicon for {lang} (skipping)")
            continue

        ner_dir = get_ner_path(lang)
        if ner_dir is None or not Path(ner_dir).exists():
            print(f"[RQ2] Missing MasakhaNER folder for {lang} (skipping)")
            continue

        try:
            eval_sents: List[Sentence] = load_masakhaner_split(
                Path(ner_dir), config.masakha_split
            )
        except Exception as e:
            print(f"[RQ2] Failed to load MasakhaNER split for {lang}: {e}")
            continue

        try:
            subset_stats = ensure_language_subsets(
                lang, fractions=config.fractions, seed=42, force=config.force
            )
        except FileNotFoundError as e:
            print(f"[RQ2] {e} — skipping {lang}")
            continue

        token_count_by_fraction = {s.fraction: s.selected_tokens for s in subset_stats}

        import fasttext

        # Data fraction (portion) loop
        for fraction in config.fractions:
            subset_corpus = get_subset_corpus_path(lang, fraction)
            if not subset_corpus.exists():
                print(f"[RQ2] Missing subset corpus {subset_corpus} (skipping)")
                continue

            src_bin = get_fraction_embeddings_path(lang, fraction)
            src_txt = get_fraction_text_embeddings_path(lang, fraction)
            ensure_fasttext_model(subset_corpus, src_bin, src_txt)

            src_words, src_matrix = load_source_embeddings(src_bin)
            ft_src = fasttext.load_model(str(src_bin))

            from rq2.pipeline.fasttext_stage import supplement_source_embeddings
            src_words, src_matrix = supplement_source_embeddings(
                lang, src_words, src_matrix, lexicon_pairs
            )
            print(f"[RQ2] {lang} vocab after supplementation: {len(src_words)} words")
            ft_src = fasttext.load_model(str(src_bin))

            # Method loop
            for method in config.methods:
                print(f"\n[RQ2] lang={lang} fraction={fraction:.2f} method={method}")

                # Embeddings Alignment
                aligner = make_aligner(method, lang, fraction, lex_path, config)
                try:
                    fit_aligner(
                        aligner, src_words, src_matrix,
                        en_words, en_matrix, lexicon_pairs
                    )
                except Exception as e:
                    print(f"  [{method}] fit failed: {e}")
                    continue

                # Embeddings alignment diagnostics
                diag = run_diagnostics(
                    aligner, src_words, src_matrix,
                    en_words, en_matrix, lexicon_pairs,
                    method,
                )

                # NER task performance evaluation
                metrics = run_evaluation(
                    ner_model, idx_to_tag, eval_sents,
                    aligner=aligner,
                    ft_model=ft_src,
                    allowed_types=allowed_types,
                    device=config.device,
                )

                results.append(assemble_result_row(
                    lang=lang,
                    fraction=fraction,
                    subset_tokens=int(token_count_by_fraction.get(float(fraction), -1)),
                    method=method,
                    metrics=metrics,
                    diag=diag,
                    aligner=aligner,
                    en_baseline_f1=en_baseline_f1,
                ))

    if not results:
        print("[RQ2] No results produced (missing data?).")
        return

    df = pd.DataFrame(results)
    out_csv = results_dir / "results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[RQ2] wrote {out_csv} ({len(df)} rows)")
