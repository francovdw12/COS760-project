import numpy as np
import pandas as pd
from pathlib import Path
from embeddings import load_embeddings_as_matrix, save_embeddings_as_txt, train_fasttext, supplement_with_oov
from lexicon import load_lexicon, split_lexicon, build_anchor_matrices
from alignment.CCA import CCAAligner
from alignment.KCCA import KCCAAligner
from alignment.VecMap import run_vecmap
from evaluation import precision_at_k, mean_cosine_similarity, linear_cka
from transfer.zero_shot_eval import (
    evaluate_sentences,
    load_masakhaner_split,
    train_or_load_english_ner_model,
)
from config import (
    EMBEDDING_DIM,
    ENGLISH,
    HELD_OUT_PAIRS,
    LANGUAGES,
    OUTPUTS_ROOT,
    PROJECT_ROOT,
    RESULTS_ROOT,
    get_conll2003_root,
    get_corpus_path,
    get_display_name,
    get_embeddings_path,
    get_ner_path,
    get_seed_lexicon_path,
    get_text_embeddings_path,
)

def get_or_train_embeddings(lang):
    emb_path = get_embeddings_path(lang)
    if emb_path.exists():
        words, matrix = load_embeddings_as_matrix(str(emb_path))
    else:
        corpus_path = get_corpus_path(lang)
        display_name = get_display_name(lang)
        if not corpus_path.exists():
            print(f"ERROR: Missing corpus for {display_name} ({lang}) to train embeddings: {corpus_path}")
            return None, None

        print(f"Training FastText for {display_name} ({lang}) from {corpus_path}")
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        train_fasttext(str(corpus_path), str(emb_path))
        words, matrix = load_embeddings_as_matrix(str(emb_path))

    txt_path = get_text_embeddings_path(lang)
    if not txt_path.exists():
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        save_embeddings_as_txt(str(emb_path), str(txt_path))

    return words, matrix

def _entity_types_from_tagset(idx_to_tag):
    types = set()
    for tag in idx_to_tag:
        if tag != "O" and "-" in tag:
            types.add(tag.split("-", 1)[1])
    return sorted(types)


def run_rq1():
    results = []
    results_dir = Path(RESULTS_ROOT)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading/training English embeddings...")
    en_words_orig, en_matrix_orig = get_or_train_embeddings(ENGLISH)
    if en_words_orig is None:
        print("Failed to get English embeddings. Stopping.")
        return

    n_en_orig = len(en_words_orig)
    en_idx_orig = {w: i for i, w in enumerate(en_words_orig)}

    # Supplement English with ALL target words from all bilingual lexicons.
    # FastText computes subword-based vectors even for words below minCount,
    # so these OOV targets become usable anchor endpoints.
    all_en_targets: list[str] = []
    for _lang in LANGUAGES:
        _lp = get_seed_lexicon_path(_lang)
        if _lp and _lp.exists():
            all_en_targets.extend(tw for _, tw in load_lexicon(str(_lp)))

    en_words, en_matrix = supplement_with_oov(
        get_embeddings_path(ENGLISH), en_words_orig, en_matrix_orig, all_en_targets
    )
    n_en_added = len(en_words) - n_en_orig
    if n_en_added:
        print(f"English vocab supplemented: {n_en_orig} → {len(en_words)} (+{n_en_added} OOV from lexicons)")

    # Train / load English NER model once (cached at outputs/ner/)
    ner_ckpt = Path(OUTPUTS_ROOT) / "ner" / "bilstm_crf_conll2003.pt"
    try:
        ner_model, _tag_to_idx, idx_to_tag = train_or_load_english_ner_model(
            conll_root=get_conll2003_root(),
            english_fasttext_bin=get_embeddings_path(ENGLISH),
            out_path=ner_ckpt,
            embedding_dim=EMBEDDING_DIM,
            epochs=5,
            batch_size=16,
            seed=42,
            force_retrain=False,
            device="cpu",
        )
        allowed_types = _entity_types_from_tagset(idx_to_tag)
        print(f"[NER] model ready — tag types: {allowed_types}")
    except FileNotFoundError as e:
        print(f"[RQ1] NER model unavailable ({e}); NER_F1 will be skipped.")
        ner_model = None
        idx_to_tag = None
        allowed_types = []

    for lang in LANGUAGES:
        display_name = get_display_name(lang)
        print(f"\n{'='*50}\nLanguage: {display_name} ({lang})\n{'='*50}")

        # Load lexicon FIRST so we know which source words to supplement.
        lexicon_path = get_seed_lexicon_path(lang)
        if lexicon_path is None or not lexicon_path.exists():
            print(f"Missing bilingual seed lexicon for {display_name} ({lang}): {lexicon_path}")
            print("Expected a tab-separated file with one source-target pair per line.")
            continue

        all_pairs = load_lexicon(str(lexicon_path))

        src_words_orig, src_matrix_orig = get_or_train_embeddings(lang)
        if src_words_orig is None:
            continue

        n_src_orig = len(src_words_orig)
        src_idx_orig = {w: i for i, w in enumerate(src_words_orig)}

        # Supplement source with lexicon source words (OOV via subword composition).
        src_words, src_matrix = supplement_with_oov(
            get_embeddings_path(lang), src_words_orig, src_matrix_orig,
            [sw for sw, _ in all_pairs]
        )
        n_src_added = len(src_words) - n_src_orig
        if n_src_added:
            print(f"  {display_name} vocab supplemented: {n_src_orig} → {len(src_words)} (+{n_src_added} OOV)")

        # CKA before alignment: compare only original vocabulary rows
        # (appended OOV rows are irrelevant to the pre-alignment geometry).
        n_cka = min(n_src_orig, n_en_orig)
        cka_before = linear_cka(src_matrix_orig[:n_cka], en_matrix_orig[:n_cka])
        print(f"CKA before alignment: {cka_before:.4f}")

        train_pairs, test_pairs = split_lexicon(all_pairs, held_out=HELD_OUT_PAIRS)

        # Build anchor matrices from the FULL (supplemented) vocabularies.
        X_src, X_tgt = build_anchor_matrices(
            train_pairs, src_words, src_matrix, en_words, en_matrix
        )
        print(f"Anchor pairs: {len(X_src)}  (vocab-only would have been fewer)")

        if len(X_src) < 2:
            print("Not enough anchor pairs for alignment; skipping this language.")
            continue

        src_txt_path = get_text_embeddings_path(lang)
        en_txt_path = get_text_embeddings_path(ENGLISH)
        vecmap_ready = src_txt_path.exists() and en_txt_path.exists()

        # Load NER test sentences once per language
        ner_sents = None
        if ner_model is not None:
            ner_dir = get_ner_path(lang)
            if ner_dir is not None and Path(ner_dir).exists():
                try:
                    ner_sents = load_masakhaner_split(Path(ner_dir), "test")
                    print(f"  NER test sentences: {len(ner_sents)}")
                except Exception as e:
                    print(f"  [NER] Could not load test data for {lang}: {e}")

        def make_embedder(lookup, dim):
            def embedder(tokens):
                zero = np.zeros(dim, dtype=np.float32)
                vecs = []
                for t in tokens:
                    v = lookup.get(t)
                    if v is None:
                        v = lookup.get(t.lower())
                    if v is not None:
                        v = v.astype(np.float32)
                        norm = np.linalg.norm(v)
                        v = v / (norm + 1e-8) if norm > 1e-8 else v
                    else:
                        v = zero
                    vecs.append(v)
                return np.stack(vecs)
            return embedder

        for method in ["CCA", "KCCA", "VecMap"]:
            print(f"\n  -> Method: {method}")

            if method == "CCA":
                # K/p >= 10 rule: cap components so the system stays well-determined
                n_components = min(len(X_src) // 10, 100, src_matrix.shape[1], en_matrix.shape[1])
                n_components = max(n_components, 2)
                aligner = CCAAligner(n_components=n_components)
                aligner.fit(X_src, X_tgt)

                # Evaluation: transform ORIGINAL vocabulary only (fair comparison).
                src_aligned = aligner.transform_src(src_matrix_orig)
                tgt_aligned = aligner.transform_tgt(en_matrix_orig)

                # NER embedder: project FULL vocab (includes OOV supplement)
                # so rare NER tokens still get an aligned vector.
                src_aligned_full = aligner.transform_src(src_matrix)
                tgt_aligned_full = aligner.transform_tgt(en_matrix)
                R = np.linalg.lstsq(tgt_aligned_full, en_matrix, rcond=None)[0].astype(np.float32)
                src_aligned_en = (src_aligned_full @ R)
                norms = np.linalg.norm(src_aligned_en, axis=1, keepdims=True)
                src_aligned_en = (src_aligned_en / np.maximum(norms, 1e-8)).astype(np.float32)
                src_lookup = {w: src_aligned_en[i] for i, w in enumerate(src_words)}
                embedder = make_embedder(src_lookup, en_matrix_orig.shape[1])

            elif method == "KCCA":
                max_anchors = min(1000, len(X_src))
                aligner = KCCAAligner(n_components=min(50, max_anchors), gamma=0.1, reg=1e-3)
                aligner.fit(X_src[:max_anchors], X_tgt[:max_anchors])

                # Evaluation: original vocabulary only.
                src_aligned = aligner.transform_src(src_matrix_orig)
                tgt_aligned = aligner.transform_tgt(en_matrix_orig)

                # NER embedder: full vocab projection.
                src_aligned_full = aligner.transform_src(src_matrix)
                Z_tgt_reg = aligner.transform_tgt(en_matrix_orig[:max_anchors])
                R = np.linalg.lstsq(Z_tgt_reg, en_matrix_orig[:max_anchors], rcond=None)[0].astype(np.float32)
                src_aligned_en = (src_aligned_full @ R)
                norms = np.linalg.norm(src_aligned_en, axis=1, keepdims=True)
                src_aligned_en = (src_aligned_en / np.maximum(norms, 1e-8)).astype(np.float32)
                src_lookup = {w: src_aligned_en[i] for i, w in enumerate(src_words)}
                embedder = make_embedder(src_lookup, en_matrix_orig.shape[1])

            elif method == "VecMap":
                if not vecmap_ready:
                    print(f"    Skipping VecMap (missing {src_txt_path} or {en_txt_path})")
                    continue
                aligned_src_dict, aligned_tgt_dict = run_vecmap(
                    str(src_txt_path),
                    str(en_txt_path),
                    str(lexicon_path),
                    output_dir=str(PROJECT_ROOT / "outputs" / f"vecmap_{lang}")
                )
                # VecMap operates on the original .txt vocabulary; OOV words fall back
                # to their unaligned vectors.
                src_aligned = np.array([
                    aligned_src_dict.get(w, src_matrix_orig[i])
                    for i, w in enumerate(src_words_orig)
                ])
                tgt_aligned = np.array([
                    aligned_tgt_dict.get(w, en_matrix_orig[i])
                    for i, w in enumerate(en_words_orig)
                ])

                def make_vecmap_embedder(adict, dim):
                    def embedder(tokens):
                        zero = np.zeros(dim, dtype=np.float32)
                        vecs = []
                        for tok in tokens:
                            v = adict.get(tok)
                            if v is None:
                                v = adict.get(tok.lower())
                            if v is not None:
                                v = v.astype(np.float32)
                                v = v / (np.linalg.norm(v) + 1e-8)
                            else:
                                v = zero
                            vecs.append(v)
                        return np.stack(vecs)
                    return embedder
                embedder = make_vecmap_embedder(aligned_src_dict, en_matrix_orig.shape[1])

            # Intrinsic evaluation on ORIGINAL vocabulary (ensures fair cross-method comparison).
            p1 = precision_at_k(src_aligned, tgt_aligned, test_pairs, src_words_orig, en_words_orig, k=1)
            p5 = precision_at_k(src_aligned, tgt_aligned, test_pairs, src_words_orig, en_words_orig, k=5)
            mcs = mean_cosine_similarity(src_aligned, tgt_aligned, test_pairs, src_idx_orig, en_idx_orig)
            cka_after = linear_cka(src_aligned[:n_cka], tgt_aligned[:n_cka])

            # Extrinsic NER evaluation
            ner_f1 = None
            if ner_model is not None and ner_sents is not None:
                try:
                    metrics = evaluate_sentences(
                        ner_model, idx_to_tag, ner_sents,
                        embedder=embedder,
                        device="cpu",
                        allowed_entity_types=allowed_types,
                        unknown_type_strategy="to_misc",
                    )
                    ner_f1 = round(metrics["f1"], 4)
                    print(f"    P@1={p1:.3f}  P@5={p5:.3f}  MCS={mcs:.3f}  CKA_after={cka_after:.3f}  NER_F1={ner_f1:.4f}")
                except Exception as e:
                    print(f"    [NER] evaluation failed: {e}")
                    print(f"    P@1={p1:.3f}  P@5={p5:.3f}  MCS={mcs:.3f}  CKA_after={cka_after:.3f}")
            else:
                print(f"    P@1={p1:.3f}  P@5={p5:.3f}  MCS={mcs:.3f}  CKA_after={cka_after:.3f}")

            results.append({
                "language": lang,
                "method": method,
                "CKA_before": round(cka_before, 4),
                "CKA_after": round(cka_after, 4),
                "P@1": round(p1, 3),
                "P@5": round(p5, 3),
                "MCS": round(mcs, 3),
                "NER_F1": ner_f1 if ner_f1 is not None else "",
            })

    if results:
        df = pd.DataFrame(results)
        df.to_csv(results_dir / "rq1_results.csv", index=False)
        print("\n\n", df.to_string(index=False))

if __name__ == "__main__":
    run_rq1()
