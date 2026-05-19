import numpy as np
import pandas as pd
from pathlib import Path
from embeddings import load_embeddings_as_matrix, save_embeddings_as_txt, train_fasttext
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
    en_words, en_matrix = get_or_train_embeddings(ENGLISH)
    if en_words is None:
        print("Failed to get English embeddings. Stopping.")
        return

    en_idx = {w: i for i, w in enumerate(en_words)}

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

        lexicon_path = get_seed_lexicon_path(lang)
        if lexicon_path is None or not lexicon_path.exists():
            print(f"Missing bilingual seed lexicon for {display_name} ({lang}): {lexicon_path}")
            print("Expected a tab-separated file with one source-target pair per line.")
            continue

        src_words, src_matrix = get_or_train_embeddings(lang)
        if src_words is None:
            continue

        src_idx = {w: i for i, w in enumerate(src_words)}

        n_cka = min(len(src_words), len(en_words))
        cka_before = linear_cka(src_matrix[:n_cka], en_matrix[:n_cka])
        print(f"CKA before alignment: {cka_before:.4f}")

        all_pairs = load_lexicon(str(lexicon_path))
        train_pairs, test_pairs = split_lexicon(all_pairs, held_out=HELD_OUT_PAIRS)

        X_src, X_tgt = build_anchor_matrices(
            train_pairs, src_words, src_matrix, en_words, en_matrix
        )
        print(f"Anchor pairs: {len(X_src)}")

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
                return np.stack([lookup.get(t, zero) for t in tokens])
            return embedder

        for method in ["CCA", "KCCA", "VecMap"]:
            print(f"\n  -> Method: {method}")

            if method == "CCA":
                n_components = min(100, len(X_src), src_matrix.shape[1], en_matrix.shape[1])
                aligner = CCAAligner(n_components=n_components)
                aligner.fit(X_src, X_tgt)
                src_aligned = aligner.transform_src(src_matrix)
                tgt_aligned = aligner.transform_tgt(en_matrix)
                # Regression back to 100-dim English space for NER
                Z_tgt = aligner.transform_tgt(en_matrix)
                R = np.linalg.lstsq(Z_tgt, en_matrix, rcond=None)[0].astype(np.float32)
                src_aligned_en = (aligner.transform_src(src_matrix) @ R)
                norms = np.linalg.norm(src_aligned_en, axis=1, keepdims=True)
                src_aligned_en = (src_aligned_en / np.maximum(norms, 1e-8)).astype(np.float32)
                src_lookup = {w: src_aligned_en[i] for i, w in enumerate(src_words)}
                embedder = make_embedder(src_lookup, en_matrix.shape[1])

            elif method == "KCCA":
                max_anchors = min(1000, len(X_src))
                aligner = KCCAAligner(n_components=min(50, max_anchors), gamma=0.1, reg=1e-3)
                aligner.fit(X_src[:max_anchors], X_tgt[:max_anchors])
                src_aligned = aligner.transform_src(src_matrix)
                tgt_aligned = aligner.transform_tgt(en_matrix)
                Z_tgt = aligner.transform_tgt(en_matrix[:max_anchors])
                R = np.linalg.lstsq(Z_tgt, en_matrix[:max_anchors], rcond=None)[0].astype(np.float32)
                src_aligned_en = (aligner.transform_src(src_matrix) @ R)
                norms = np.linalg.norm(src_aligned_en, axis=1, keepdims=True)
                src_aligned_en = (src_aligned_en / np.maximum(norms, 1e-8)).astype(np.float32)
                src_lookup = {w: src_aligned_en[i] for i, w in enumerate(src_words)}
                embedder = make_embedder(src_lookup, en_matrix.shape[1])

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
                src_aligned = np.array([
                    aligned_src_dict.get(w, src_matrix[i])
                    for i, w in enumerate(src_words)
                ])
                tgt_aligned = np.array([
                    aligned_tgt_dict.get(w, en_matrix[i])
                    for i, w in enumerate(en_words)
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
                embedder = make_vecmap_embedder(aligned_src_dict, en_matrix.shape[1])

            p1 = precision_at_k(src_aligned, tgt_aligned, test_pairs, src_words, en_words, k=1)
            p5 = precision_at_k(src_aligned, tgt_aligned, test_pairs, src_words, en_words, k=5)
            mcs = mean_cosine_similarity(src_aligned, tgt_aligned, test_pairs, src_idx, en_idx)
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
