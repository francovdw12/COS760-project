import numpy as np
import pandas as pd
from pathlib import Path
from embeddings import load_embeddings_as_matrix, save_embeddings_as_txt, train_fasttext
from lexicon import load_lexicon, split_lexicon, build_anchor_matrices
from alignment.CCA import CCAAligner
from alignment.KCCA import KCCAAligner
from alignment.VecMap import run_vecmap
from evaluation import precision_at_k, mean_cosine_similarity, linear_cka
from config import (
    ENGLISH,
    HELD_OUT_PAIRS,
    LANGUAGES,
    PROJECT_ROOT,
    RESULTS_ROOT,
    get_corpus_path,
    get_display_name,
    get_embeddings_path,
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

        n_sample = min(2000, len(src_words), len(en_words))
        cka_before = linear_cka(src_matrix[:n_sample], en_matrix[:n_sample])
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

        for method in ["CCA", "KCCA", "VecMap"]:
            print(f"\n  -> Method: {method}")

            if method == "CCA":
                n_components = min(100, len(X_src), src_matrix.shape[1], en_matrix.shape[1])
                aligner = CCAAligner(n_components=n_components)
                aligner.fit(X_src, X_tgt)
                src_aligned = aligner.transform_src(src_matrix)
                tgt_aligned = aligner.transform_tgt(en_matrix)

            elif method == "KCCA":
                max_anchors = min(1000, len(X_src))
                aligner = KCCAAligner(n_components=min(50, max_anchors), gamma=0.1, reg=1e-3)
                aligner.fit(X_src[:max_anchors], X_tgt[:max_anchors])
                src_aligned = aligner.transform_src(src_matrix[:5000])
                tgt_aligned = aligner.transform_tgt(en_matrix[:5000])

            elif method == "VecMap":
                if not vecmap_ready:
                    print(f"    Skipping VecMap (missing {src_txt_path} or {en_txt_path})")
                    continue
                aligned_dict = run_vecmap(
                    str(src_txt_path),
                    str(en_txt_path),
                    str(lexicon_path),
                    output_dir=str(PROJECT_ROOT / "outputs" / f"vecmap_{lang}")
                )
                src_aligned = np.array([
                    aligned_dict.get(w, src_matrix[i])
                    for i, w in enumerate(src_words[:5000])
                ])
                tgt_aligned = en_matrix[:5000]

            p1 = precision_at_k(src_aligned, tgt_aligned, test_pairs, src_words, en_words, k=1)
            p5 = precision_at_k(src_aligned, tgt_aligned, test_pairs, src_words, en_words, k=5)
            mcs = mean_cosine_similarity(src_aligned, tgt_aligned, test_pairs, src_idx, en_idx)
            cka_after = linear_cka(src_aligned[:n_sample], tgt_aligned[:n_sample])

            print(f"    P@1={p1:.3f}  P@5={p5:.3f}  MCS={mcs:.3f}  CKA_after={cka_after:.3f}")

            results.append({
                "language": lang,
                "method": method,
                "CKA_before": round(cka_before, 4),
                "CKA_after": round(cka_after, 4),
                "P@1": round(p1, 3),
                "P@5": round(p5, 3),
                "MCS": round(mcs, 3),
            })

    if results:
        df = pd.DataFrame(results)
        df.to_csv(results_dir / "rq1_results.csv", index=False)
        print("\n\n", df.to_string(index=False))

if __name__ == "__main__":
    run_rq1()
