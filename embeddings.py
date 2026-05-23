# embeddings.py - FastText training and loading helpers.
import fasttext
import numpy as np


def supplement_with_oov(model_path, words, matrix, extra_words):
    """Extend (words, matrix) with FastText subword vectors for OOV words.

    FastText computes a vector for any string via character n-gram composition,
    even if the token was below minCount.  This lets lexicon words that missed
    the vocabulary threshold still contribute anchor pairs.

    OOV words are appended at the END so that existing row indices are preserved.
    """
    vocab_set = set(words)
    missing = list(dict.fromkeys(w for w in extra_words if w not in vocab_set))
    if not missing:
        return list(words), matrix
    model = fasttext.load_model(str(model_path))
    vecs = np.array([model.get_word_vector(w) for w in missing], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= np.maximum(norms, 1e-8)
    return list(words) + missing, np.vstack([matrix, vecs]).astype(np.float32)


def train_fasttext(corpus_path, output_path, dim=100):
    """Train FastText with subwords (important for conjunctive isiZulu)."""
    model = fasttext.train_unsupervised(
        corpus_path,
        model="skipgram",
        dim=dim,
        minCount=2,  # was 5 — small corpora lose too many lexicon words at 5
        wordNgrams=1,
        minn=3,      # subword min - helps agglutinative morphology
        maxn=6,      # subword max
        epoch=20,
        thread=4,
    )
    model.save_model(output_path)
    return model


def save_embeddings_as_txt(model_path, output_path):
    """Export a FastText model to word2vec text format for VecMap."""
    model = fasttext.load_model(model_path)
    words = model.words
    if not words:
        raise ValueError(f"No words found in FastText model: {model_path}")

    dim = len(model.get_word_vector(words[0]))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{len(words)} {dim}\n")
        for word in words:
            vector = model.get_word_vector(word)
            vector_text = " ".join(f"{value:.6f}" for value in vector)
            f.write(f"{word} {vector_text}\n")


def load_embeddings_as_matrix(model_path):
    """Load a FastText model and return (words, matrix).

    words: list of tokens.
    matrix: np.array (vocab, dim), L2-normalised (required before alignment).
    """
    model = fasttext.load_model(model_path)
    words = model.words
    matrix = np.array([model.get_word_vector(w) for w in words])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-8)
    return words, matrix