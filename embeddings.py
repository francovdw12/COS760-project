# embeddings.py
import fasttext
import numpy as np

def train_fasttext(corpus_path: str, output_path: str, dim: int = 100):
    """Train FastText with subwords (important for conjunctive isiZulu)."""
    model = fasttext.train_unsupervised(
        corpus_path,
        model='skipgram',
        dim=dim,
        minCount=5,
        wordNgrams=1,
        minn=3,      # subword min - helps agglutinative morphology
        maxn=6,      # subword max
        epoch=10,
        thread=4,
    )
    model.save_model(output_path)
    return model


def save_embeddings_as_txt(model_path: str, output_path: str, vocab_limit: int = 50000):
    """Export a FastText model to word2vec text format for VecMap."""
    model = fasttext.load_model(model_path)
    words = model.words[:vocab_limit]
    if not words:
        raise ValueError(f"No words found in FastText model: {model_path}")

    dim = len(model.get_word_vector(words[0]))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{len(words)} {dim}\n")
        for word in words:
            vector = model.get_word_vector(word)
            vector_text = " ".join(f"{value:.6f}" for value in vector)
            f.write(f"{word} {vector_text}\n")

def load_embeddings_as_matrix(model_path: str, vocab_limit: int = 50000):
    """
    Load a FastText model and return:
    - words: list of tokens
    - matrix: np.array (vocab_limit, dim) L2-normalized
    """
    model = fasttext.load_model(model_path)
    words = model.words[:vocab_limit]
    matrix = np.array([model.get_word_vector(w) for w in words])
    # L2 normalization is required before alignment
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-8)
    return words, matrix