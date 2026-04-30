
import subprocess
import numpy as np
import os

def run_vecmap(src_emb_path: str, tgt_emb_path: str,
               lexicon_path: str, output_dir: str) -> np.ndarray:
    """
    Run VecMap in supervised mode via subprocess.
    Return the aligned source matrix.
    """
    os.makedirs(output_dir, exist_ok=True)
    src_out = os.path.join(output_dir, "src_aligned.txt")
    tgt_out = os.path.join(output_dir, "tgt_aligned.txt")

    vecmap_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "vecmap", "vecmap-master")
    )
    map_script = os.path.join(vecmap_root, "map_embeddings.py")

    cmd = [
        "python", map_script,
        "--supervised", lexicon_path,
        src_emb_path, tgt_emb_path,
        src_out, tgt_out,
    ]
    subprocess.run(cmd, check=True)

    # Reload the aligned matrix
    aligned = load_txt_embeddings(src_out)
    return aligned

def load_txt_embeddings(path: str):
    """Read a text word2vec-format embeddings file."""
    vectors = {}
    with open(path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            vectors[word] = vec
    return vectors