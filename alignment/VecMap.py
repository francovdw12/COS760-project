# VecMap alignment (RQ1).
# Wraps the bundled VecMap tool (vecmap/vecmap-master) and runs it in
# supervised mode using the bilingual seed lexicon.
import os
import sys
import subprocess
import tempfile
import numpy as np


def load_txt_embeddings(path):
    """Read a word2vec-format text embeddings file into a {word: vector} dict."""
    vectors = {}
    with open(path, encoding="utf-8") as f:
        next(f)  # skip header line "<vocab> <dim>"
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < 3:
                continue
            word = parts[0]
            vectors[word] = np.array(parts[1:], dtype=np.float32)
    return vectors


def _write_vecmap_lexicon(lexicon_path, out_path):
    """Write a VecMap-compatible lexicon: one space-separated pair per line.

    VecMap does `src, trg = line.split()` so it needs exactly 2 tokens.
    We skip multi-word entries and normalise tabs to spaces.
    """
    kept = 0
    with open(lexicon_path, encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            # Accept tab-separated or space-separated
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) != 2:
                continue
            src, tgt = parts
            # VecMap requires single-token entries on each side
            if " " in src or " " in tgt:
                continue
            fout.write(f"{src} {tgt}\n")
            kept += 1
    return kept


def run_vecmap(src_emb_path, tgt_emb_path, lexicon_path, output_dir):
    """Run VecMap in supervised mode via subprocess.

    Returns (src_aligned_dict, tgt_aligned_dict).
    """
    os.makedirs(output_dir, exist_ok=True)
    src_out = os.path.join(output_dir, "src_aligned.txt")
    tgt_out = os.path.join(output_dir, "tgt_aligned.txt")

    vecmap_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "vecmap", "vecmap-master")
    )
    map_script = os.path.join(vecmap_root, "map_embeddings.py")

    # Write a cleaned lexicon that VecMap can parse (exactly 2 tokens per line)
    clean_lex = os.path.join(output_dir, "lexicon_clean.txt")
    kept = _write_vecmap_lexicon(lexicon_path, clean_lex)
    if kept == 0:
        raise ValueError(f"No valid single-word pairs found in lexicon: {lexicon_path}")

    cmd = [
        sys.executable, map_script,
        "--supervised", clean_lex,
        src_emb_path, tgt_emb_path,
        src_out, tgt_out,
    ]
    subprocess.run(cmd, check=True)

    return load_txt_embeddings(src_out), load_txt_embeddings(tgt_out)
