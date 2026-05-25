# COS760 — Cross-lingual Embedding Alignment

Cross-lingual embedding alignment experiments for isiZulu (conjunctive), Sepedi,
and Setswana (disjunctive) — Group 40.

## Repository layout

```text
COS760-project/
├── config.py                # Centralised paths and hyperparameters
├── embeddings.py            # FastText training and loading helpers
├── evaluation.py            # P@k, mean cosine similarity, CKA
├── lexicon.py               # Bilingual lexicon loading and anchor building
├── run_rq1.py               # RQ1 pipeline (alignment quality)
├── run_rq2.py               # RQ2 pipeline (data efficiency / learning curves)
├── visualize_rq1.py         # RQ1 figures (P@k, CKA, NER F1, radar)
├── visualize_rq2.py         # RQ2 figures (learning curves, break-even, heatmap)
├── alignment/
│   ├── CCA.py
│   ├── KCCA.py
│   └── VecMap.py
├── transfer/
│   ├── corpus_subsets.py    # Deterministic NCHLT subset builder (RQ2)
│   └── zero_shot_eval.py    # BiLSTM-CRF NER + zero-shot evaluation
├── vecmap/vecmap-master/    # Bundled VecMap tool
├── data/                    # All datasets (see section below)
├── embeddings/              # FastText .bin/.txt models (generated)
├── outputs/                 # NER checkpoints, VecMap outputs (generated)
├── results/                 # CSVs and PNGs (generated)
├── Dockerfile               # CPU-only Docker image
├── docker-compose.yml       # Convenience compose file
├── docker-entrypoint.sh     # Container command dispatcher
└── requirements.txt         # Python dependencies
```

## Data layout

All datasets live under `data/`.  The canonical NCHLT directory is
`data/NCHLT Text Corpora/` (with a space); the directory
`data/NCHLT-Text-Corpora/` (with a dash) is an unused duplicate and can be
safely deleted.

```text
data/
├── NCHLT Text Corpora/
│   ├── en/corpora/1_Corpus_nchlt/CORP.NCHLT.eng.CLEAN.1.0.0.txt
│   ├── nso/2.Corpora/CORP.NCHLT.nso.CLEAN.2.0.txt
│   ├── tn/2.Corpora/CORP.NCHLT.tn.CLEAN.2.0.txt
│   └── zu/2.Corpora/CORP.NCHLT.zu.CLEAN.2.0.txt
├── Bilingual Seed Lexicons/
│   ├── zul_en.txt
│   ├── nso_en.txt
│   └── tsn_en.txt
├── ner_MasakhaNER 2.0/masakhaner2/
│   ├── zul/  (train.txt  dev.txt  test.txt)
│   ├── nso/  (train.txt  dev.txt  test.txt)
│   └── tsn/  (train.txt  dev.txt  test.txt)
└── conll2003/
    ├── train.txt
    ├── dev.txt
    └── test.txt
```

All required files are present in the current workspace.

## Running locally

### Setup

```bash
# Recreate the virtual environment (Python 3.11 or 3.13 recommended)
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Install CPU PyTorch first (avoids pulling the large CUDA wheel)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### RQ1 — Alignment quality

```bash
source .venv/bin/activate
python run_rq1.py
python visualize_rq1.py
```

Outputs:
- `results/rq1_results.csv` — P@1, P@5, MCS, CKA, NER F1 per language/method
- `results/rq1_*.png` — six figures

### RQ2 — Data efficiency / learning curves

```bash
source .venv/bin/activate
python run_rq2.py
python visualize_rq2.py
```

Useful options for `run_rq2.py`:

```bash
python run_rq2.py --langs zul tsn --fractions 1.0 0.25 0.05 --methods CCA KCCA VecMap --split test
python run_rq2.py --force   # retrain embeddings/NER and rebuild cached alignment artifacts
```

Outputs:
- `results/rq2_results.csv` — precision/recall/F1 per (lang, fraction, method)
- `results/rq2_learning_curves.png` — F1 vs corpus size per language
- `results/rq2_breakeven_table.png` — min tokens to reach F1 ≥ 0.50
- `results/rq2_conjunctive_vs_disjunctive.png` — isiZulu vs Sepedi+Setswana
- `results/rq2_method_heatmap.png` — F1 heatmap across all (method, lang, fraction)
- `outputs/ner/bilstm_crf_conll2003.pt` — English BiLSTM-CRF checkpoint
- `embeddings/aligned/` — cached CCA/KCCA alignment artifacts per fraction
- `outputs/vecmap_*` — VecMap alignment outputs

## Running via Docker (recommended for reproducibility)

The Docker image is CPU-only and bakes all required datasets into the image so
no extra setup is needed on a fresh machine.

### Build

```bash
docker compose build
```

The first build takes ~10 minutes (downloads PyTorch CPU wheel and compiles
fasttext-wheel).

### Run individual stages

```bash
docker compose run --rm cos760 rq1
docker compose run --rm cos760 rq2
docker compose run --rm cos760 viz1
docker compose run --rm cos760 viz2
```

### Run the full pipeline in one command

```bash
docker compose run --rm cos760 all
```

Generated files (`results/`, `outputs/`, `embeddings/`) are bind-mounted from
the host via `docker-compose.yml`, so all artifacts appear in the project
directory after the container finishes.

### Advanced options

```bash
# Run RQ2 on a specific language and fraction subset
docker compose run --rm cos760 rq2 --langs tsn --fractions 0.25 0.05

# Force retrain (ignore cached checkpoints and alignment artifacts)
docker compose run --rm cos760 rq2 --force

# Open an interactive shell inside the container
docker compose run --rm cos760 shell
```

## Quick git reference

| Action | Command |
| :--- | :--- |
| Clone | `git clone <url>` |
| Update | `git pull origin main` |
| New branch | `git checkout -b <branch>` |
| Stage | `git add .` |
| Commit | `git commit -m "message"` |
| Push | `git push origin <branch>` |
