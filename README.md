# COS760 — Cross-Lingual Embedding Alignment for isiZulu, Sepedi & Setswana

Optimising cross-lingual word embeddings for three South African Bantu
languages (isiZulu, Sepedi, Setswana) aligned onto an English pivot space.

The project answers two research questions:

- **RQ1 — Alignment strategy.** Does **CCA** outperform **VecMap** and
  **Kernel CCA (KCCA)** when aligning each language onto English, and does the
  relative advantage of non-linear methods differ by morphology (conjunctive
  isiZulu vs. disjunctive Sepedi/Setswana)?
- **RQ2 — Data efficiency.** How does monolingual corpus size affect zero-shot 
    NER transfer quality across alignment methods, and is there is systematic
    difference in data-efficiency profiles between conjunctive and disjunctive
    morphology?
    
---

## Repository layout

```text
COS760-project/
├── config.py                 # Centralised paths, language codes, hyperparameters
├── embeddings.py             # FastText training / loading / OOV subword inference
├── lexicon.py                # Bilingual lexicon parsing + anchor-matrix building
├── evaluation.py             # Intrinsic metrics: P@k, mean cosine similarity, linear CKA
├── run_rq1.py                # RQ1 orchestrator (alignment quality)  ← fully commented
├── run_rq2.py                # RQ2 orchestrator (data-efficiency learning curves)
├── visualize_rq1.py          # Generates the 6 RQ1 figures from rq1_results.csv
├── visualize_rq2.py          # Generates per-method F1 and BLI p@1 learning curve plots
├── 760.sh                    # Bootstrap: create .venv and install requirements
├── 760.ps1                   # Bootstrap: create .venv and install requirements -- PowerShell
├── requirements.txt
│
├── alignment/
│   ├── CCA.py                # Linear CCA (sklearn) — correlation-maximising projection
│   ├── KCCA.py               # Kernel CCA (RBF) — non-linear alignment, from scratch
│   └── VecMap.py             # Wrapper around the bundled VecMap tool (orthogonal mapping)
│
├── rq2/                      # RQ2 pipeline
│   ├── __init__.py
│   ├── aligners/
│   │   ├── __init__.py
│   │   ├── base.py             # AlignerBase abstract base class (interface contract for aligners)
│   │   ├── cca.py              # CCAWrapper — linear alignment
│   │   ├── kcca.py             # KCCAWrapper — kernel (non-linear) alignment
│   │   └── vecmap.py           # VecMapWrapper — orthogonal mapping via subprocess
│   ├── diagnostics/
│   │   ├── __init__.py
│   │   ├── bli.py              # Bilingual Lexicon Induction precision@1
│   │   └── cka.py              # Centered Kernel Alignment
│   └── pipeline/
│       ├── __init__.py
│       ├── config.py            # RQ2Config dataclass — centralized hyperparameter data object
│       ├── fasttext_stage.py    # FastText training and embedding loading
│       ├── ner_stage.py         # English NER model training, validation, tag utilities
│       ├── alignment_stage.py   # Aligner factory, fit, and diagnostic layer
│       ├── evaluation_stage.py  # Zero-shot NER evaluation and result row assembly
│       └── experiment_runner.py # Three-stage pipeline loop; called by CLI entry point
|
|
├── transfer/
│   ├── corpus_subsets.py     # Deterministic NCHLT corpus subsets for RQ2
│   └── zero_shot_eval.py     # English BiLSTM-CRF training + zero-shot NER evaluation
│
├── vecmap/vecmap-master/     # Bundled VecMap (Artetxe et al., 2018), called as a subprocess
│
├── data/                     # Datasets 
│   ├── NCHLT Text Corpora/   # Monolingual corpora: en/, zu/, nso/, tn/
│   ├── Bilingual Seed Lexicons/   # zul_en.txt, nso_en.txt, tsn_en.txt
│   ├── ner_MasakhaNER 2.0/masakhaner2/  # zul/, nso/, tsn/ (train/dev/test, IOB2)
│   ├── conll2003/            # English NER training data (train/dev/test)
│   └── subsets/              # RQ2 fractional corpora (generated automatically)
│
├── embeddings/               # Trained models: {lang}.bin / {lang}.txt; aligned/ cache
├── outputs/                  # NER checkpoint + per-language VecMap outputs
└── results/
    ├── rq1_results.csv         # RQ1: 9 rows (3 languages x 3 methods)
    ├── rq1_*.png               # RQ1 figures
    └── rq2/
        └── {run_name}/
            ├── config.json     # Hyperparameters for this run (serialised automatically)
            └── results.csv     # RQ2: rows per language x fraction x method
```

---

## Setup

The bootstrap script creates a virtual environment and installs everything.
It looks for `python3.11` / `python3.12` / `python3` (in that order).

```bash
cd COS760-project
./760.sh
```

```PowerShell
cd COS760-project
./760.ps1
```

Or manually:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Dependencies** (`requirements.txt`): numpy, pandas, scikit-learn, scipy,
`fasttext-wheel`, matplotlib, `seaborn>=0.13`, torch, seqeval, TorchCRF.

---

## Datasets

Datasets live under `data/`. The expected
layout is:

```text
data/
├── NCHLT Text Corpora/
│   ├── en/corpora/1_Corpus_nchlt/CORP.NCHLT.eng.CLEAN.1.0.0.txt
│   ├── zu/2.Corpora/CORP.NCHLT.zu.CLEAN.2.0.txt
│   ├── nso/2.Corpora/CORP.NCHLT.nso.CLEAN.2.0.txt
│   └── tn/2.Corpora/CORP.NCHLT.tn.CLEAN.2.0.txt
├── Bilingual Seed Lexicons/
│   ├── zul_en.txt            # ~8.2k entries (TAB-separated source<TAB>english)
│   ├── nso_en.txt            # ~9.2k entries
│   └── tsn_en.txt            # ~9.1k entries
├── ner_MasakhaNER 2.0/masakhaner2/
│   ├── zul/  nso/  tsn/      # each: train.txt / dev.txt / test.txt (IOB2)
└── conll2003/
    ├── train.txt  dev.txt  test.txt
```

Exact paths are defined in [`config.py`](config.py); change them there if your
filenames differ.

### Embedding training parameters

100-dimensional **FastText** (`skipgram`) embeddings, chosen over Word2Vec
because character n-grams produce vectors for unseen tokens — important for
isiZulu's conjunctive morphology. Key settings (`embeddings.py`):

| Parameter | Value | Why |
| :--- | :--- | :--- |
| `dim` | 100 | Project-wide embedding dimension |
| `minCount` | 2 | Low threshold keeps rare conjunctive forms in vocab |
| `minn` / `maxn` | 3 / 6 | Subword n-gram range (helps agglutinative morphology) |
| `epoch` | 10 | |

---

## Running RQ1 — alignment quality

```bash
source .venv/bin/activate
python run_rq1.py        # writes results/rq1_results.csv
python visualize_rq1.py  # generates the 6 PNG figures
```

### What `run_rq1.py` does (six phases)

1. **English embeddings** — load or train FastText for English (cached in
   `embeddings/eng.bin`), then *supplement* the vocabulary with every English
   target word from all three lexicons via FastText subword inference.
2. **English NER model** — train (or load) a BiLSTM-CRF on CoNLL-2003 once,
   cached at `outputs/ner/bilstm_crf_conll2003.pt`.
3. **Per language** (`zul`, `nso`, `tsn`) — load the bilingual lexicon, load/
   train source embeddings, supplement source vocab with OOV lexicon words,
   compute pre-alignment CKA, split the lexicon 80/20, and build anchor matrices.
4. **Per method** (`CCA`, `KCCA`, `VecMap`) — align the source space onto
   English and build a NER `embedder` that maps Bantu tokens into the English
   embedding space.
5. **Evaluation** — intrinsic (P@1, P@5, MCS, CKA-after) on 1 000 held-out
   pairs **using the original vocabulary only** for a fair comparison, plus
   extrinsic zero-shot NER F1 on MasakhaNER.
6. **Export** — aggregate the 3 × 3 grid into `results/rq1_results.csv`.

### Metrics

- **P@1 / P@5** — word-translation accuracy: is the true English translation
  among the *k* nearest neighbours of the aligned source word?
- **MCS** — mean cosine similarity between aligned translation pairs.
- **CKA (before/after)** — linear Centered Kernel Alignment; global geometric
  similarity of the two spaces, invariant to rotation and isotropic scaling.
- **NER F1** — entity-level F1 of the English BiLSTM-CRF applied zero-shot to
  MasakhaNER using each method's aligned embeddings.

### Alignment methods

| Method | Type | Key assumption |
| :--- | :--- | :--- |
| **CCA** (`alignment/CCA.py`) | Linear (correlation max.) | Linear relationship between spaces |
| **KCCA** (`alignment/KCCA.py`) | Non-linear (RBF kernel) | Allows non-linear geometric structure |
| **VecMap** (`alignment/VecMap.py`) | Linear (orthogonal mapping) | Spaces are (near) isomorphic; self-learning + CSLS |

### RQ1 outputs

- `results/rq1_results.csv` — 9 rows (3 languages × 3 methods) × 8 columns.
- `results/rq1_translation_quality.png` — P@1 / P@5 / MCS bars.
- `results/rq1_cka_before_after.png` — CKA before vs. after alignment.
- `results/rq1_cka_delta.png` — ΔCKA per method.
- `results/rq1_ner_f1.png` — zero-shot NER F1.
- `results/rq1_heatmap.png` — full metric summary grid.
- `results/rq1_radar.png` — per-language method radar.

---

## Running RQ2 — data efficiency and effect of morphological variation thereon (efficiency)

RQ2 uses a class-based pipeline that separates alignemnt, diagnostics, and evaluation into
explicit stages. Every run is reproducible with hyperparameters serialized to 'config.json'
alongside the results CSV.

### Pipeline architecture

```
Stage 1  — Alignment       AlignerBase.fit()                fitted Aligner
Diagnostic layer           AlignerBase.alignment_quality()  BLI p@1, CKA

Stage 2  — Projection      AlignerBase.project()         (n_tokens, dim) float32
Stage 3  — Evaluation      run_evaluation()              entity-level F1
```

The diagnostic layer sits between fitting and evaluation so that alignment geometry can
be assessed independently of NER performance. This allows the analysis to distinguish 
between failures caused by poor alignment and failures caused by vocabulary coverage limitations


### Intermediate diagnostics

| Metric | What it measures | Why it matters |
| :--- | :--- | :--- |
| **BLI p@1** | Fraction of lexicon pairs where the projected source vector retrieves the correct English translation as its nearest neighbour | Alignment quality independent of NER; a reliable predictor of downstream F1 |
| **CKA** | Geometric similarity between source anchor space and English anchor space | Whether the two embedding spaces have compatible structure; low CKA predicts poor alignment regardless of method |
| **VecMap coverage** | Fraction of inference-time tokens found in the aligned vocabulary | Isolates vocabulary coverage as a failure mode distinct from alignment geometry |
| **n_anchors** | Number of valid lexicon pairs used for fitting and diagnostics | Contextualises BLI and CKA reliability — estimates from fewer than ~50 pairs are noisy |

### Experimentation procedure

RQ2 is designed for systematic hyperparameter sweeps. Each run accepts a named
configuration and writes results to an isolated directory:

```bash | PowerShell
# Baseline run — all languages, all fractions, all methods
python run_rq2_class_based.py --run-name baseline

# Single method sweep for rapid iteration
python run_rq2_class_based.py --run-name cca_full --methods CCA

# KCCA gamma sweep (optional)
python run_rq2_class_based.py --run-name kcca_g001 --methods KCCA --kcca-gamma 0.01
python run_rq2_class_based.py --run-name kcca_g01  --methods KCCA --kcca-gamma 0.1
python run_rq2_class_based.py --run-name kcca_g1   --methods KCCA --kcca-gamma 1.0

# CCA n_components sweep (optional)
python run_rq2_class_based.py --run-name cca_k50  --methods CCA --cca-n-components 50
python run_rq2_class_based.py --run-name cca_k100 --methods CCA --cca-n-components 100
python run_rq2_class_based.py --run-name cca_k150 --methods CCA --cca-n-components 150

# Subset of languages or fractions
python run_rq2_class_based.py --run-name zul_only --langs zul
python run_rq2_class_based.py --run-name high_fractions --fractions 0.5 0.75 1.0

# Force retrain (ignore cached artifacts)
python run_rq2_class_based.py --run-name baseline --force
```

### CLI reference

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--run-name` | `default` | Labels the output directory under `results/rq2/` |
| `--langs` | all | Space-separated language codes: `zul nso tsn` |
| `--fractions` | all | Corpus fractions: `0.05 0.1 0.25 0.5 0.75 1.0` |
| `--methods` | all | Alignment methods: `CCA KCCA VecMap` |
| `--split` | `test` | MasakhaNER split: `train`, `dev`, or `test` |
| `--force` | off | Recompute cached embeddings and alignment artifacts |
| `--device` | `cpu` | PyTorch device: `cpu` or `cuda` |
| `--ner-epochs` | `10` | Training epochs for the English BiLSTM-CRF |
| `--cca-n-components` | `100` | CCA dimensionality cap |
| `--kcca-n-components` | `50` | KCCA dimensionality cap |
| `--kcca-gamma` | `0.1` | RBF kernel bandwidth |
| `--kcca-reg` | `1e-3` | Tikhonov regularisation strength |
| `--kcca-max-anchors` | `1000` | Maximum anchor pairs used for KCCA fitting |
| `--no-validate-ner` | off | Skip English NER validation on CoNLL-2003 test |

### RQ2 outputs

Each run writes to `results/rq2/{run_name}/`:

- `config.json` — full hyperparameter record for reproducibility
- `results.csv` — one row per language × fraction × method with columns:

| Column | Description |
| :--- | :--- |
| `language` | Language code |
| `fraction` | Corpus fraction used for source embeddings |
| `subset_tokens` | Token count of the corpus subset |
| `method` | Alignment method |
| `en_baseline_f1` | English NER F1 on CoNLL-2003 test (fixed reference) |
| `precision` | Zero-shot NER precision on MasakhaNER |
| `recall` | Zero-shot NER recall on MasakhaNER |
| `f1` | Zero-shot NER F1 on MasakhaNER |
| `bli_p1` | BLI precision@1 after alignment |
| `cka` | CKA between source and English anchor spaces |
| `n_anchors` | Number of valid lexicon pairs used |
| `vecmap_coverage` | VecMap vocabulary hit rate at inference (VecMap only) |

### Visualisation

```bash | PowerShell
python visualize_rq2.py --results-csv results/rq2/cca_full/results.csv
```

Generates one figure per alignment method with:
- Primary y-axis — zero-shot NER F1 (solid lines)
- Secondary y-axis — BLI p@1 (dashed lines)
- X-axis — corpus fraction
- One line colour per language, consistent across both axes
- Horizontal reference line for English baseline F1

Plots are saved alongside the results CSV.

---

## Module reference

| File | Responsibility |
| :--- | :--- |
| `config.py` | All paths, language codes (`zul`/`nso`/`tsn`/`eng`), hyperparameters, fraction helpers |
| `embeddings.py` | `train_fasttext`, `load_embeddings_as_matrix` (L2-normalised), `save_embeddings_as_txt`, `supplement_with_oov` |
| `lexicon.py` | `load_lexicon`, `split_lexicon`, `build_anchor_matrices` |
| `evaluation.py` | `precision_at_k`, `mean_cosine_similarity`, `linear_cka` |
| `alignment/CCA.py` | `CCAAligner` over sklearn CCA |
| `alignment/KCCA.py` | `KCCAAligner` — RBF kernels, centred Gram matrices, batched projection |
| `alignment/VecMap.py` | `run_vecmap` (subprocess), `load_txt_embeddings` |
| `rq2/aligners/base.py` | `AlignerBase` — abstract interface: `fit`, `project`, `alignment_quality` |
| `rq2/aligners/cca.py` | `CCAWrapper` — linear alignment behind `AlignerBase` |
| `rq2/aligners/kcca.py` | `KCCAWrapper` — kernel alignment behind `AlignerBase` |
| `rq2/aligners/vecmap.py` | `VecMapWrapper` — orthogonal mapping behind `AlignerBase`; tracks vocabulary coverage |
| `rq2/diagnostics/bli.py` | `bli_precision_at_1` — nearest-neighbour retrieval accuracy |
| `rq2/diagnostics/cka.py` | `compute_cka` — linear CKA between anchor spaces |
| `rq2/pipeline/config.py` | `RQ2Config` dataclass — single source of truth for all hyperparameters |
| `rq2/pipeline/fasttext_stage.py` | `ensure_fasttext_model`, `load_source_embeddings` |
| `rq2/pipeline/ner_stage.py` | `load_ner_model`, `validate_ner_model`, `entity_types_from_tagset` |
| `rq2/pipeline/alignment_stage.py` | `make_aligner`, `fit_aligner`, `run_diagnostics` |
| `rq2/pipeline/evaluation_stage.py` | `run_evaluation`, `assemble_result_row` |
| `rq2/pipeline/experiment_runner.py` | `run_rq2(config)` — main pipeline loop |
| `transfer/zero_shot_eval.py` | `BiLSTMCRF`, `train_or_load_english_ner_model`, `evaluate_sentences`, dataset readers |
| `transfer/corpus_subsets.py` | Deterministic, nested, token-budgeted corpus subsets |

---

## Notes and caveats

- **English pivot corpus.** English FastText is trained on the NCHLT English
  corpus, not Wikipedia. Embeddings may be less rich than published baselines
  using large general corpora.
- **NER model validation.** The English BiLSTM-CRF is validated on CoNLL-2003
  test before every RQ2 run. A warning is raised if F1 falls below 0.75 —
  zero-shot results should be interpreted cautiously below that threshold.
- **OOV handling.** CCA and KCCA generalise to unseen tokens via their learned
  transformation matrix applied to FastText subword vectors. VecMap has a fixed
  aligned vocabulary and falls back to unaligned FastText vectors for OOV tokens
  — tracked by the `vecmap_coverage` column.
- **CKA reliability.** CKA estimates from fewer than ~50 anchor pairs are
  noisy. The `n_anchors` column should be reported alongside CKA scores.
- **Hyperparameter tuning.** `cca_n_components` is capped at 100 following
  convention in the cross-lingual embedding literature. `kcca_gamma` and
  `kcca_reg` are carried over from prior work without language-specific
  tuning.
- **Reproducibility.** Every run serialises its full `RQ2Config` to
  `config.json` in the results directory. Results from different runs are
  stored in separate directories and never overwrite each other.

---


## Quick git reference

| Action | Command |
| :--- | :--- |
| Update local | `git pull origin main` |
| New branch | `git checkout -b <branch-name>` |
| Stage changes | `git add .` |
| Commit | `git commit -m "message"` |
| Push | `git push origin <branch-name>` |
