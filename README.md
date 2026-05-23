# COS760 — Cross-Lingual Embedding Alignment for isiZulu, Sepedi & Setswana

Optimising cross-lingual word embeddings for three South African Bantu
languages (isiZulu, Sepedi, Setswana) aligned onto an English pivot space.

The project answers two research questions:

- **RQ1 — Alignment strategy.** Does **CCA** outperform **VecMap** and
  **Kernel CCA (KCCA)** when aligning each language onto English, and does the
  relative advantage of non-linear methods differ by morphology (conjunctive
  isiZulu vs. disjunctive Sepedi/Setswana)?
- **RQ2 — Data efficiency.** How does the corpus size needed to reach an
  acceptable zero-shot NER F1 differ between conjunctive and disjunctive
  languages?

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
├── visualize_rq1.py          # Regenerates the 6 RQ1 figures from rq1_results.csv
├── 760.sh                    # Bootstrap: create .venv and install requirements
├── requirements.txt
│
├── alignment/
│   ├── CCA.py                # Linear CCA (sklearn) — correlation-maximising projection
│   ├── KCCA.py               # Kernel CCA (RBF) — non-linear alignment, from scratch
│   └── VecMap.py             # Wrapper around the bundled VecMap tool (orthogonal mapping)
│
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
└── results/                  # rq1_results.csv + 6 PNG figures
```

---

## Setup

The bootstrap script creates a virtual environment and installs everything.
It looks for `python3.11` / `python3.12` / `python3` (in that order).

```bash
cd COS760-project
./760.sh
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

## Running RQ2 — data efficiency *(implemented, not yet run)*

`run_rq2.py` trains source embeddings on progressively smaller corpus subsets
(100 / 75 / 50 / 25 / 10 / 5 %), aligns each with every method, and measures
zero-shot NER F1 to build learning curves.

```bash
source .venv/bin/activate
python run_rq2.py

# Useful options:
python run_rq2.py --langs zul tsn --fractions 1.0 0.25 0.05 --methods CCA KCCA VecMap --split test
python run_rq2.py --force        # rebuild cached embeddings / alignments
```

It generates `data/subsets/{lang}/` corpora automatically and writes
`results/rq2_results.csv` (per language × fraction × method:
precision / recall / F1, plus VecMap coverage).

> Learning-curve plots and the 50 %-F1 break-even analysis are **not yet
> implemented** — there is currently no `visualize_rq2.py`.

---

## Module reference

| File | Responsibility |
| :--- | :--- |
| `config.py` | All paths, language codes (`zul`/`nso`/`tsn`/`eng`), hyperparameters, fraction helpers. |
| `embeddings.py` | `train_fasttext`, `load_embeddings_as_matrix` (L2-normalised), `save_embeddings_as_txt` (for VecMap), `supplement_with_oov` (subword inference for OOV anchors). |
| `lexicon.py` | `load_lexicon` (slash-synonym expansion + NCHLT dash stripping), `split_lexicon` (train/test), `build_anchor_matrices`. |
| `evaluation.py` | `precision_at_k`, `mean_cosine_similarity`, `linear_cka` (memory-efficient `‖XᵀY‖²_F` form). |
| `alignment/CCA.py` | `CCAAligner` over sklearn's CCA. |
| `alignment/KCCA.py` | `KCCAAligner` — RBF kernels, centred Gram matrices, batched out-of-sample projection. |
| `alignment/VecMap.py` | `run_vecmap` (subprocess call to bundled VecMap), `load_txt_embeddings`. |
| `transfer/zero_shot_eval.py` | `BiLSTMCRF`, `train_or_load_english_ner_model`, `evaluate_sentences`, MasakhaNER/CoNLL readers. |
| `transfer/corpus_subsets.py` | Deterministic, nested, token-budgeted corpus subsets for RQ2. |

---

## Notes & caveats

- **English pivot corpus.** The proposal mentions English Wikipedia; the code
  trains English FastText on the **NCHLT English** corpus (see `config.py`).
- **OOV supplementation.** Source/English vocabularies are extended with
  FastText subword vectors for lexicon words that fall below `minCount`. Anchors
  for *training* the aligners use these supplemented vocabularies; **intrinsic
  metrics are computed on the original vocabulary only** to keep CCA/KCCA/VecMap
  comparable.
- **VecMap** runs as an external subprocess on the exported `.txt` embeddings
  and is skipped if the text export is missing.
- The `.bin` FastText models are large and regenerated from the corpora when
  absent.

---

## Quick git reference

| Action | Command |
| :--- | :--- |
| Update local | `git pull origin main` |
| New branch | `git checkout -b <branch-name>` |
| Stage changes | `git add .` |
| Commit | `git commit -m "message"` |
| Push | `git push origin <branch-name>` |
