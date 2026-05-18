# COS760-project

Cross-lingual embedding alignment experiments for isiZulu, Sepedi, and Setswana.

## Repository Layout
```text
COS760-project/
|-- .venv/
|-- config.py
|-- embeddings.py
|-- evaluation.py
|-- lexicon.py
|-- run_rq1.py
|-- run_rq2.py
|-- run_rq3.py
|-- visualize_rq1.py
|-- alignment/
|   |-- CCA.py
|   |-- KCCA.py
|   `-- VecMap.py
|-- data/
|   |-- Bilingual Seed Lexicons/
|   |-- NCHLT Text Corpora/
|   |   |-- en/
|   |   |-- nso/
|   |   |-- tn/
|   |   `-- zu/
|   |-- ner_MasakhaNER 2.0/
|   |   `-- masakhaner2/
|   |       |-- zul/
|   |       `-- tsn/
|   |-- zulu/
|   `-- ...
|-- embeddings/
|-- results/
`-- vecmap/
```

## Required Data
`run_rq1.py` needs three types of data.

### 1. NCHLT Text Corpora
Required languages:
- isiZulu
- Sepedi
- Setswana

Available in this workspace now:
- English clean corpus
- isiZulu clean and raw corpora
- Sepedi clean and raw corpora
- Setswana clean and raw corpora

### 2. MasakhaNER 2.0 / SADiLaR NER
Required languages:
- isiZulu
- Sepedi
- Setswana



Available in this workspace now:
- isiZulu folder
- Setswana folder
- **Sepedi folder is still missing**


Status in this workspace:
- **all three bilingual seed lexicons are still missing**

## How to Run
Activate the environment and run the RQ1 pipeline:
```bash
source .venv/bin/activate
python run_rq1.py
```
visualization:
```bash
python visualize_rq1.py
```

## RQ2 (Zero-shot NER + Data Efficiency)
RQ2 is implemented in `run_rq2.py`.

### Required Data (local-only)
This repo ignores datasets by default (see `.gitignore`). Place files locally under `data/`.

Minimal expected layout:
```text
data/
   NCHLT Text Corpora/              # as in config.py (keep original NCHLT subfolders)
   Bilingual Seed Lexicons/
      zul_en.txt
      nso_en.txt
      tsn_en.txt
   ner_MasakhaNER 2.0/masakhaner2/
      zul/
      nso/
      tsn/
   conll2003/
      train.txt (or any file containing "train" in its name)
      dev.txt   (or any file containing "dev"/"valid")
      test.txt  (or any file containing "test")
```

RQ2 will also generate subset corpora under `data/subsets/{lang}/` automatically.

### Run
```bash
source .venv/bin/activate
python run_rq2.py
```

Useful options:
```bash
python run_rq2.py --langs zul tsn --fractions 1.0 0.25 0.05 --methods CCA KCCA VecMap --split test
python run_rq2.py --force   # retrain embeddings/NER and rebuild cached alignment artifacts
```

### Outputs
- `results/rq2_results.csv` — per (language, fraction, method): entity-level Precision/Recall/F1.
- `outputs/ner/bilstm_crf_conll2003.pt` — English BiLSTM-CRF checkpoint.
- `embeddings/aligned/` — cached CCA/KCCA alignment artifacts per fraction.
- `outputs/vecmap_*` — VecMap alignment outputs.

## What `run_rq1.py` Does
1. Loads or trains FastText embeddings for English, isiZulu, Sepedi, and Setswana.
2. Loads the bilingual seed lexicon for each language pair.
3. Builds train/test lexicon splits.
4. Aligns embeddings using:
   - CCA
   - KCCA
   - VecMap
5. Writes the results to `results/rq1_results.csv`.

## Current Missing Files
The script will keep printing missing-data warnings until these files are added:
- `data/Bilingual Seed Lexicons/zul_en.txt`
- `data/Bilingual Seed Lexicons/nso_en.txt`
- `data/Bilingual Seed Lexicons/tsn_en.txt`

If you want the full RQ1 pipeline to complete end-to-end, these three files are the next thing to fetch.

## Notes
- The project now uses the `data/` folder inside the repo as the canonical location for datasets.
- FastText `.bin` files are generated automatically from the NCHLT corpora when they are missing.
- VecMap also requires text embeddings and will be skipped if the text export is not available yet.

## Quick Git Reference
| Action | Command |
| :--- | :--- |
| Clone repo | `git clone <url>` |
| Update local | `git pull origin main` |
| New branch | `git checkout -b <branch-name>` |
| Switch branch | `git checkout <branch-name>` |
| Check status | `git status` |
| Stage changes | `git add .` |
| Commit | `git commit -m "Your message"` |
| Push | `git push origin <branch-name>` |
