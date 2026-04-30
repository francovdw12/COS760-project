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
