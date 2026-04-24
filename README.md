# COS760-project: Optimizing Cross-Lingual Embeddings for Bantu Languages

## 📂 Folder Structure
```text
/COS760-project
│
├── .gitignore               <-- Files we don't want to upload (data, venv, binaries)
├── requirements.txt         <-- Project dependencies (pandas, fasttext, torch, etc.)
│
├── /scripts
│   ├── /preprocessing       <-- [Franco] Cleaning, normalization, & RQ2 subset generation
│   ├── /training            <-- [Franco] FastText training & BiLSTM-CRF NER pipeline
│   ├── /alignment           <-- [Tom] VecMap, CCA, and KCCA alignment logic
│   └── /evaluation          <-- [Ndamulelo] P@k, CKA scores, and learning curve plotting
│
├── /configs                 <-- Hyperparameters (n-gram sizes, learning rates, epochs)
└── /notebooks               <-- Exploratory Data Analysis (EDA) and visualization
```
## 🚀 Quick Git Reference
```text
Use this cheat sheet for the standard project workflow. For new folders to appear in `git status`, ensure they contain at least one file (e.g., `.gitkeep`).

| Action | Command |
| :--- | :--- |
| **Clone Repo** | `git clone <url>` |
| **Update Local** | `git pull origin main` |
| **New Branch** | `git checkout -b <branch-name>` |
| **Switch Branch** | `git checkout <branch-name>` |
| **Check Status** | `git status` |
| **Stage Changes** | `git add .` |
| **Commit** | `git commit -m "Your message"` |
| **Push to GitLab** | `git push origin <branch-name>` |

### Standard Workflow Example:
```bash
# 1. Get the latest code
git pull origin main

# 2. Create a branch for your work
git checkout -b feature/setup-folders

# 3. (Make your changes/add files)

# 4. Check, Stage, and Commit
git status
git add .
git commit -m "feat: initial directory structure"

# 5. Upload to GitLab
git push origin feature/setup-folders
```
