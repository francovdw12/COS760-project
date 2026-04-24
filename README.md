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
