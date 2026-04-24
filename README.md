# COS760-project

## Folder Structure:
/COS760-project
│
├── .gitignore               <-- files we don't want to upload
├── requirements.txt         <-- List pandas, fasttext, torch, etc.
├── /scripts
│   ├── /preprocessing       <-- Franco's cleaning & subset scripts
│   ├── /training            <-- Franco's FastText & BiLSTM-CRF scripts
│   ├── /alignment           <-- Tom's VecMap/KCCA scripts
│   └── /evaluation          <-- Ndamulelo's scoring scripts
│
├── /configs                 <-- Hyperparameters (learning rates, n-gram sizes)
└── /notebooks               <-- For quick EDA or plotting learning curves
