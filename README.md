# MLCOE Task 3: Unsupervised Learning & Ensemble Techniques

An end-to-end machine learning project analyzing high-dimensional security telemetry from the Global Terrorism Database (GTD)[cite: 3, 9]. This repository covers exploratory data analysis, dimensionality reduction via Principal Component Analysis (PCA), a conceptual study of unsupervised clustering, supervised classification using ensemble architectures (Random Forest, AdaBoost, and XGBoost), and an interactive Streamlit inference engine.

---

## 📌 Project Overview

* **Domain:** Defense, Geopolitical Risk & National Security  
* **Dataset:** Global Terrorism Database (GTD)[cite: 9]  
* **Scale:** ~181,690 records × 135 attributes (meets the 150k+ rows and 30+ columns criteria)  
* **Core Objective:** Clean and analyze complex, multi-feature incident data, apply PCA for dimensional compression, benchmark bagging versus boosting ensemble models to predict attack success/outcomes, and deploy the optimal pipeline to an interactive dashboard.

---

## 📂 Predicted Repository Architecture

```text
TASK_3/
├── data/                                # Local raw dataset directory (untracked)
│   └── globalterrorismdb_0718dist.csv
├── models/                              # Serialized models and inference preprocessors
│   ├── best_ensemble_model.pkl
│   ├── scaler.pkl
│   ├── encoder.pkl
│   └── pca_transformer.pkl
├── notebooks/
│   ├── 01_eda_and_pca.ipynb             # Parts 1, 2, and 3
│   └── 02_ensemble_modeling.ipynb       # Parts 4 and 5
├── app.py                               # Part 6: Streamlit interactive frontend
├── requirements.txt                     # Pinned project dependencies
├── .gitignore                           # Ignores large data, cache, and virtualenvs
└── README.md                            # Project documentation and performance audit