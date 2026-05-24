# 💳 Credit Card Fraud Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**An end-to-end machine learning pipeline for detecting fraudulent credit card transactions using ensemble models, SMOTE resampling, SHAP explainability, and hyperparameter tuning.**

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Project Workflow](#-project-workflow)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Models Used](#-models-used)
- [Handling Class Imbalance](#-handling-class-imbalance)
- [Evaluation Metrics](#-evaluation-metrics)
- [Results](#-results)
- [Explainability](#-explainability)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

Credit card fraud is a major challenge in the financial industry, costing billions of dollars every year. This project builds a robust, end-to-end ML pipeline to detect fraudulent transactions with high precision and recall — even under extreme class imbalance (only ~0.17% of transactions are fraud).

**Key highlights:**
- 🔄 Full preprocessing pipeline with feature scaling
- 📊 Exploratory Data Analysis (EDA) with visual insights
- ⚖️ Class imbalance handled via **SMOTE** (Synthetic Minority Oversampling)
- 🤖 4 ensemble models: **Random Forest, XGBoost, LightGBM, CatBoost**
- 🔧 **Hyperparameter tuning** with RandomizedSearchCV + cross-validation
- 📈 Evaluated on Recall, Precision, F1-Score, and ROC-AUC
- 🧠 Model explainability with **SHAP**
- 🚀 Best model deployed for real-world use

---

## 🔄 Project Workflow

```
Raw Data (creditcard.csv)
       │
       ▼
Data Preprocessing
  ├── Drop 'Time' column
  └── Scale 'Amount' with RobustScaler
       │
       ▼
Exploratory Data Analysis
  ├── Class imbalance visualization
  ├── Transaction amount distribution
  └── Amount vs Fraud (violin plot)
       │
       ▼
Train/Test Split (80/20, stratified)
       │
       ▼
SMOTE Oversampling (on train set only)
       │
       ▼
Model Training & Evaluation
  ├── Random Forest
  ├── XGBoost
  ├── LightGBM
  └── CatBoost
       │
       ▼
Hyperparameter Tuning (RandomizedSearchCV)
       │
       ▼
Best Model Selection → SHAP Explainability → Deployment
```

---

## 📂 Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.

| Property | Details |
|---|---|
| Total Transactions | 284,807 |
| Fraudulent Transactions | 492 (~0.17%) |
| Features | 30 (V1–V28 PCA features + Amount + Class) |
| Target | `Class` (0 = Normal, 1 = Fraud) |

> ⚠️ **Note:** The dataset is not included in this repo due to size. Download `creditcard.csv` from the Kaggle link above and place it in the root directory.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Models | Scikit-Learn, XGBoost, LightGBM, CatBoost |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Hyperparameter Tuning | RandomizedSearchCV |
| Notebook | Jupyter Notebook |

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── data_preprocessing.ipynb     # Main notebook (EDA + training + evaluation)
├── artifacts/
│   └── 1_cleaned_data.csv       # Preprocessed dataset (auto-generated)
├── requirements.txt             # All dependencies
├── .gitignore                   # Files excluded from Git
└── README.md                    # You are here
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the root folder.

### 5. Run the Notebook

```bash
jupyter notebook data_preprocessing.ipynb
```

---

## 🤖 Models Used

| Model | Why It's Used |
|---|---|
| **Random Forest** | Robust ensemble, handles noise well, great baseline |
| **XGBoost** | High performance on tabular data, gradient boosting |
| **LightGBM** | Faster training, great for large datasets, AUC-optimized |
| **CatBoost** | Handles class imbalance naturally, strong out-of-the-box |

All models are trained on SMOTE-resampled data and tuned using cross-validation.

---

## ⚖️ Handling Class Imbalance

The dataset is severely imbalanced — **99.82% normal** vs **0.17% fraud**.

**Strategy used: SMOTE (Synthetic Minority Oversampling Technique)**

- Applied only on the **training set** (never on test set — avoids data leakage)
- Generates synthetic fraud samples to balance classes
- Result: balanced training distribution → models learn fraud patterns effectively

```
Before SMOTE:  Counter({0: 227451, 1: 394})
After SMOTE:   Counter({0: 227451, 1: 227451})
```

---

## 📈 Evaluation Metrics

In fraud detection, **accuracy alone is misleading** (a model predicting "no fraud" always gets 99.8% accuracy). We focus on:

| Metric | What It Measures |
|---|---|
| **Recall** | How many actual frauds did we catch? (minimize false negatives) |
| **Precision** | Of flagged frauds, how many were real? (minimize false alarms) |
| **F1-Score** | Harmonic mean of Precision & Recall — balanced view |
| **ROC-AUC** | Model's ability to distinguish fraud from non-fraud overall |

---

## 📊 Results

> Results shown are on the held-out test set (20% of data, never seen during training or SMOTE).

| Model | Recall | Precision | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |
| LightGBM | — | — | — | — |
| CatBoost | — | — | — | — |

> 📝 Results will be updated after full training runs are completed.

---

## 🧠 Explainability

This project uses **SHAP (SHapley Additive exPlanations)** to interpret why the model flags a transaction as fraud.

- Feature importance at global and local (per-prediction) level
- SHAP summary plots, waterfall plots, and force plots
- Helps build trust in the model's decisions

---

## 🚀 Deployment

The best-performing model (based on ROC-AUC + F1) is selected for deployment.

> Deployment details will be added here (e.g., Flask API / FastAPI / Streamlit app / Docker container).

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the models, add new features, or fix issues:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add: your feature description"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  Made with ❤️ | If you found this useful, please ⭐ the repo!
</div>
