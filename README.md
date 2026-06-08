# Loan Approval Prediction — Financial Risk Management

> Automated loan approval decisions using 5 classical ML models + an Artificial Neural Network, with SMOTE for class imbalance and SHAP for interpretability.

![Python](https://img.shields.io/badge/Python-3.12.4-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-ANN-FF6F00?logo=tensorflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

Financial institutions process thousands of loan applications daily. Manual review is slow, inconsistent, and prone to human bias. This project builds and compares multiple machine learning models to automate loan approval decisions based on applicant demographics, financial history, and credit behavior.

The core challenge: a severely imbalanced dataset where only **23.9% of applications are approved**. The pipeline addresses this with SMOTE and validates model performance using stratified cross-validation.

**Key results:** All models achieved 100% accuracy on the test set after SMOTE and hyperparameter tuning — see [the important caveat below](#-important-note-on-perfect-metrics).

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Financial Risk for Loan Approval](https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval?select=Loan.csv) |
| File | `Loan.csv` |
| Rows | 20,000 |
| Features | 36 |
| Target | `LoanApproved` (0 = Rejected, 1 = Approved) |
| Class distribution | 76.1% Rejected / 23.9% Approved |

### Key Features

| Category | Features |
|---|---|
| Demographic | Age, MaritalStatus, NumberOfDependents, EducationLevel, EmploymentStatus |
| Financial | AnnualIncome, MonthlyIncome, NetWorth, DebtToIncomeRatio, CreditCardUtilizationRate |
| Credit history | CreditScore, BankruptcyHistory, PreviousLoanDefaults, PaymentHistory, LengthOfCreditHistory |
| Loan details | LoanAmount, LoanDuration, LoanPurpose, InterestRate, RiskScore |

---

## Architecture

```
Raw Data (20,000 rows, 36 features)
         │
         ▼
Preprocessing
  ├── Handle missing values (none present in this dataset)
  ├── Remove duplicates (none found)
  ├── Encode categorical variables (one-hot & label encoding)
  └── Standardize numerical features (StandardScaler)
         │
         ▼
Exploratory Data Analysis
  ├── Distribution plots & histograms for all features
  ├── Correlation heatmap of key financial features
  ├── Target class distribution
  ├── CreditScore vs InterestRate scatter plot
  ├── AnnualIncome vs LoanAmount analysis
  └── Outlier detection (boxplots + IQR method)
         │
         ▼
Class Imbalance Detected (76.1% Rejected / 23.9% Approved)
         │
         ▼
SMOTE (Synthetic Minority Over-sampling Technique)
  └── Applied on training split only → balanced 50/50 training set
         │
         ▼
5 Classical ML Models + ANN
  ├── Logistic Regression      (baseline linear model)
  ├── Decision Tree            (interpretable rule-based)
  ├── Random Forest            (ensemble, 100 trees)
  ├── Support Vector Machine   (RBF kernel)
  ├── K-Nearest Neighbors      (k=5–15)
  └── Artificial Neural Network (TensorFlow/Keras)
         │
         ▼
Evaluation
  ├── Confusion Matrix
  ├── Precision / Recall / F1-Score
  ├── Accuracy & ROC-AUC
  └── 5-fold Stratified Cross-Validation
         │
         ▼
SHAP Analysis — Feature Importance & Explainability
```

---

## Approach

### 1. Preprocessing

- **Missing values:** None present — no imputation needed.
- **Duplicates:** Zero duplicates found (`df.duplicated().sum() == 0`).
- **Encoding:** One-hot for nominal categories (LoanPurpose, MaritalStatus); label encoding for ordinal categories (EducationLevel, EmploymentStatus).
- **Scaling:** `StandardScaler` applied to all numerical features.
- **Outliers:** Detected via boxplots and IQR method. Retained intentionally — high-income edge cases are valid real-world applicants.

### 2. Handling Class Imbalance

The dataset is severely imbalanced: 76.1% rejected, 23.9% approved. Without correction, models learn to predict "rejected" for everything and still appear 76% accurate.

**Solution:** SMOTE (Synthetic Minority Over-sampling Technique) from `imbalanced-learn`.

- SMOTE generates synthetic samples of the minority class (Approved) using k-nearest neighbors interpolation.
- Applied **after** the train/test split, on the training set only — to avoid data leakage into evaluation.
- Result: Balanced 50/50 training set.

### 3. Models

| Model | Why It Was Included |
|---|---|
| Logistic Regression | Interpretable linear baseline; coefficients via `statsmodels` for statistical significance |
| Decision Tree | Transparent, rule-based — shows exact decision logic |
| Random Forest | Handles non-linear relationships, robust to outliers, strong ensemble performance |
| SVM | Effective for high-dimensional spaces; tests margin-based classification |
| KNN | Distance-based baseline; tests local neighborhood assumptions |
| Neural Network (ANN) | Captures complex non-linear interactions between financial features |

**Hyperparameter tuning:** `GridSearchCV` with 5-fold stratified cross-validation for all 5 classical models.

### 4. Artificial Neural Network

Built with TensorFlow/Keras:

| Layer | Units | Activation |
|---|---|---|
| Dense 1 | 10 | ReLU + L2 regularization |
| Dense 2 | 15 | Tanh |
| Dense 3 | 10 | SiLU |
| Dense 4 | 7 | Tanh |
| Dense 5 | 5 | ELU |
| Dense 6 | 3 | ReLU |
| Output | 1 | Sigmoid |

- **Optimizer:** Adam
- **Loss:** Binary Crossentropy
- **Epochs:** 10 &nbsp;|&nbsp; **Batch Size:** 32
- Evaluated with classification report and confusion matrix

### 5. Interpretability — SHAP Analysis

SHAP (SHapley Additive exPlanations) used on the best-performing Random Forest model to explain individual predictions and identify globally important features.

---

## Results

### Model Performance (after SMOTE + hyperparameter tuning)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 100% | 1.00 | 1.00 | 1.00 | 1.00 |
| Decision Tree | 100% | 1.00 | 1.00 | 1.00 | 1.00 |
| Random Forest | 100% | 1.00 | 1.00 | 1.00 | 1.00 |
| SVM | 100% | 1.00 | 1.00 | 1.00 | 1.00 |
| KNN | 100% | 1.00 | 1.00 | 1.00 | 1.00 |
| Neural Network (ANN) | 100% | 1.00 | 1.00 | 1.00 | 1.00 |

Test set: 4,000 instances (held out before SMOTE).

### Feature Importance (SHAP — Top 5)

| Rank | Feature | Interpretation |
|---|---|---|
| 1 | CreditScore | Strongest single predictor of repayment ability |
| 2 | AnnualIncome | Higher income → lower default risk |
| 3 | DebtToIncomeRatio | High ratio signals over-leveraged applicants |
| 4 | LoanAmount | Larger loans increase risk exposure |
| 5 | PaymentHistory | Past behavior predicts future repayment |

---

## Important Note on Perfect Metrics

All 6 models achieved 100% accuracy. This is an expected outcome of this dataset + SMOTE combination, not necessarily an indicator of real-world performance.

**Two likely explanations:**

1. **Dataset structure:** The Kaggle dataset contains highly predictive features (CreditScore, PaymentHistory, DebtToIncomeRatio) that almost perfectly separate classes once imbalance is corrected. With clean, structured tabular data and strong features, tree-based ensembles can genuinely achieve near-perfect results.

2. **Potential data leakage risk:** If SMOTE is applied before the train/test split, synthetic minority samples could appear in both train and test sets — inflating test metrics. In this project, SMOTE was applied to the training set only, which reduces (but may not eliminate) this risk. For full rigor, SMOTE should be placed inside each cross-validation fold using `Pipeline + imblearn`.

**What this means:** The models perform perfectly on this specific dataset. Real-world generalization would require validation on an independent loan dataset from a different source or institution.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12.4 |
| Data handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML models | Scikit-learn (RandomForest, SVM, KNN, LogisticRegression, DecisionTree, GridSearchCV, StandardScaler) |
| Statistical regression | Statsmodels |
| Class imbalance | Imbalanced-learn (SMOTE) |
| Neural network | TensorFlow / Keras |
| Interpretability | SHAP |
| Development | Jupyter Notebook, Google Colab |

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/prabhathv07/Financial-Risk-Loan-Approval.git
cd Financial-Risk-Loan-Approval
```

```bash
# Install dependencies
pip install pandas numpy scikit-learn tensorflow imbalanced-learn shap matplotlib seaborn statsmodels
```

Download `Loan.csv` from [Kaggle](https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval?select=Loan.csv) and place it in the project directory.

```bash
# Run via Jupyter Notebook
jupyter notebook Loan_Approval_Project.ipynb

# Or run the Python script directly
python loan_approval_project.py
```

The program will:
1. Load the dataset and run EDA (distribution plots, correlation matrix, class balance)
2. Preprocess (encode, scale, split, apply SMOTE to training set)
3. Train all 6 models with `GridSearchCV` hyperparameter tuning
4. Output per-model metrics (precision, recall, F1, accuracy, ROC-AUC)
5. Display SHAP summary plots for feature importance

---

## What I'd Do Next

- **Validate on external data** — Test on a real-world loan dataset from a different source to assess true generalization beyond this specific Kaggle dataset.
- **Fix SMOTE placement** — Use `imblearn.pipeline.Pipeline` to apply SMOTE inside each CV fold, eliminating any residual data leakage risk.
- **Deploy as API** — Wrap the Random Forest model in a FastAPI service with input validation and a simple front-end dashboard for loan officers.
- **Add live explainability** — Integrate SHAP into the API response so reviewers can see exactly why an application was approved or rejected.
- **Time-based validation** — If the data is time-stamped, train on older applications and test on newer ones to simulate real deployment conditions.
- **Monitor for drift** — Implement automated monitoring to detect when model performance degrades (data drift / concept drift) after deployment.
- **A/B testing** — Compare the ML model against the existing manual approval process in a controlled live pilot.

---

## Author

**Prabhath Vinay Vipparthi**
MS Data Science — New Jersey Institute of Technology (May 2026)
[GitHub](https://github.com/prabhathv07) · [LinkedIn](https://www.linkedin.com/in/prabhath-vipparthi-90544b225/)
