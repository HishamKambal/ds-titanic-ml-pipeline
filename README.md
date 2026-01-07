# Titanic Survival Prediction — End-to-End ML Pipeline

A compact, production-style machine learning project using the **Kaggle Titanic** dataset.  
This repository demonstrates a clean and reproducible ML workflow, focusing on structure,
correctness, and engineering best practices rather than leaderboard optimization.

## What this project demonstrates
- Reproducible project structure
- Data validation and preprocessing
- Model training and evaluation
- Artifact generation (metrics, model, submission)
- One-command execution via CLI

---

## Dataset
Download `train.csv` and `test.csv` from the Kaggle Titanic competition and place them into:

data/raw/

yaml
Copy code

Kaggle competition page:  
https://www.kaggle.com/competitions/titanic

---

## Quickstart (Windows CMD)

```bat
cd ds-titanic-ml-pipeline
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m src.train --train data/raw/train.csv --test data/raw/test.csv --out outputs
Outputs
After running the pipeline, the following artifacts are generated:

outputs/metrics.json
Validation accuracy and classification report

outputs/model.joblib
Trained sklearn pipeline (preprocessing + model)

outputs/submission.csv
Kaggle-ready prediction file

Output artifacts are intentionally excluded from Git to keep the repository clean.

Project structure
text
Copy code
ds-titanic-ml-pipeline/
  src/
    train.py
  data/
    raw/                # Kaggle CSV files (not tracked)
  outputs/              # Generated artifacts (not tracked)
  requirements.txt
  README.md
  TECHNICAL_REPORT.md
  .gitignore
Modeling notes
Preprocessing implemented using ColumnTransformer

Missing values handled inside the pipeline to prevent leakage

Logistic Regression used as a strong, interpretable baseline

Validation performed using a deterministic train/validation split

Limitations
Baseline model only (no advanced feature engineering)

Single holdout validation split

No hyperparameter tuning

Next improvements
Stratified cross-validation

Feature engineering (Title extraction, FamilySize, Cabin indicators)

Unit tests and CI pipeline

yaml
Copy code

---

# ✅ FINAL `TECHNICAL_REPORT.md` (copy–paste exactly)

```md
# Technical Report — Titanic Survival Prediction (ML Pipeline)

## 1. Objective
The objective of this project is to predict passenger survival on the Titanic using the Kaggle Titanic dataset.  
The primary goal is to demonstrate a clean, reproducible, and well-structured end-to-end machine learning pipeline rather than optimizing for leaderboard rank.

---

## 2. Data

### 2.1 Source
Dataset sourced from the Kaggle Titanic competition.

### 2.2 Files
- `train.csv`: feature set with target column `Survived`
- `test.csv`: feature set without target labels

---

## 3. Features Used

A compact and defensible feature subset was selected:

### Numeric features
- `Pclass`
- `Age`
- `SibSp`
- `Parch`
- `Fare`

### Categorical features
- `Sex`
- `Embarked`

**Rationale:**  
These features are well-known high-signal predictors for Titanic survival and allow the construction of a strong baseline without complex feature engineering.

---

## 4. Preprocessing

Preprocessing is implemented inside an sklearn `Pipeline` to avoid data leakage.

### Steps
- Numeric features: median imputation
- Categorical features: most-frequent imputation
- Encoding: one-hot encoding with `handle_unknown="ignore"`

### Implementation
- `ColumnTransformer` combines numeric and categorical preprocessing
- Entire preprocessing logic is coupled with the model inside a single pipeline

---

## 5. Model

### Algorithm
- Logistic Regression

### Justification
- Interpretable coefficients
- Stable and fast training
- Suitable as a baseline for small-to-medium tabular datasets
- Low risk of overfitting when combined with simple preprocessing

---

## 6. Evaluation

### Validation strategy
- Train/validation split with fixed random seed
- Validation performed only on training data
- Final model retrained on full dataset before inference

### Metrics
- Accuracy
- Full classification report (precision, recall, F1-score)

---

## 7. Artifacts

The pipeline generates the following artifacts:

- `metrics.json`: validation metrics and classification report
- `model.joblib`: trained sklearn pipeline
- `submission.csv`: Kaggle submission file

All artifacts are generated via a single command and are excluded from version control.

---

## 8. Key Engineering Decisions

1. Pipeline-based preprocessing to prevent training/serving skew  
2. Explicit feature selection to maintain clarity and interpretability  
3. Deterministic execution via fixed random seed  
4. Clear separation between source code, data, and outputs  

---

## 9. Limitations

- No cross-validation
- No hyperparameter tuning
- Minimal feature engineering
- Baseline model only

---

## 10. Recommended Next Steps

- Stratified cross-validation
- Feature engineering (Title extraction, FamilySize, CabinDeck)
- Model comparison (tree-based models)
- Unit tests and CI integration
