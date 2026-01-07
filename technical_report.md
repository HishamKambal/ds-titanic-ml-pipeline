TECHNICAL_REPORT.md


## Quickstart (Windows CMD)
```bat
cd ds-titanic-ml-pipeline
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m src.train --train data/raw/train.csv --test data/raw/test.csv --out outputs

Outputs

outputs/metrics.json — evaluation metrics on validation split

outputs/model.joblib — trained sklearn pipeline

outputs/submission.csv — Kaggle-ready submission file

Notes

This baseline prioritizes clean structure and reproducibility over leaderboard optimization.

Improvements are listed in TECHNICAL_REPORT.md.


## New Technical Report (add `TECHNICAL_REPORT.md`)
```md
# Technical Report — Titanic Survival Prediction (ML Pipeline)

## 1. Objective
Predict passenger survival on the Titanic using the Kaggle dataset. The goal of this repository is to present an end-to-end ML pipeline with clean engineering practices (reproducible runs, clear outputs, and maintainable structure).

## 2. Approach
### 2.1 Data inputs
- Training data: `train.csv` containing features + target `Survived`
- Test data: `test.csv` containing features only

### 2.2 Features used
A compact, defensible feature set:
- Numeric: `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`
- Categorical: `Sex`, `Embarked`

Rationale: these features are well-known high-signal predictors for Titanic and allow a clean baseline without heavy feature engineering.

### 2.3 Preprocessing
- Numeric: median imputation
- Categorical: most-frequent imputation + one-hot encoding (handle unknown categories)

Implementation: `sklearn.compose.ColumnTransformer` inside an sklearn `Pipeline`.

### 2.4 Model choice
- Logistic Regression baseline
Rationale: strong baseline, interpretable coefficients, stable training, low risk of leakage and overfit in a small dataset.

## 3. Evaluation
- Validation approach: train/validation split
- Metric: accuracy + classification report summary

The goal is consistent and reproducible evaluation rather than leaderboard maximization.

## 4. Artifacts
- `model.joblib`: persisted sklearn pipeline
- `metrics.json`: accuracy + classification report
- `submission.csv`: Kaggle submission format

## 5. Key Engineering Decisions
1) Pipeline-based preprocessing to prevent training/serving skew  
2) Robust handling for missing values  
3) One-command run producing deterministic artifacts

## 6. Limitations
- Simple baseline model, minimal feature engineering
- Single validation split (no cross-validation)
- No hyperparameter search

## 7. Recommended next upgrades
- Stratified split with fixed random seed
- Cross-validation and calibration
- Add engineered features (Title from Name, FamilySize, Cabin indicator)
- Add unit tests and CI linting