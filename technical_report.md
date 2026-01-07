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
- Model comparison with tree-based algorithms
- Unit tests and CI integration
Final step (commit & push)
bat
Copy code
cd /d C:\GitHub\portfolio\ds-titanic-ml-pipeline
git add TECHNICAL_REPORT.md
git commit -m "Add technical report"
git push
