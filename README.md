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

### What to do now
1. Open `README.md`
2. Replace everything with the content above
3. Save
4. Run:

```bat
git add README.md
git commit -m "Finalize README"
git push
