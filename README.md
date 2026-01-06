# Titanic Survival Prediction (End-to-End ML Pipeline)

A compact, production-style ML project using the **Kaggle Titanic** dataset. It demonstrates:
- Reproducible project structure
- Data validation + preprocessing
- Model training + evaluation
- One-command runs

## Dataset
Download `train.csv` and `test.csv` from Kaggle (Titanic competition) and place them into:
`data/raw/`

Kaggle competition page: https://www.kaggle.com/competitions/titanic

## Quickstart (Windows CMD)
```bat
cd ds-titanic-ml-pipeline
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m src.train --train data/raw/train.csv --test data/raw/test.csv --out outputs
```

## Outputs
- `outputs/metrics.json`  (accuracy + classification report summary)
- `outputs/model.joblib`  (trained model)
- `outputs/submission.csv` (Kaggle-format predictions)

## Tech
Python, pandas, scikit-learn, joblib
