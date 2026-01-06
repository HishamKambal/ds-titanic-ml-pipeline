\
import argparse
import json
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class Paths:
    train_csv: str
    test_csv: str
    out_dir: str


def build_pipeline(num_cols, cat_cols) -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(max_iter=500)

    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def main(p: Paths) -> int:
    os.makedirs(p.out_dir, exist_ok=True)

    train = pd.read_csv(p.train_csv)
    test = pd.read_csv(p.test_csv)

    if "Survived" not in train.columns:
        raise ValueError("train.csv must contain a 'Survived' column (Kaggle Titanic).")

    y = train["Survived"].astype(int)
    X = train.drop(columns=["Survived"])

    # Standard, reasonable feature set for a clean baseline
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    missing = [c for c in features if c not in X.columns]
    if missing:
        raise ValueError(f"Missing expected columns in train.csv: {missing}")

    X = X[features].copy()
    X_test = test[features].copy()

    num_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    cat_cols = ["Sex", "Embarked"]

    pipe = build_pipeline(num_cols=num_cols, cat_cols=cat_cols)

    # quick internal validation split (not Kaggle-optimized; meant to show workflow)
    rng = np.random.default_rng(42)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(len(X) * 0.8)
    tr_idx, va_idx = idx[:split], idx[split:]

    pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    pred = pipe.predict(X.iloc[va_idx])

    acc = float(accuracy_score(y.iloc[va_idx], pred))
    report = classification_report(y.iloc[va_idx], pred, output_dict=True)

    metrics_path = os.path.join(p.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=2)

    model_path = os.path.join(p.out_dir, "model.joblib")
    joblib.dump(pipe, model_path)

    # Train full model and generate submission
    pipe.fit(X, y)
    test_pred = pipe.predict(X_test)

    if "PassengerId" not in test.columns:
        raise ValueError("test.csv must contain PassengerId column (Kaggle Titanic).")

    sub = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_pred.astype(int)})
    sub_path = os.path.join(p.out_dir, "submission.csv")
    sub.to_csv(sub_path, index=False)

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved model:   {model_path}")
    print(f"Saved submit:  {sub_path}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to Kaggle Titanic train.csv")
    ap.add_argument("--test", required=True, help="Path to Kaggle Titanic test.csv")
    ap.add_argument("--out", default="outputs", help="Output directory")
    args = ap.parse_args()

    raise SystemExit(main(Paths(args.train, args.test, args.out)))
