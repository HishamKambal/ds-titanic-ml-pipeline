import argparse
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class Config:
    features: List[str]
    num_cols: List[str]
    cat_cols: List[str]
    test_size: float
    seed: int


DEFAULT_FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
DEFAULT_NUM = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
DEFAULT_CAT = ["Sex", "Embarked"]


def build_pipeline(num_cols, cat_cols) -> Pipeline:
    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
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


def require_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def make_run_dir(out_root: str) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out_root) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(train_csv: str, test_csv: str, out_dir: str, test_size: float, seed: int) -> int:
    train_path = Path(train_csv)
    test_path = Path(test_csv)

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if "Survived" not in train.columns:
        raise ValueError("train.csv must contain a 'Survived' column (Kaggle Titanic).")
    if "PassengerId" not in test.columns:
        raise ValueError("test.csv must contain 'PassengerId' column (Kaggle Titanic).")

    cfg = Config(
        features=DEFAULT_FEATURES,
        num_cols=DEFAULT_NUM,
        cat_cols=DEFAULT_CAT,
        test_size=test_size,
        seed=seed,
    )

    # Validate columns in BOTH files
    require_columns(train, cfg.features, "train.csv")
    require_columns(test, cfg.features, "test.csv")

    y = train["Survived"].astype(int)
    X = train[cfg.features].copy()
    X_test = test[cfg.features].copy()

    pipe = build_pipeline(num_cols=cfg.num_cols, cat_cols=cfg.cat_cols)

    # Professional validation split: deterministic + stratified
    X_tr, X_va, y_tr, y_va = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=y,
    )

    pipe.fit(X_tr, y_tr)
    va_pred = pipe.predict(X_va)

    acc = float(accuracy_score(y_va, va_pred))
    report = classification_report(y_va, va_pred, output_dict=True)

    run_dir = make_run_dir(out_dir)

    save_json(
        run_dir / "metrics.json",
        {
            "accuracy": acc,
            "classification_report": report,
            "validation": {"test_size": cfg.test_size, "seed": cfg.seed, "stratify": True},
            "features": cfg.features,
        },
    )

    # Save model trained on FULL train set
    pipe.fit(X, y)
    joblib.dump(pipe, run_dir / "model.joblib")

    test_pred = pipe.predict(X_test).astype(int)
    sub = pd.DataFrame({"PassengerId": test["PassengerId"].astype(int), "Survived": test_pred})
    sub.to_csv(run_dir / "submission.csv", index=False)

    # Pro: run metadata
    save_json(
        run_dir / "run_metadata.json",
        {
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "out_dir": str(run_dir),
        },
    )

    print(f"[OK] Run dir:    {run_dir}")
    print(f"[OK] Metrics:    {run_dir / 'metrics.json'}")
    print(f"[OK] Model:      {run_dir / 'model.joblib'}")
    print(f"[OK] Submission: {run_dir / 'submission.csv'}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to Kaggle Titanic train.csv")
    ap.add_argument("--test", required=True, help="Path to Kaggle Titanic test.csv")
    ap.add_argument("--out", default="outputs", help="Output directory root")
    ap.add_argument("--test-size", type=float, default=0.2, help="Validation split fraction")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    raise SystemExit(main(args.train, args.test, args.out, args.test_size, args.seed))
