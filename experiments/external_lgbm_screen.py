from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
ORIGINAL_DIR = PROJECT_DIR / "某个公开算法" / "original_data"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "public_stack"
TARGET = "satisfaction"


def prepare(df):
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        if col != TARGET:
            out[col] = out[col].astype(str)
    return out


def main():
    comp = pd.read_csv(DATA_DIR / "train.csv")
    original = pd.read_csv(ORIGINAL_DIR / "train.csv")
    original[TARGET] = original[TARGET].map(
        {"neutral or dissatisfied": 0, "satisfied": 1}
    )

    y = comp[TARGET].astype(int).to_numpy()
    X_comp = comp.drop(columns=["id", TARGET])
    y_orig = original[TARGET].astype(int).to_numpy()
    X_orig = original.drop(
        columns=["Unnamed: 0", "id", "Inflight service", TARGET]
    )
    X_orig = X_orig[X_comp.columns]

    cat_cols = ["Gender", "Customer Type", "Type of Travel", "Class"]
    combined = pd.concat([X_comp, X_orig], ignore_index=True)
    combined = pd.get_dummies(combined, columns=cat_cols, dtype=np.int8)
    X_comp = combined.iloc[: len(X_comp)]
    X_orig = combined.iloc[len(X_comp):]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, valid_idx = next(cv.split(X_comp, y))
    X_train = pd.concat(
        [X_comp.iloc[train_idx], X_orig], ignore_index=True
    )
    y_train = np.concatenate([y[train_idx], y_orig])
    weights = np.concatenate(
        [np.ones(len(train_idx)), np.full(len(X_orig), 0.5)]
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=2500,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=2.0,
        n_jobs=8,
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=weights,
        eval_set=[(X_comp.iloc[valid_idx], y[valid_idx])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    prediction = model.predict_proba(X_comp.iloc[valid_idx])[:, 1]
    print(
        "external lgb fold0 AUC", roc_auc_score(y[valid_idx], prediction),
        "iter", model.best_iteration_,
        flush=True,
    )
    np.save(OUTPUT_DIR / "external_lgb_fold0.npy", prediction)


if __name__ == "__main__":
    main()
