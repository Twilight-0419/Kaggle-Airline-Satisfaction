from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "public_stack"
TARGET = "satisfaction"


DEFINITIONS = [
    ("base", 4, 50, 5.0, 0.7),
    ("shallow", 3, 100, 10.0, 0.5),
    ("mid", 5, 100, 10.0, 0.5),
    ("shallow_loose", 3, 30, 1.0, 0.7),
    ("mid_loose", 5, 30, 1.0, 0.7),
]


def main():
    X = np.load(OUTPUT_DIR / "X_oof.npy", mmap_mode="r")
    y = pd.read_csv(DATA_DIR / "train.csv", usecols=[TARGET])[TARGET].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = np.zeros(len(y), dtype=int)
    for fold, (_, valid_idx) in enumerate(cv.split(np.zeros(len(y)), y)):
        folds[valid_idx] = fold
    train_mask = folds != 0
    valid_mask = folds == 0
    results = []
    for name, depth, min_child, reg_lambda, colsample in DEFINITIONS:
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=depth,
            min_child_weight=min_child,
            subsample=0.8,
            colsample_bytree=colsample,
            reg_lambda=reg_lambda,
            early_stopping_rounds=100,
            tree_method="hist",
            device="cpu",
            n_jobs=8,
            random_state=42,
        )
        model.fit(
            X[train_mask],
            y[train_mask],
            eval_set=[(X[valid_mask], y[valid_mask])],
            verbose=False,
        )
        prediction = model.predict_proba(X[valid_mask])[:, 1]
        row = {
            "name": name,
            "auc": roc_auc_score(y[valid_mask], prediction),
            "iteration": int(model.best_iteration),
            "depth": depth,
            "min_child": min_child,
            "reg_lambda": reg_lambda,
            "colsample": colsample,
        }
        results.append(row)
        print(row, flush=True)
    pd.DataFrame(results).sort_values("auc", ascending=False).to_csv(
        OUTPUT_DIR / "xgb_meta_screen.csv", index=False
    )


if __name__ == "__main__":
    main()
