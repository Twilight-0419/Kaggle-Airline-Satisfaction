from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "public_stack"
TARGET = "satisfaction"


DEFINITIONS = [
    ("base", 31, 100, 5.0, 0.7),
    ("small", 15, 50, 2.0, 0.7),
    ("large", 63, 100, 5.0, 0.7),
    ("xlarge_reg", 127, 200, 20.0, 0.5),
    ("strong_reg", 31, 200, 20.0, 0.5),
    ("large_low_mcw", 63, 50, 2.0, 0.5),
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
    for name, leaves, min_child, reg_lambda, colsample in DEFINITIONS:
        model = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=leaves,
            min_child_samples=min_child,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=colsample,
            reg_lambda=reg_lambda,
            n_jobs=8,
            random_state=42,
            verbosity=-1,
        )
        model.fit(
            X[train_mask],
            y[train_mask],
            eval_set=[(X[valid_mask], y[valid_mask])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        prediction = model.predict_proba(X[valid_mask])[:, 1]
        row = {
            "name": name,
            "auc": roc_auc_score(y[valid_mask], prediction),
            "best_iteration": int(model.best_iteration_),
            "leaves": leaves,
            "min_child": min_child,
            "reg_lambda": reg_lambda,
            "colsample": colsample,
        }
        results.append(row)
        print(row, flush=True)

    pd.DataFrame(results).sort_values("auc", ascending=False).to_csv(
        OUTPUT_DIR / "lgbm_meta_screen.csv", index=False
    )


if __name__ == "__main__":
    main()
