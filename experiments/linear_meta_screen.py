from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "public_stack"
TARGET = "satisfaction"


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

    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        model = make_pipeline(
            StandardScaler(),
            RidgeClassifier(alpha=alpha, solver="lsqr", tol=1e-4),
        )
        model.fit(X[train_mask], y[train_mask])
        score = model.decision_function(X[valid_mask])
        row = {
            "model": "ridge",
            "alpha": alpha,
            "auc": roc_auc_score(y[valid_mask], score),
        }
        results.append(row)
        print(row, flush=True)

    for alpha in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]:
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            learning_rate="optimal",
            max_iter=20,
            tol=1e-4,
            average=True,
            random_state=42,
        )
        model.fit(X[train_mask], y[train_mask])
        score = model.decision_function(X[valid_mask])
        row = {
            "model": "sgd_log",
            "alpha": alpha,
            "auc": roc_auc_score(y[valid_mask], score),
        }
        results.append(row)
        print(row, flush=True)

    pd.DataFrame(results).sort_values("auc", ascending=False).to_csv(
        OUTPUT_DIR / "linear_meta_screen.csv", index=False
    )


if __name__ == "__main__":
    main()
