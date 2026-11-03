from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "public_stack"
TARGET = "satisfaction"


def main():
    X = np.load(OUTPUT_DIR / "X_oof.npy", mmap_mode="r")
    train = pd.read_csv(DATA_DIR / "train.csv", usecols=[TARGET])
    y = train[TARGET].astype(int).to_numpy()
    table = pd.read_csv(OUTPUT_DIR / "model_table.csv").sort_values(
        "oof_auc", ascending=False
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = np.zeros(len(y), dtype=int)
    for fold, (_, valid_idx) in enumerate(cv.split(np.zeros(len(y)), y)):
        folds[valid_idx] = fold

    train_mask = folds != 0
    valid_mask = folds == 0
    results = []

    for c in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        model = LogisticRegression(
            penalty="l2",
            C=c,
            solver="lbfgs",
            max_iter=300,
            tol=1e-4,
        )
        model.fit(X[train_mask], y[train_mask])
        prediction = model.predict_proba(X[valid_mask])[:, 1]
        auc = roc_auc_score(y[valid_mask], prediction)
        results.append({"kind": "C", "c": c, "top_n": X.shape[1], "auc": auc})
        print(results[-1], flush=True)

    for top_n in [10, 20, 40, 60, 80, 100, 135]:
        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=300,
            tol=1e-4,
        )
        model.fit(X[train_mask, :top_n], y[train_mask])
        prediction = model.predict_proba(X[valid_mask, :top_n])[:, 1]
        auc = roc_auc_score(y[valid_mask], prediction)
        results.append({"kind": "top_n", "c": 1.0, "top_n": top_n, "auc": auc})
        print(results[-1], flush=True)

    pd.DataFrame(results).sort_values("auc", ascending=False).to_csv(
        OUTPUT_DIR / "logistic_screen.csv", index=False
    )


if __name__ == "__main__":
    main()
