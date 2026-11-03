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
    y = pd.read_csv(DATA_DIR / "train.csv", usecols=[TARGET])[TARGET].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = np.zeros(len(y), dtype=int)
    for fold, (_, valid_idx) in enumerate(cv.split(np.zeros(len(y)), y)):
        folds[valid_idx] = fold
    train_mask = folds != 0
    valid_mask = folds == 0
    results = []

    candidates = [
        ("newton", "l2", 0.03),
        ("newton", "l2", 0.1),
        ("newton", "l2", 1.0),
        ("liblinear_l1", "l1", 0.03),
        ("liblinear_l1", "l1", 0.1),
        ("liblinear_l1", "l1", 1.0),
    ]
    for name, penalty, c in candidates:
        kwargs = dict(
            penalty=penalty,
            C=c,
            max_iter=1000,
            tol=1e-5,
        )
        if name == "newton":
            kwargs["solver"] = "newton-cholesky"
        else:
            kwargs["solver"] = "liblinear"
        model = LogisticRegression(**kwargs)
        model.fit(X[train_mask], y[train_mask])
        prediction = model.predict_proba(X[valid_mask])[:, 1]
        row = {
            "name": name,
            "c": c,
            "auc": roc_auc_score(y[valid_mask], prediction),
            "nonzero": int(np.count_nonzero(model.coef_)),
        }
        results.append(row)
        print(row, flush=True)

    pd.DataFrame(results).sort_values("auc", ascending=False).to_csv(
        OUTPUT_DIR / "logistic_solver_screen.csv", index=False
    )


if __name__ == "__main__":
    main()
