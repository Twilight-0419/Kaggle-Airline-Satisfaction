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
    for c in [0.005, 0.01, 0.02, 0.04, 0.05]:
        model = LogisticRegression(
            penalty="l1",
            C=c,
            solver="liblinear",
            max_iter=1000,
            tol=1e-5,
        )
        model.fit(X[train_mask], y[train_mask])
        prediction = model.predict_proba(X[valid_mask])[:, 1]
        row = {
            "c": c,
            "auc": roc_auc_score(y[valid_mask], prediction),
            "nonzero": int(np.count_nonzero(model.coef_)),
        }
        results.append(row)
        print(row, flush=True)
    pd.DataFrame(results).sort_values("auc", ascending=False).to_csv(
        OUTPUT_DIR / "l1_c_fine_screen.csv", index=False
    )


if __name__ == "__main__":
    main()
