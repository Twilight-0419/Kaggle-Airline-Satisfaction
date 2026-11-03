from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    X_rank = np.empty_like(X, dtype=np.float32)
    for column in range(X.shape[1]):
        X_rank[:, column] = rankdata(X[:, column], method="average") / len(X)

    results = []
    candidates = [
        ("scaled", X_scaled, 1.0),
        ("scaled", X_scaled, 0.03),
        ("rank", X_rank, 1.0),
        ("rank", X_rank, 0.03),
        ("rank_top60", X_rank[:, :60], 1.0),
        ("rank_top60", X_rank[:, :60], 0.03),
    ]
    for name, matrix, c in candidates:
        model = LogisticRegression(
            penalty="l2",
            C=c,
            solver="lbfgs",
            max_iter=300,
            tol=1e-4,
        )
        model.fit(matrix[train_mask], y[train_mask])
        prediction = model.predict_proba(matrix[valid_mask])[:, 1]
        row = {
            "transform": name,
            "c": c,
            "auc": roc_auc_score(y[valid_mask], prediction),
        }
        results.append(row)
        print(row, flush=True)

    pd.DataFrame(results).sort_values("auc", ascending=False).to_csv(
        OUTPUT_DIR / "logistic_transform_screen.csv", index=False
    )


if __name__ == "__main__":
    main()
