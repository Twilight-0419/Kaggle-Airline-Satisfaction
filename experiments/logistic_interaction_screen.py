from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from itertools import combinations
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

    for top_n in [5, 10, 15, 20]:
        base = np.asarray(X[:, :top_n], dtype=np.float32)
        pairs = list(combinations(range(top_n), 2))
        interactions = np.column_stack(
            [base[:, i] * base[:, j] for i, j in pairs]
        ).astype(np.float32)
        matrix = np.column_stack([base, interactions])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(matrix[train_mask]).astype(np.float32)
        valid = scaler.transform(matrix[valid_mask]).astype(np.float32)
        for c in [0.01, 0.03, 0.1]:
            model = LogisticRegression(
                penalty="l2",
                C=c,
                solver="lbfgs",
                max_iter=500,
                tol=1e-4,
            )
            model.fit(scaled, y[train_mask])
            prediction = model.predict_proba(valid)[:, 1]
            print(
                "top", top_n, "features", matrix.shape[1], "c", c,
                "auc", roc_auc_score(y[valid_mask], prediction),
                flush=True,
            )


if __name__ == "__main__":
    main()
