from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
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
    names = pd.read_csv(OUTPUT_DIR / "model_names.csv")["model"].tolist()
    short = [name.split(":", 1)[1] for name in names]
    groups = [name.split("_")[0] for name in short]
    unique = sorted(set(groups))
    columns = []
    for group in unique:
        idx = [i for i, value in enumerate(groups) if value == group]
        columns.append(np.asarray(X[:, idx]).mean(axis=1))
    G = np.column_stack(columns).astype(np.float32)
    print("groups", len(unique), unique)

    y = pd.read_csv(DATA_DIR / "train.csv", usecols=[TARGET])[TARGET].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = np.zeros(len(y), dtype=int)
    for fold, (_, valid_idx) in enumerate(cv.split(G, y)):
        folds[valid_idx] = fold
    tr = folds != 0
    va = folds == 0
    for scaled in [False, True]:
        matrix = G
        if scaled:
            scaler = StandardScaler()
            train_matrix = scaler.fit_transform(G[tr])
            valid_matrix = scaler.transform(G[va])
        else:
            train_matrix, valid_matrix = G[tr], G[va]
        for c in [0.01, 0.03, 0.1, 0.3, 1.0]:
            model = LogisticRegression(
                penalty="l2",
                C=c,
                solver="lbfgs",
                max_iter=500,
                tol=1e-4,
            )
            model.fit(train_matrix, y[tr])
            p = model.predict_proba(valid_matrix)[:, 1]
            print("scaled", scaled, "c", c, "auc", roc_auc_score(y[va], p))


if __name__ == "__main__":
    main()
