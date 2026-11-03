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
    X_base = np.load(OUTPUT_DIR / "X_oof_ext.npy", mmap_mode="r")
    X_all = np.load(OUTPUT_DIR / "X_oof_ext2.npy", mmap_mode="r")
    names_base = pd.read_csv(OUTPUT_DIR / "model_names_ext.csv")["model"].tolist()
    names_all = pd.read_csv(OUTPUT_DIR / "model_names_ext2.csv")["model"].tolist()
    table = pd.read_csv(OUTPUT_DIR / "model_table_ext2.csv")
    new_names = [name for name in names_all if name.startswith(("sadam_", "sachith_"))]
    ranked = [name for name in table["model"] if name in new_names]
    new_index = {name: i for i, name in enumerate(names_all)}
    y = pd.read_csv(DATA_DIR / "train.csv", usecols=[TARGET])[TARGET].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = np.zeros(len(y), dtype=int)
    for fold, (_, valid_idx) in enumerate(cv.split(X_all, y)):
        folds[valid_idx] = fold
    tr = folds != 0
    va = folds == 0
    for k in [1, 3, 5, 10, 20, 40]:
        selected = ranked[:k]
        matrix = np.column_stack(
            [X_base, X_all[:, [new_index[name] for name in selected]]]
        )
        model = LogisticRegression(
            penalty="l2",
            C=0.03,
            solver="lbfgs",
            max_iter=500,
            tol=1e-4,
        )
        model.fit(matrix[tr], y[tr])
        p = model.predict_proba(matrix[va])[:, 1]
        print("k", k, "auc", roc_auc_score(y[va], p), selected, flush=True)


if __name__ == "__main__":
    main()
