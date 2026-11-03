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
    X_test = np.load(OUTPUT_DIR / "X_test.npy", mmap_mode="r")
    names = pd.read_csv(OUTPUT_DIR / "model_names.csv")["model"].tolist()
    indices = [i for i, name in enumerate(names) if "TFM OOF" in name]
    print("TFM models", len(indices), [names[i] for i in indices])
    y = pd.read_csv(DATA_DIR / "train.csv", usecols=[TARGET])[TARGET].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best = None
    for c in [0.1, 0.3, 1.0, 3.0]:
        oof = np.zeros(len(y))
        test_pred = np.zeros(X_test.shape[0])
        for train_idx, valid_idx in cv.split(X, y):
            model = LogisticRegression(
                penalty="l2",
                C=c,
                solver="lbfgs",
                max_iter=500,
                tol=1e-5,
            )
            X_tr = np.asarray(X[train_idx][:, indices])
            X_va = np.asarray(X[valid_idx][:, indices])
            X_te = np.asarray(X_test[:, indices])
            model.fit(X_tr, y[train_idx])
            oof[valid_idx] = model.predict_proba(X_va)[:, 1]
            test_pred += model.predict_proba(X_te)[:, 1] / 5
        auc = roc_auc_score(y, oof)
        print("c", c, "OOF", auc, flush=True)
        if best is None or auc > best[0]:
            best = (auc, c, oof, test_pred)

    auc, c, oof, test_pred = best
    np.save(OUTPUT_DIR / "tfm_stack_oof.npy", oof)
    np.save(OUTPUT_DIR / "tfm_stack_test.npy", test_pred)
    print("best TFM stack", auc, c)


if __name__ == "__main__":
    main()
