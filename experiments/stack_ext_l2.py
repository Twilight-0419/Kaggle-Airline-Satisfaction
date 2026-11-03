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
    X = np.load(OUTPUT_DIR / "X_oof_ext.npy", mmap_mode="r")
    X_test = np.load(OUTPUT_DIR / "X_test_ext.npy", mmap_mode="r")
    train = pd.read_csv(DATA_DIR / "train.csv", usecols=["id", TARGET])
    test = pd.read_csv(DATA_DIR / "test.csv", usecols=["id"])
    y = train[TARGET].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    coefs = []
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), 1):
        model = LogisticRegression(
            penalty="l2",
            C=0.03,
            solver="lbfgs",
            max_iter=500,
            tol=1e-5,
        )
        model.fit(X[train_idx], y[train_idx])
        oof[valid_idx] = model.predict_proba(X[valid_idx])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / 5
        coefs.append(np.ravel(model.coef_))
        print(
            f"fold={fold} auc={roc_auc_score(y[valid_idx], oof[valid_idx]):.6f}",
            flush=True,
        )
    auc = roc_auc_score(y, oof)
    print("extended L2 OOF", auc)
    np.save(OUTPUT_DIR / "ext_l2_oof.npy", oof)
    np.save(OUTPUT_DIR / "ext_l2_test.npy", test_pred)
    names = pd.read_csv(OUTPUT_DIR / "model_names_ext.csv")["model"].to_numpy()
    coef = np.mean(coefs, axis=0)
    pd.DataFrame(
        {"model": names, "coef": coef, "abs": np.abs(coef)}
    ).sort_values("abs", ascending=False).to_csv(
        OUTPUT_DIR / "ext_l2_coefficients.csv", index=False
    )


if __name__ == "__main__":
    main()
