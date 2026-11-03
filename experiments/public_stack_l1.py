from __future__ import annotations

import time
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
    X_oof = np.load(OUTPUT_DIR / "X_oof.npy", mmap_mode="r")
    X_test = np.load(OUTPUT_DIR / "X_test.npy", mmap_mode="r")
    train = pd.read_csv(DATA_DIR / "train.csv", usecols=["id", TARGET])
    test = pd.read_csv(DATA_DIR / "test.csv", usecols=["id"])
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    y = train[TARGET].astype(int).to_numpy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    coefs = []
    started = time.perf_counter()

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X_oof, y), 1):
        model = LogisticRegression(
            penalty="l1",
            C=0.025,
            solver="liblinear",
            max_iter=1000,
            tol=1e-5,
        )
        model.fit(X_oof[train_idx], y[train_idx])
        oof[valid_idx] = model.predict_proba(X_oof[valid_idx])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / 5
        coefs.append(np.ravel(model.coef_))
        print(
            f"fold={fold} auc={roc_auc_score(y[valid_idx], oof[valid_idx]):.6f} "
            f"nonzero={np.count_nonzero(model.coef_)}",
            flush=True,
        )

    auc = roc_auc_score(y, oof)
    print("L1 stack OOF", auc, "seconds", time.perf_counter() - started)
    np.save(OUTPUT_DIR / "l1_stack_oof.npy", oof)
    np.save(OUTPUT_DIR / "l1_stack_test.npy", test_pred)
    submission = pd.DataFrame(
        {
            "id": sample["id"].to_numpy(),
            "satisfaction": np.clip(test_pred, 0, 1),
        }
    )
    assert np.array_equal(submission["id"].to_numpy(), sample["id"].to_numpy())
    submission.to_csv(OUTPUT_DIR / "l1_stack_submission.csv", index=False)

    names = pd.read_csv(OUTPUT_DIR / "model_names.csv")["model"].to_numpy()
    coef = np.mean(coefs, axis=0)
    pd.DataFrame(
        {
            "model": names,
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
        }
    ).sort_values("abs_coefficient", ascending=False).to_csv(
        OUTPUT_DIR / "l1_coefficients.csv", index=False
    )
    print("saved", OUTPUT_DIR / "l1_stack_submission.csv")


if __name__ == "__main__":
    main()
