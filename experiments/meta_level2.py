from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "public_stack"
PUBLIC_DIR = PROJECT_DIR / "某个公开算法"
TARGET = "satisfaction"


def rank_norm(values):
    return rankdata(values, method="average") / len(values)


def main():
    train = pd.read_csv(DATA_DIR / "train.csv", usecols=["id", TARGET])
    test = pd.read_csv(DATA_DIR / "test.csv", usecols=["id"])
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    y = train[TARGET].astype(int).to_numpy()

    lr_oof = np.load(OUTPUT_DIR / "stack_oof.npy")
    lr_test = np.load(OUTPUT_DIR / "stack_test.npy")
    lgb_oof = np.load(OUTPUT_DIR / "lightgbm_meta_oof.npy")
    lgb_test = np.load(OUTPUT_DIR / "lightgbm_meta_test.npy")
    xgb_oof = np.load(OUTPUT_DIR / "xgboost_meta_oof.npy")
    xgb_test = np.load(OUTPUT_DIR / "xgboost_meta_test.npy")
    root = PUBLIC_DIR / "S6E10 124 Public OOF and Test Predictions"
    top_oof = np.load(root / "oof_nj_01.npy")
    top_test = np.load(root / "test_nj_01.npy")

    X_oof = np.column_stack(
        [rank_norm(v) for v in [lr_oof, lgb_oof, xgb_oof, top_oof]]
    ).astype(np.float32)
    X_test = np.column_stack(
        [rank_norm(v) for v in [lr_test, lgb_test, xgb_test, top_test]]
    ).astype(np.float32)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best = None
    for c in [0.03, 0.1, 0.3, 1.0]:
        oof = np.zeros(len(train))
        test_pred = np.zeros(len(test))
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X_oof, y), 1):
            model = LogisticRegression(
                penalty="l2",
                C=c,
                solver="lbfgs",
                max_iter=500,
                tol=1e-6,
            )
            model.fit(X_oof[train_idx], y[train_idx])
            oof[valid_idx] = model.predict_proba(X_oof[valid_idx])[:, 1]
            test_pred += model.predict_proba(X_test)[:, 1] / 5
            print(f"c={c} fold={fold} auc={roc_auc_score(y[valid_idx], oof[valid_idx]):.6f}", flush=True)
        auc = roc_auc_score(y, oof)
        print("c", c, "OOF", auc, flush=True)
        if best is None or auc > best[0]:
            best = (auc, c, oof, test_pred)

    auc, c, oof, test_pred = best
    np.save(OUTPUT_DIR / "level2_oof.npy", oof)
    np.save(OUTPUT_DIR / "level2_test.npy", test_pred)
    submission = pd.DataFrame(
        {"id": sample["id"].to_numpy(), "satisfaction": np.clip(test_pred, 0, 1)}
    )
    assert np.array_equal(submission["id"].to_numpy(), sample["id"].to_numpy())
    submission.to_csv(OUTPUT_DIR / "submission_level2.csv", index=False)
    print("best", auc, c, "saved", OUTPUT_DIR / "submission_level2.csv")


if __name__ == "__main__":
    main()
