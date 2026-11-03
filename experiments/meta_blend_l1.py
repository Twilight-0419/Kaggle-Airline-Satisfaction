from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "public_stack"
PUBLIC_DIR = PROJECT_DIR / "某个公开算法"
TARGET = "satisfaction"


def rank_norm(values):
    return rankdata(values, method="average") / len(values)


def main():
    y = pd.read_csv(DATA_DIR / "train.csv", usecols=[TARGET])[TARGET].astype(int).to_numpy()
    l1_oof = np.load(OUTPUT_DIR / "l1_stack_oof.npy")
    l1_test = np.load(OUTPUT_DIR / "l1_stack_test.npy")
    lr_oof = np.load(OUTPUT_DIR / "stack_oof.npy")
    lr_test = np.load(OUTPUT_DIR / "stack_test.npy")
    xgb_oof = np.load(OUTPUT_DIR / "xgboost_meta_oof.npy")
    xgb_test = np.load(OUTPUT_DIR / "xgboost_meta_test.npy")
    root = PUBLIC_DIR / "S6E10 124 Public OOF and Test Predictions"
    top_oof = np.load(root / "oof_nj_01.npy")
    top_test = np.load(root / "test_nj_01.npy")

    oof_matrix = np.column_stack(
        [rank_norm(v) for v in [l1_oof, lr_oof, xgb_oof, top_oof]]
    )
    test_matrix = np.column_stack(
        [rank_norm(v) for v in [l1_test, lr_test, xgb_test, top_test]]
    )
    names = ["l1", "l2", "xgboost_meta", "top_single"]
    for i, name in enumerate(names):
        print(name, roc_auc_score(y, oof_matrix[:, i]))

    best_auc = -1.0
    best_weights = None
    for w0 in np.arange(0.0, 1.0001, 0.05):
        for w1 in np.arange(0.0, 1.0001 - w0, 0.05):
            for w2 in np.arange(0.0, 1.0001 - w0 - w1, 0.05):
                w3 = 1.0 - w0 - w1 - w2
                weights = np.array([w0, w1, w2, w3])
                auc = roc_auc_score(y, oof_matrix @ weights)
                if auc > best_auc:
                    best_auc = float(auc)
                    best_weights = weights

    print("best", best_auc, best_weights)
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    submission = pd.DataFrame(
        {
            "id": sample["id"].to_numpy(),
            "satisfaction": np.clip(test_matrix @ best_weights, 0, 1),
        }
    )
    submission.to_csv(OUTPUT_DIR / "submission_l1_meta_blend.csv", index=False)
    print("saved", OUTPUT_DIR / "submission_l1_meta_blend.csv")


if __name__ == "__main__":
    main()
