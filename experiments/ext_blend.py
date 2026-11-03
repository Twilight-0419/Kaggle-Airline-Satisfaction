from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "public_stack"
EXTRA_DIR = PROJECT_DIR / "某个公开算法" / "extra_oof"
TARGET = "satisfaction"
ID = "id"


def rank_norm(values):
    return rankdata(values, method="average") / len(values)


def main():
    train_ids = pd.read_csv(DATA_DIR / "train.csv", usecols=[ID])[ID]
    test_ids = pd.read_csv(DATA_DIR / "test.csv", usecols=[ID])[ID]
    y = pd.read_csv(DATA_DIR / "train.csv", usecols=[TARGET])[TARGET].astype(int).to_numpy()
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    top_oof_df = pd.read_csv(EXTRA_DIR / "s6e10-oof" / "05_blend_oof.csv")
    top_test_df = pd.read_csv(EXTRA_DIR / "s6e10-oof" / "05_submission.csv")
    top_oof = top_oof_df.set_index(ID)["satisfaction"].reindex(train_ids).to_numpy()
    top_test = top_test_df.set_index(ID)["satisfaction"].reindex(test_ids).to_numpy()

    ext_l2_oof = np.load(OUTPUT_DIR / "ext_l2_oof.npy")
    ext_l2_test = np.load(OUTPUT_DIR / "ext_l2_test.npy")
    ext_l1_oof = np.load(OUTPUT_DIR / "ext_l1_oof.npy")
    ext_l1_test = np.load(OUTPUT_DIR / "ext_l1_test.npy")

    oof_matrix = np.column_stack(
        [rank_norm(v) for v in [ext_l2_oof, ext_l1_oof, top_oof]]
    )
    test_matrix = np.column_stack(
        [rank_norm(v) for v in [ext_l2_test, ext_l1_test, top_test]]
    )
    for i, name in enumerate(["ext_l2", "ext_l1", "najiama_05"]):
        print(name, roc_auc_score(y, oof_matrix[:, i]))

    best_auc = -1.0
    best_weights = None
    for w0 in np.arange(0.0, 1.0001, 0.02):
        for w1 in np.arange(0.0, 1.0001 - w0, 0.02):
            w2 = 1.0 - w0 - w1
            weights = np.array([w0, w1, w2])
            auc = roc_auc_score(y, oof_matrix @ weights)
            if auc > best_auc:
                best_auc = float(auc)
                best_weights = weights
    print("best", best_auc, best_weights)

    submission = pd.DataFrame(
        {
            ID: sample[ID].to_numpy(),
            TARGET: np.clip(test_matrix @ best_weights, 0, 1),
        }
    )
    assert np.array_equal(submission[ID].to_numpy(), sample[ID].to_numpy())
    submission.to_csv(OUTPUT_DIR / "ext_best_submission.csv", index=False)
    print("saved", OUTPUT_DIR / "ext_best_submission.csv")


if __name__ == "__main__":
    main()
