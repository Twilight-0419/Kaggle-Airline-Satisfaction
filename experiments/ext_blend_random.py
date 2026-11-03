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
    ext_l2_oof = np.load(OUTPUT_DIR / "ext_l2_oof.npy")
    ext_l2_test = np.load(OUTPUT_DIR / "ext_l2_test.npy")
    ext_l1_oof = np.load(OUTPUT_DIR / "ext_l1_oof.npy")
    ext_l1_test = np.load(OUTPUT_DIR / "ext_l1_test.npy")

    def load_pair(stem):
        o = pd.read_csv(EXTRA_DIR / "s6e10-oof" / f"{stem}_blend_oof.csv").set_index(ID)["satisfaction"].reindex(train_ids).to_numpy()
        t = pd.read_csv(EXTRA_DIR / "s6e10-oof" / f"{stem}_submission.csv").set_index(ID)["satisfaction"].reindex(test_ids).to_numpy()
        return o, t

    o5, t5 = load_pair("05")
    o6, t6 = load_pair("06")
    oof = np.column_stack([rank_norm(v) for v in [ext_l2_oof, ext_l1_oof, o5, o6]])
    test = np.column_stack([rank_norm(v) for v in [ext_l2_test, ext_l1_test, t5, t6]])
    rng = np.random.default_rng(2026)
    weights = rng.dirichlet([12.0, 2.0, 4.0, 3.0], size=1200)
    weights = np.vstack([weights, np.array([.88, 0, .12, 0])])
    best_auc = -1
    best_w = None
    for w in weights:
        auc = roc_auc_score(y, oof @ w)
        if auc > best_auc:
            best_auc = float(auc)
            best_w = w
    print("best", best_auc, best_w)
    submission = pd.DataFrame(
        {ID: sample[ID].to_numpy(), TARGET: np.clip(test @ best_w, 0, 1)}
    )
    submission.to_csv(OUTPUT_DIR / "ext_random_blend_submission.csv", index=False)


if __name__ == "__main__":
    main()
