from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "public_stack"
TARGET = "satisfaction"


def rank_norm(values):
    return rankdata(values, method="average") / len(values)


def main():
    y = pd.read_csv(DATA_DIR / "train.csv", usecols=[TARGET])[TARGET].astype(int).to_numpy()
    l1 = np.load(OUTPUT_DIR / "l1_stack_oof.npy")
    l2 = np.load(OUTPUT_DIR / "stack_oof.npy")
    xgb = np.load(OUTPUT_DIR / "xgboost_meta_oof.npy")
    X = np.column_stack([rank_norm(v) for v in [l1, l2, xgb]])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rng = np.random.default_rng(2026)
    fixed = np.array([0.65, 0.15, 0.20])
    held_fixed = []
    held_tuned = []
    fold_weights = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), 1):
        sample_size = min(120000, len(train_idx))
        sample_idx = rng.choice(train_idx, size=sample_size, replace=False)
        candidates = rng.dirichlet([10.0, 3.0, 4.0], size=600)
        candidates = np.vstack([candidates, fixed])
        best_auc = -1.0
        best_weight = fixed
        for weight in candidates:
            auc = roc_auc_score(y[sample_idx], X[sample_idx] @ weight)
            if auc > best_auc:
                best_auc = auc
                best_weight = weight

        fixed_auc = roc_auc_score(y[valid_idx], X[valid_idx] @ fixed)
        tuned_auc = roc_auc_score(y[valid_idx], X[valid_idx] @ best_weight)
        held_fixed.append(fixed_auc)
        held_tuned.append(tuned_auc)
        fold_weights.append(best_weight)
        print(
            f"fold={fold} weight={best_weight} "
            f"fixed={fixed_auc:.6f} tuned={tuned_auc:.6f}",
            flush=True,
        )

    mean_weight = np.mean(fold_weights, axis=0)
    print("fixed mean", np.mean(held_fixed))
    print("tuned mean", np.mean(held_tuned))
    print("mean weight", mean_weight)

    np.save(OUTPUT_DIR / "cv_mean_weights.npy", mean_weight)


if __name__ == "__main__":
    main()
