from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
PUBLIC_OUTPUT = PROJECT_DIR / "outputs" / "public_stack"
LOCAL_OUTPUT = PROJECT_DIR / "outputs" / "ensemble"
TARGET = "satisfaction"
ID = "id"


def rank_norm(values):
    return rankdata(values, method="average") / len(values)


def main():
    train = pd.read_csv(DATA_DIR / "train.csv", usecols=[ID, TARGET])
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    public_oof = pd.read_csv(PUBLIC_OUTPUT / "oof_best.csv")
    public_test = pd.read_csv(PUBLIC_OUTPUT / "submission_best.csv")

    oof = train.merge(
        public_oof.rename(columns={TARGET: "public_stack"}),
        on=ID,
        how="inner",
    )
    local_names = ["xgboost", "lightgbm", "catboost"]
    for name in local_names:
        local_oof = pd.read_csv(LOCAL_OUTPUT / f"{name}_oof.csv")
        oof = oof.merge(
            local_oof[[ID, TARGET]].rename(columns={TARGET: name}),
            on=ID,
            how="inner",
        )

    test = sample[[ID]].copy()
    test["public_stack"] = public_test[TARGET].to_numpy()
    for name in local_names:
        local_test = pd.read_csv(LOCAL_OUTPUT / f"{name}_test.csv")
        test = test.merge(
            local_test[[ID, TARGET]].rename(columns={TARGET: name}),
            on=ID,
            how="inner",
        )

    y = oof[TARGET].astype(int).to_numpy()
    oof_matrix = np.column_stack(
        [rank_norm(oof[name].to_numpy()) for name in ["public_stack"] + local_names]
    )
    test_matrix = np.column_stack(
        [rank_norm(test[name].to_numpy()) for name in ["public_stack"] + local_names]
    )

    print("merged rows", len(oof), "test", len(test))
    for index, name in enumerate(["public_stack"] + local_names):
        print(name, roc_auc_score(y, oof_matrix[:, index]))

    best_auc = -1.0
    best_weights = None
    for w0 in np.arange(0.0, 1.0001, 0.05):
        for w1 in np.arange(0.0, 1.0001 - w0, 0.05):
            for w2 in np.arange(0.0, 1.0001 - w0 - w1, 0.05):
                w3 = 1.0 - w0 - w1 - w2
                weights = np.array([w0, w1, w2, w3], dtype=np.float64)
                auc = roc_auc_score(y, oof_matrix @ weights)
                if auc > best_auc:
                    best_auc = float(auc)
                    best_weights = weights

    print("best AUC", best_auc, "weights", best_weights)
    submission = pd.DataFrame(
        {
            ID: test[ID].to_numpy(),
            TARGET: np.clip(test_matrix @ best_weights, 0, 1),
        }
    )
    assert np.array_equal(
        submission[ID].to_numpy(), sample[ID].to_numpy()
    )
    submission.to_csv(PUBLIC_OUTPUT / "submission_public_local_blend.csv", index=False)
    print("saved", PUBLIC_OUTPUT / "submission_public_local_blend.csv")


if __name__ == "__main__":
    main()
