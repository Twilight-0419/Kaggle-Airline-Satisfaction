from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
PUBLIC_DIR = PROJECT_DIR / "某个公开算法"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "public_stack"
TARGET = "satisfaction"


def norm_rank(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / len(values)


def find_dir(name: str) -> Path:
    return next(PUBLIC_DIR.glob(name))


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv", usecols=["id", TARGET])
    test = pd.read_csv(DATA_DIR / "test.csv", usecols=["id"])
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    y = train[TARGET].astype(int).to_numpy()

    table = pd.read_csv(OUTPUT_DIR / "model_table.csv")
    table = table.sort_values("oof_auc", ascending=False).reset_index(drop=True)
    dirs = {
        name: find_dir(name)
        for name in table["model"].str.split(":").str[0].unique()
    }

    oof_rank_sum = np.zeros(len(train), dtype=np.float64)
    test_rank_sum = np.zeros(len(test), dtype=np.float64)
    rank_rows = []
    checkpoint_oof = {}
    checkpoint_test = {}
    checkpoints = {1, 2, 3, 5, 10, 20, 40, 60, 80, 100, 120, len(table)}
    top_single_oof = None
    top_single_test = None

    for index, row in table.iterrows():
        directory_name, model_name = row["model"].split(":", 1)
        root = dirs[directory_name]
        oof = np.load(root / f"oof_{model_name}.npy")
        test_pred = np.load(root / f"test_{model_name}.npy")
        already_logit = bool(
            oof.min() < 0 or oof.max() > 1
            or test_pred.min() < 0 or test_pred.max() > 1
        )
        if not already_logit:
            oof = expit(np.clip(logit(np.clip(oof, 1e-15, 1 - 1e-15)), -30, 30))
            test_pred = expit(
                np.clip(
                    logit(np.clip(test_pred, 1e-15, 1 - 1e-15)),
                    -30,
                    30,
                )
            )

        if index == 0:
            top_single_oof = oof.copy()
            top_single_test = test_pred.copy()

        oof_rank_sum += norm_rank(oof)
        test_rank_sum += norm_rank(test_pred)
        count = index + 1
        if count in checkpoints:
            checkpoint_oof[count] = oof_rank_sum.copy()
            checkpoint_test[count] = test_rank_sum.copy()
            rank_rows.append(
                {
                    "top_n": count,
                    "rank_avg_oof_auc": roc_auc_score(
                        y, oof_rank_sum
                    ),
                }
            )
            print(rank_rows[-1], flush=True)

    rank_table = pd.DataFrame(rank_rows)
    rank_table.to_csv(OUTPUT_DIR / "rank_average_results.csv", index=False)

    stack_oof = np.load(OUTPUT_DIR / "stack_oof.npy")
    stack_test = np.load(OUTPUT_DIR / "stack_test.npy")

    best_name = "logistic_stack"
    best_auc = roc_auc_score(y, stack_oof)
    best_test = stack_test.copy()
    best_oof = stack_oof.copy()

    for top_n in rank_table["top_n"].astype(int):
        oof_rank = checkpoint_oof[top_n]
        test_rank = checkpoint_test[top_n]

        for weight_stack in np.arange(0.0, 1.0001, 0.05):
            candidate_oof = (
                weight_stack * norm_rank(stack_oof)
                + (1 - weight_stack) * norm_rank(oof_rank)
            )
            auc = roc_auc_score(y, candidate_oof)
            if auc > best_auc:
                best_auc = float(auc)
                best_name = f"stack_vs_rank_top{top_n}_w{weight_stack:.2f}"
                best_oof = candidate_oof
                best_test = (
                    weight_stack * norm_rank(stack_test)
                    + (1 - weight_stack) * norm_rank(test_rank)
                )

    # Blend the logistic stack with the best single model.
    for weight_stack in np.arange(0.0, 1.0001, 0.05):
        candidate = (
            weight_stack * norm_rank(stack_oof)
            + (1 - weight_stack) * norm_rank(top_single_oof)
        )
        auc = roc_auc_score(y, candidate)
        if auc > best_auc:
            best_auc = float(auc)
            best_name = f"stack_vs_top_single_w{weight_stack:.2f}"
            best_oof = candidate
            best_test = (
                weight_stack * norm_rank(stack_test)
                + (1 - weight_stack) * norm_rank(top_single_test)
            )

    submission = pd.DataFrame(
        {
            "id": test["id"].to_numpy(),
            "satisfaction": np.clip(best_test, 0, 1),
        }
    )
    assert submission.columns.tolist() == sample.columns.tolist()
    assert len(submission) == len(sample)
    assert np.array_equal(
        submission["id"].to_numpy(), sample["id"].to_numpy()
    )
    submission.to_csv(OUTPUT_DIR / "submission_best.csv", index=False)
    pd.DataFrame({"id": train["id"], TARGET: best_oof}).to_csv(
        OUTPUT_DIR / "oof_best.csv", index=False
    )

    print(
        "BEST", best_name, "OOF AUC", best_auc,
        "saved", OUTPUT_DIR / "submission_best.csv",
        flush=True,
    )


if __name__ == "__main__":
    main()
