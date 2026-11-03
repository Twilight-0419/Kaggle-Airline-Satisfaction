from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
PUBLIC_DIR = PROJECT_DIR / "某个公开算法"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "public_stack"
TARGET = "satisfaction"

PRED_DIRS = [
    PUBLIC_DIR / "S6E10 124 Public OOF and Test Predictions",
    PUBLIC_DIR / "S6E10 TFM OOF and Test Predictions",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def to_logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float64), 1e-15, 1 - 1e-15)
    return np.log(values / (1 - values))


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_DIR / "train.csv", usecols=["id", TARGET])
    test = pd.read_csv(DATA_DIR / "test.csv", usecols=["id"])
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    y = train[TARGET].astype(int).to_numpy()

    folds = np.zeros(len(y), dtype=int)
    cv = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    for fold, (_, valid_idx) in enumerate(cv.split(np.zeros(len(y)), y)):
        folds[valid_idx] = fold

    names = []
    oof_columns = []
    test_columns = []
    rows = []

    for root in PRED_DIRS:
        for oof_path in sorted(root.glob("oof_*.npy")):
            name = oof_path.stem[len("oof_"):]
            test_path = oof_path.with_name(f"test_{name}.npy")
            if not test_path.exists():
                print("skip missing test", oof_path, flush=True)
                continue

            oof = np.load(oof_path)
            test_pred = np.load(test_path)
            if oof.shape != (len(train),) or test_pred.shape != (len(test),):
                print("skip shape", oof_path, oof.shape, test_pred.shape, flush=True)
                continue
            if not np.isfinite(oof).all() or not np.isfinite(test_pred).all():
                print("skip non-finite", oof_path, flush=True)
                continue

            already_logit = bool(
                oof.min() < 0 or oof.max() > 1
                or test_pred.min() < 0 or test_pred.max() > 1
            )
            if not already_logit:
                oof = to_logit(oof)
                test_pred = to_logit(test_pred)

            unique_name = f"{root.name}:{name}"
            names.append(unique_name)
            oof_columns.append(np.clip(oof, -30, 30).astype(np.float32))
            test_columns.append(np.clip(test_pred, -30, 30).astype(np.float32))
            rows.append(
                {
                    "model": unique_name,
                    "oof_auc": roc_auc_score(y, oof_columns[-1]),
                    "already_logit": already_logit,
                }
            )

    X_oof = np.stack(oof_columns, axis=1)
    X_test = np.stack(test_columns, axis=1)
    model_table = pd.DataFrame(rows).sort_values(
        "oof_auc", ascending=False
    )
    model_table.to_csv(OUTPUT_DIR / "model_table.csv", index=False)

    print(
        f"models={len(names)} X_oof={X_oof.shape} X_test={X_test.shape}",
        flush=True,
    )
    print(model_table.head(15).to_string(index=False), flush=True)

    oof_stack = np.zeros(len(y), dtype=np.float64)
    test_stack = np.zeros(len(test), dtype=np.float64)
    coefs = []
    started = time.perf_counter()

    for fold in range(args.folds):
        train_mask = folds != fold
        valid_mask = folds == fold
        model = LogisticRegression(
            penalty="l2",
            C=args.c,
            fit_intercept=True,
            solver="lbfgs",
            max_iter=args.max_iter,
            tol=1e-5,
        )
        model.fit(X_oof[train_mask], y[train_mask])
        oof_stack[valid_mask] = model.predict_proba(X_oof[valid_mask])[:, 1]
        test_stack += model.predict_proba(X_test)[:, 1] / args.folds
        coefs.append(np.ravel(model.coef_))
        print(
            f"fold={fold + 1} auc="
            f"{roc_auc_score(y[valid_mask], oof_stack[valid_mask]):.6f}",
            flush=True,
        )

    stack_auc = roc_auc_score(y, oof_stack)
    print("Stack OOF AUC:", stack_auc, "seconds:", time.perf_counter() - started)

    np.save(OUTPUT_DIR / "stack_oof.npy", oof_stack)
    np.save(OUTPUT_DIR / "stack_test.npy", test_stack)

    coef = np.mean(coefs, axis=0)
    importance = model_table.assign(
        coefficient=coef,
        abs_coefficient=np.abs(coef),
    ).sort_values("abs_coefficient", ascending=False)
    importance.to_csv(OUTPUT_DIR / "coefficients.csv", index=False)

    submission = pd.DataFrame(
        {
            "id": test["id"].to_numpy(),
            "satisfaction": np.clip(test_stack, 0, 1),
        }
    )
    assert submission.columns.tolist() == sample.columns.tolist()
    assert len(submission) == len(sample)
    assert np.array_equal(
        submission["id"].to_numpy(), sample["id"].to_numpy()
    )
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    print("Saved:", OUTPUT_DIR / "submission.csv", flush=True)


if __name__ == "__main__":
    main()
