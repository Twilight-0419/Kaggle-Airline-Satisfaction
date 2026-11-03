from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "public_stack"
TARGET = "satisfaction"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lightgbm", "xgboost"], required=True)
    parser.add_argument("--rounds", type=int, default=1200)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    X_oof = np.load(OUTPUT_DIR / "X_oof.npy", mmap_mode="r")
    X_test = np.load(OUTPUT_DIR / "X_test.npy", mmap_mode="r")
    train = pd.read_csv(DATA_DIR / "train.csv", usecols=["id", TARGET])
    test = pd.read_csv(DATA_DIR / "test.csv", usecols=["id"])
    y = train[TARGET].astype(int).to_numpy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    best_iterations = []
    started = time.perf_counter()

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X_oof, y), 1):
        if args.model == "lightgbm":
            model = lgb.LGBMClassifier(
                objective="binary",
                metric="auc",
                n_estimators=args.rounds,
                learning_rate=0.03,
                num_leaves=31,
                min_child_samples=100,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.7,
                reg_lambda=5.0,
                n_jobs=args.threads,
                random_state=42,
                verbosity=-1,
            )
            model.fit(
                X_oof[train_idx],
                y[train_idx],
                eval_set=[(X_oof[valid_idx], y[valid_idx])],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            best_iteration = int(model.best_iteration_)
        else:
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                n_estimators=args.rounds,
                learning_rate=0.03,
                max_depth=4,
                min_child_weight=50,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_lambda=5.0,
                early_stopping_rounds=100,
                tree_method="hist",
                device="cpu",
                n_jobs=args.threads,
                random_state=42,
            )
            model.fit(
                X_oof[train_idx],
                y[train_idx],
                eval_set=[(X_oof[valid_idx], y[valid_idx])],
                verbose=False,
            )
            best_iteration = int(model.best_iteration)

        valid_pred = model.predict_proba(X_oof[valid_idx])[:, 1]
        oof[valid_idx] = valid_pred
        test_pred += model.predict_proba(X_test)[:, 1] / 5
        best_iterations.append(best_iteration)
        print(
            f"{args.model} fold={fold}/5 "
            f"auc={roc_auc_score(y[valid_idx], valid_pred):.6f} "
            f"iter={best_iteration}",
            flush=True,
        )

    auc = roc_auc_score(y, oof)
    print(
        args.model, "OOF AUC", auc,
        "mean_iter", float(np.mean(best_iterations)),
        "seconds", time.perf_counter() - started,
        flush=True,
    )
    np.save(OUTPUT_DIR / f"{args.model}_meta_oof.npy", oof)
    np.save(OUTPUT_DIR / f"{args.model}_meta_test.npy", test_pred)
    pd.DataFrame(
        {
            "id": test["id"].to_numpy(),
            "satisfaction": np.clip(test_pred, 0, 1),
        }
    ).to_csv(OUTPUT_DIR / f"{args.model}_meta_submission.csv", index=False)


if __name__ == "__main__":
    main()
