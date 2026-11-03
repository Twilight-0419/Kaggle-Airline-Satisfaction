from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from feature_screen import add_engineered_features, feature_configs


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "train.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
TARGET = "satisfaction"
ID_COL = "id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["raw", "avg_wifi", "avg_combo", "current"],
    )
    parser.add_argument("--sample-rows", type=int, default=280000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_PATH)
    train["Arrival Delay in Minutes"] = train[
        "Arrival Delay in Minutes"
    ].fillna(train["Arrival Delay in Minutes"].mean())

    if args.sample_rows and args.sample_rows < len(train):
        train, _ = train_test_split(
            train,
            train_size=args.sample_rows,
            stratify=train[TARGET],
            random_state=args.seed,
        )
    train = train.reset_index(drop=True)

    y = train[TARGET].astype("int8")
    data = add_engineered_features(train.drop(columns=[ID_COL, TARGET]))
    all_configs = feature_configs()
    results: list[dict[str, object]] = []
    results_path = OUTPUT_DIR / "oof_feature_results.csv"

    cv = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )

    print(
        f"rows={len(data):,} folds={args.folds} rounds={args.rounds}",
        flush=True,
    )

    for name in args.configs:
        features, categorical = all_configs[name]
        started = time.perf_counter()
        oof = np.zeros(len(data), dtype=np.float64)
        fold_scores: list[float] = []
        best_iterations: list[int] = []
        feature_count = 0

        for fold, (train_idx, valid_idx) in enumerate(
            cv.split(data, y), start=1
        ):
            x_train = data.iloc[train_idx][features]
            x_valid = data.iloc[valid_idx][features]
            y_train = y.iloc[train_idx]
            y_valid = y.iloc[valid_idx]

            x_train = pd.get_dummies(
                x_train, columns=categorical, dtype=np.int8
            )
            x_valid = pd.get_dummies(
                x_valid, columns=categorical, dtype=np.int8
            ).reindex(columns=x_train.columns, fill_value=0)
            feature_count = x_train.shape[1]

            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                n_estimators=args.rounds,
                learning_rate=0.05,
                max_depth=7,
                min_child_weight=10,
                subsample=0.85,
                colsample_bytree=0.7,
                reg_lambda=2.0,
                tree_method="hist",
                device="cpu",
                n_jobs=args.n_jobs,
                random_state=args.seed,
            )
            model.fit(x_train, y_train, verbose=False)

            prediction = model.predict_proba(x_valid)[:, 1]
            oof[valid_idx] = prediction
            fold_scores.append(roc_auc_score(y_valid, prediction))
            best_iterations.append(args.rounds - 1)

            print(
                f"{name:12s} fold={fold}/{args.folds} "
                f"auc={fold_scores[-1]:.6f}",
                flush=True,
            )
            del model, x_train, x_valid, prediction
            gc.collect()

        elapsed = time.perf_counter() - started
        row = {
            "name": name,
            "oof_auc": roc_auc_score(y, oof),
            "fold_mean": float(np.mean(fold_scores)),
            "fold_std": float(np.std(fold_scores)),
            "n_features": feature_count,
            "seconds": elapsed,
        }
        results.append(row)
        pd.DataFrame(results).sort_values(
            "oof_auc", ascending=False
        ).to_csv(results_path, index=False)
        print(
            f"{name:12s} oof_auc={row['oof_auc']:.6f} "
            f"mean={row['fold_mean']:.6f} std={row['fold_std']:.6f} "
            f"time={elapsed:.1f}s",
            flush=True,
        )

    print(results_path, flush=True)


if __name__ == "__main__":
    main()
