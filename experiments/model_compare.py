from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from feature_screen import feature_configs


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "ensemble"
TARGET = "satisfaction"
ID_COL = "id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xgboost", "lightgbm", "catboost"],
    )
    parser.add_argument("--sample-rows", type=int, default=280000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def load_data(sample_rows: int, seed: int):
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    train["Arrival Delay in Minutes"] = train[
        "Arrival Delay in Minutes"
    ].fillna(train["Arrival Delay in Minutes"].mean())
    test["Arrival Delay in Minutes"] = test[
        "Arrival Delay in Minutes"
    ].fillna(test["Arrival Delay in Minutes"].mean())

    if sample_rows and sample_rows < len(train):
        train, _ = train_test_split(
            train,
            train_size=sample_rows,
            stratify=train[TARGET],
            random_state=seed,
        )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    y = train[TARGET].astype("int8")
    train_ids = train[ID_COL].copy()
    test_ids = test[ID_COL].copy()

    features, categorical = feature_configs()["raw"]
    x_train = pd.get_dummies(
        train[features], columns=categorical, dtype=np.int8
    )
    x_test = pd.get_dummies(
        test[features], columns=categorical, dtype=np.int8
    ).reindex(columns=x_train.columns, fill_value=0)
    return x_train, y, x_test, train_ids, test_ids


def fit_predict(
    name: str,
    fold: int,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    x_test: pd.DataFrame,
    seed: int,
    threads: int,
):
    if name == "xgboost":
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=2500,
            learning_rate=0.03,
            max_depth=7,
            min_child_weight=10,
            subsample=0.85,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            early_stopping_rounds=100,
            tree_method="hist",
            device="cpu",
            n_jobs=threads,
            random_state=seed,
        )
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            verbose=False,
        )
        best_iteration = int(model.best_iteration)
    elif name == "lightgbm":
        model = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=2500,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            n_jobs=threads,
            random_state=seed,
            verbosity=-1,
        )
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        best_iteration = int(model.best_iteration_)
    elif name == "catboost":
        model = cb.CatBoostClassifier(
            iterations=2500,
            learning_rate=0.03,
            depth=7,
            l2_leaf_reg=3.0,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            thread_count=threads,
            od_type="Iter",
            od_wait=100,
            use_best_model=True,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            x_train,
            y_train,
            eval_set=(x_valid, y_valid),
            use_best_model=True,
        )
        best_iteration = int(model.get_best_iteration())
    else:
        raise ValueError(f"Unknown model: {name}")

    valid_pred = model.predict_proba(x_valid)[:, 1]
    test_pred = model.predict_proba(x_test)[:, 1]
    print(
        f"{name:9s} fold={fold} auc={roc_auc_score(y_valid, valid_pred):.6f} "
        f"iter={best_iteration}",
        flush=True,
    )
    return valid_pred, test_pred, best_iteration


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, X_test, train_ids, test_ids = load_data(
        args.sample_rows, args.seed
    )
    cv = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    print(
        f"rows={len(X):,} test={len(X_test):,} "
        f"features={X.shape[1]} folds={args.folds}",
        flush=True,
    )

    results: list[dict[str, object]] = []

    for name in args.models:
        started = time.perf_counter()
        oof = np.zeros(len(X), dtype=np.float64)
        test_pred = np.zeros(len(X_test), dtype=np.float64)
        fold_scores: list[float] = []
        best_iterations: list[int] = []

        for fold, (train_idx, valid_idx) in enumerate(
            cv.split(X, y), start=1
        ):
            valid_pred, fold_test_pred, best_iteration = fit_predict(
                name,
                fold,
                X.iloc[train_idx],
                y.iloc[train_idx],
                X.iloc[valid_idx],
                y.iloc[valid_idx],
                X_test,
                args.seed,
                args.threads,
            )
            oof[valid_idx] = valid_pred
            test_pred += fold_test_pred / args.folds
            fold_scores.append(
                roc_auc_score(y.iloc[valid_idx], valid_pred)
            )
            best_iterations.append(best_iteration)
            del valid_pred, fold_test_pred
            gc.collect()

        oof_auc = roc_auc_score(y, oof)
        elapsed = time.perf_counter() - started
        row = {
            "model": name,
            "oof_auc": oof_auc,
            "fold_mean": float(np.mean(fold_scores)),
            "fold_std": float(np.std(fold_scores)),
            "best_iteration_mean": float(np.mean(best_iterations)),
            "seconds": elapsed,
        }
        results.append(row)

        pd.DataFrame(
            {
                ID_COL: train_ids.to_numpy(),
                TARGET: oof,
                "model": name,
            }
        ).to_csv(OUTPUT_DIR / f"{name}_oof.csv", index=False)
        pd.DataFrame(
            {
                ID_COL: test_ids.to_numpy(),
                TARGET: test_pred,
                "model": name,
            }
        ).to_csv(OUTPUT_DIR / f"{name}_test.csv", index=False)
        pd.DataFrame(results).sort_values(
            "oof_auc", ascending=False
        ).to_csv(OUTPUT_DIR / "model_results.csv", index=False)

        print(
            f"{name:9s} oof_auc={oof_auc:.6f} "
            f"mean_iter={np.mean(best_iterations):.1f} "
            f"time={elapsed:.1f}s",
            flush=True,
        )

    print(OUTPUT_DIR, flush=True)


if __name__ == "__main__":
    main()
