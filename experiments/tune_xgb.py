from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from feature_screen import feature_configs


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "train.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
TARGET = "satisfaction"
ID_COL = "id"


CONFIGS = {
    "baseline": dict(
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=10,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_lambda=2.0,
    ),
    "lr03_d6": dict(
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=10,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_lambda=2.0,
    ),
    "lr03_d7": dict(
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=10,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_lambda=2.0,
    ),
    "lr03_d8": dict(
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=10,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_lambda=2.0,
    ),
    "mcw05": dict(
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_lambda=2.0,
    ),
    "mcw20": dict(
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=20,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_lambda=2.0,
    ),
    "col06": dict(
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=10,
        subsample=0.85,
        colsample_bytree=0.6,
        reg_lambda=2.0,
    ),
    "col08": dict(
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=10,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=2.0,
    ),
    "lambda5": dict(
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=10,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_lambda=5.0,
    ),
    "deep_regularized": dict(
        learning_rate=0.04,
        max_depth=8,
        min_child_weight=20,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=5.0,
    ),
    "deep9_regularized": dict(
        learning_rate=0.03,
        max_depth=9,
        min_child_weight=20,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=5.0,
    ),
    "slow_deep": dict(
        learning_rate=0.02,
        max_depth=8,
        min_child_weight=10,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=2.0,
    ),
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed = 42
    sample_rows = 280000
    folds = 5

    train = pd.read_csv(DATA_PATH)
    train["Arrival Delay in Minutes"] = train[
        "Arrival Delay in Minutes"
    ].fillna(train["Arrival Delay in Minutes"].mean())
    train, _ = train_test_split(
        train,
        train_size=sample_rows,
        stratify=train[TARGET],
        random_state=seed,
    )
    train = train.reset_index(drop=True)

    y = train[TARGET].astype("int8")
    data = train.drop(columns=[ID_COL, TARGET])
    features, categorical = feature_configs()["raw"]
    X = pd.get_dummies(data[features], columns=categorical, dtype=np.int8)

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    results: list[dict[str, object]] = []
    best_oof = -1.0
    results_path = OUTPUT_DIR / "xgb_tuning_results.csv"
    oof_path = OUTPUT_DIR / "xgb_best_oof.csv"

    print(
        f"rows={len(X):,} features={X.shape[1]} folds={folds}",
        flush=True,
    )

    for name, params in CONFIGS.items():
        started = time.perf_counter()
        oof = np.zeros(len(X), dtype=np.float64)
        fold_scores: list[float] = []
        best_iterations: list[int] = []

        for fold, (train_idx, valid_idx) in enumerate(
            cv.split(X, y), start=1
        ):
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                n_estimators=2500,
                early_stopping_rounds=100,
                tree_method="hist",
                device="cpu",
                n_jobs=8,
                random_state=seed,
                **params,
            )
            model.fit(
                X.iloc[train_idx],
                y.iloc[train_idx],
                eval_set=[(X.iloc[valid_idx], y.iloc[valid_idx])],
                verbose=False,
            )
            prediction = model.predict_proba(X.iloc[valid_idx])[:, 1]
            oof[valid_idx] = prediction
            fold_scores.append(
                roc_auc_score(y.iloc[valid_idx], prediction)
            )
            best_iterations.append(int(model.best_iteration))

            print(
                f"{name:18s} fold={fold}/{folds} "
                f"auc={fold_scores[-1]:.6f} iter={best_iterations[-1]}",
                flush=True,
            )
            del model, prediction
            gc.collect()

        elapsed = time.perf_counter() - started
        oof_auc = roc_auc_score(y, oof)
        row = {
            "name": name,
            **params,
            "oof_auc": oof_auc,
            "fold_mean": float(np.mean(fold_scores)),
            "fold_std": float(np.std(fold_scores)),
            "best_iteration_mean": float(np.mean(best_iterations)),
            "seconds": elapsed,
        }
        results.append(row)
        pd.DataFrame(results).sort_values(
            "oof_auc", ascending=False
        ).to_csv(results_path, index=False)

        if oof_auc > best_oof:
            best_oof = oof_auc
            pd.DataFrame(
                {ID_COL: train[ID_COL], TARGET: oof}
            ).to_csv(oof_path, index=False)

        print(
            f"{name:18s} oof_auc={oof_auc:.6f} "
            f"mean_iter={np.mean(best_iterations):.1f} "
            f"time={elapsed:.1f}s",
            flush=True,
        )

    print(results_path, flush=True)
    print(oof_path, flush=True)


if __name__ == "__main__":
    main()
