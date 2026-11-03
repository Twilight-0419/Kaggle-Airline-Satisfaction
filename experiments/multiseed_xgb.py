from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from model_compare import load_data


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs" / "ensemble"
TARGET = "satisfaction"
ID_COL = "id"
SEEDS = [42, 2026, 777]


def main() -> None:
    X, y, X_test, train_ids, test_ids = load_data(280000, 42)
    seed_oofs = []
    seed_tests = []
    metric_rows = []

    seed42_oof = pd.read_csv(OUTPUT_DIR / "xgboost_oof.csv")
    seed42_test = pd.read_csv(OUTPUT_DIR / "xgboost_test.csv")
    seed_oofs.append(seed42_oof[TARGET].to_numpy())
    seed_tests.append(seed42_test[TARGET].to_numpy())
    metric_rows.append(
        {
            "seed": 42,
            "oof_auc": roc_auc_score(y, seed42_oof[TARGET]),
            "seconds": 0.0,
        }
    )

    for seed in [2026, 777]:
        started = time.perf_counter()
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        )
        oof = np.zeros(len(X), dtype=np.float64)
        test_pred = np.zeros(len(X_test), dtype=np.float64)

        for fold, (train_idx, valid_idx) in enumerate(
            cv.split(X, y), start=1
        ):
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
                n_jobs=8,
                random_state=seed,
            )
            model.fit(
                X.iloc[train_idx],
                y.iloc[train_idx],
                eval_set=[(X.iloc[valid_idx], y.iloc[valid_idx])],
                verbose=False,
            )
            valid_pred = model.predict_proba(X.iloc[valid_idx])[:, 1]
            oof[valid_idx] = valid_pred
            test_pred += model.predict_proba(X_test)[:, 1] / 5
            print(
                f"seed={seed} fold={fold}/5 "
                f"auc={roc_auc_score(y.iloc[valid_idx], valid_pred):.6f} "
                f"iter={model.best_iteration}",
                flush=True,
            )
            del model, valid_pred
            gc.collect()

        seed_oofs.append(oof)
        seed_tests.append(test_pred)
        auc = roc_auc_score(y, oof)
        metric_rows.append(
            {
                "seed": seed,
                "oof_auc": auc,
                "seconds": time.perf_counter() - started,
            }
        )
        pd.DataFrame(
            {ID_COL: train_ids, TARGET: oof}
        ).to_csv(OUTPUT_DIR / f"xgboost_seed{seed}_oof.csv", index=False)
        pd.DataFrame(
            {ID_COL: test_ids, TARGET: test_pred}
        ).to_csv(OUTPUT_DIR / f"xgboost_seed{seed}_test.csv", index=False)
        print(f"seed={seed} oof_auc={auc:.6f}", flush=True)

    oof_mean = np.mean(seed_oofs, axis=0)
    test_mean = np.mean(seed_tests, axis=0)
    metric_rows.append(
        {
            "seed": "mean",
            "oof_auc": roc_auc_score(y, oof_mean),
            "seconds": 0.0,
        }
    )
    pd.DataFrame(metric_rows).to_csv(
        OUTPUT_DIR / "xgb_multiseed_results.csv", index=False
    )
    pd.DataFrame(
        {ID_COL: train_ids, TARGET: oof_mean}
    ).to_csv(OUTPUT_DIR / "xgboost_multiseed_oof.csv", index=False)
    pd.DataFrame(
        {ID_COL: test_ids, TARGET: test_mean}
    ).to_csv(OUTPUT_DIR / "xgboost_multiseed_test.csv", index=False)
    print(pd.DataFrame(metric_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
