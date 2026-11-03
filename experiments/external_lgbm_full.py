from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
ORIGINAL_DIR = PROJECT_DIR / "某个公开算法" / "original_data"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "public_stack"
TARGET = "satisfaction"


def main():
    comp = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    original = pd.read_csv(ORIGINAL_DIR / "train.csv")
    original[TARGET] = original[TARGET].map(
        {"neutral or dissatisfied": 0, "satisfied": 1}
    )
    y = comp[TARGET].astype(int).to_numpy()
    y_orig = original[TARGET].astype(int).to_numpy()
    X_comp = comp.drop(columns=["id", TARGET])
    X_test = test.drop(columns=["id"])
    X_orig = original.drop(
        columns=["Unnamed: 0", "id", "Inflight service", TARGET]
    )[X_comp.columns]

    cat_cols = ["Gender", "Customer Type", "Type of Travel", "Class"]
    all_x = pd.concat([X_comp, X_orig, X_test], ignore_index=True)
    all_x = pd.get_dummies(all_x, columns=cat_cols, dtype=np.int8)
    X_comp = all_x.iloc[: len(X_comp)]
    X_orig = all_x.iloc[len(X_comp):len(X_comp) + len(X_orig)]
    X_test = all_x.iloc[len(X_comp) + len(X_orig):].reset_index(drop=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(comp))
    test_pred = np.zeros(len(test))
    started = time.perf_counter()
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X_comp, y), 1):
        X_train = pd.concat(
            [X_comp.iloc[train_idx], X_orig], ignore_index=True
        )
        y_train = np.concatenate([y[train_idx], y_orig])
        weights = np.concatenate(
            [np.ones(len(train_idx)), np.full(len(X_orig), 0.5)]
        )
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
            n_jobs=8,
            random_state=42,
            verbosity=-1,
        )
        model.fit(
            X_train,
            y_train,
            sample_weight=weights,
            eval_set=[(X_comp.iloc[valid_idx], y[valid_idx])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof[valid_idx] = model.predict_proba(X_comp.iloc[valid_idx])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / 5
        print(
            f"fold={fold} auc={roc_auc_score(y[valid_idx], oof[valid_idx]):.6f} "
            f"iter={model.best_iteration_}",
            flush=True,
        )
    auc = roc_auc_score(y, oof)
    print("external LGB OOF", auc, "seconds", time.perf_counter() - started)
    np.save(OUTPUT_DIR / "external_lgb_oof.npy", oof)
    np.save(OUTPUT_DIR / "external_lgb_test.npy", test_pred)
    pd.DataFrame(
        {"id": test["id"].to_numpy(), "satisfaction": np.clip(test_pred, 0, 1)}
    ).to_csv(OUTPUT_DIR / "external_lgb_submission.csv", index=False)


if __name__ == "__main__":
    main()
