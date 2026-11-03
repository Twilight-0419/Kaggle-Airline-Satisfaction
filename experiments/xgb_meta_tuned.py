from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "public_stack"
TARGET = "satisfaction"


def main():
    X_oof = np.load(OUTPUT_DIR / "X_oof.npy", mmap_mode="r")
    X_test = np.load(OUTPUT_DIR / "X_test.npy", mmap_mode="r")
    train = pd.read_csv(DATA_DIR / "train.csv", usecols=["id", TARGET])
    test = pd.read_csv(DATA_DIR / "test.csv", usecols=["id"])
    y = train[TARGET].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    started = time.perf_counter()
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X_oof, y), 1):
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=30,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            early_stopping_rounds=100,
            tree_method="hist",
            device="cpu",
            n_jobs=8,
            random_state=42,
        )
        model.fit(
            X_oof[train_idx],
            y[train_idx],
            eval_set=[(X_oof[valid_idx], y[valid_idx])],
            verbose=False,
        )
        oof[valid_idx] = model.predict_proba(X_oof[valid_idx])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / 5
        print(
            f"fold={fold} auc={roc_auc_score(y[valid_idx], oof[valid_idx]):.6f} "
            f"iter={model.best_iteration}",
            flush=True,
        )
    auc = roc_auc_score(y, oof)
    print("tuned xgb meta OOF", auc, "seconds", time.perf_counter() - started)
    np.save(OUTPUT_DIR / "xgboost_meta_tuned_oof.npy", oof)
    np.save(OUTPUT_DIR / "xgboost_meta_tuned_test.npy", test_pred)


if __name__ == "__main__":
    main()
