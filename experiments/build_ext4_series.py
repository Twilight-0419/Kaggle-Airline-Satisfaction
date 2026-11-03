from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "public_stack"
SERIES_DIR = PROJECT_DIR / "某个公开算法" / "extra_oof2" / "s6e10-goodpjw2008-extra-members-oof"
TARGET = "satisfaction"


def to_logit(values):
    values = np.asarray(values)
    if values.min() < 0 or values.max() > 1:
        return np.clip(values, -30, 30).astype(np.float32)
    values = np.clip(values.astype(np.float64), 1e-15, 1 - 1e-15)
    return np.log(values / (1 - values)).astype(np.float32)


def main():
    y = pd.read_csv(DATA_DIR / "train.csv", usecols=[TARGET])[TARGET].astype(int).to_numpy()
    X = np.load(OUTPUT_DIR / "X_oof_ext.npy")
    X_test = np.load(OUTPUT_DIR / "X_test_ext.npy")
    names = pd.read_csv(OUTPUT_DIR / "model_names_ext.csv")["model"].tolist()
    new_oof, new_test, new_names = [], [], []

    for filename, prefix in [
        ("series_members_p1_p8.npz", "series_member"),
        ("series_stacks_p1_p8.npz", "series_stack"),
    ]:
        data = np.load(SERIES_DIR / filename, allow_pickle=True)
        for key in data.files:
            if not key.startswith("oof_"):
                continue
            name = key[len("oof_"):]
            test_key = f"test_{name}"
            if test_key not in data.files:
                continue
            oof = data[key]
            test_pred = data[test_key]
            if oof.shape != (len(y),) or test_pred.shape != (299844,):
                continue
            new_names.append(f"{prefix}_{name}")
            new_oof.append(to_logit(oof))
            new_test.append(to_logit(test_pred))

    stack_oof = np.load(SERIES_DIR / "stack182_lr_lgb_oof.npz")
    stack_test = np.load(SERIES_DIR / "stack182_lr_lgb_test.npz")
    stack_key_oof = stack_oof.files[0]
    stack_key_test = stack_test.files[0]
    new_names.append("goodpjw_stack182")
    new_oof.append(to_logit(stack_oof[stack_key_oof]))
    new_test.append(to_logit(stack_test[stack_key_test]))

    all_oof = [X[:, i] for i in range(X.shape[1])] + new_oof
    all_test = [X_test[:, i] for i in range(X_test.shape[1])] + new_test
    all_names = names + new_names
    keep = []
    seen = set()
    for i, values in enumerate(all_oof):
        digest = hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        keep.append(i)
    X4 = np.column_stack([all_oof[i] for i in keep]).astype(np.float32)
    X4_test = np.column_stack([all_test[i] for i in keep]).astype(np.float32)
    names4 = [all_names[i] for i in keep]
    np.save(OUTPUT_DIR / "X_oof_ext4u.npy", X4)
    np.save(OUTPUT_DIR / "X_test_ext4u.npy", X4_test)
    pd.Series(names4, name="model").to_csv(
        OUTPUT_DIR / "model_names_ext4u.csv", index=False
    )
    rows = [
        {"model": name, "oof_auc": roc_auc_score(y, X4[:, i])}
        for i, name in enumerate(names4)
    ]
    table = pd.DataFrame(rows).sort_values("oof_auc", ascending=False)
    table.to_csv(OUTPUT_DIR / "model_table_ext4u.csv", index=False)
    print("base", X.shape[1], "added", len(new_names), "unique", X4.shape[1])
    print(table.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
