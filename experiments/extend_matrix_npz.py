from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "public_stack"
EXTRA_DIR = PROJECT_DIR / "某个公开算法" / "extra_oof"
TARGET = "satisfaction"


def to_logit(values):
    values = np.asarray(values, dtype=np.float64)
    if np.isnan(values).any():
        return None
    if values.min() < 0 or values.max() > 1:
        return np.clip(values, -30, 30).astype(np.float32)
    values = np.clip(values, 1e-15, 1 - 1e-15)
    return np.log(values / (1 - values)).astype(np.float32)


def main():
    y = pd.read_csv(DATA_DIR / "train.csv", usecols=[TARGET])[TARGET].astype(int).to_numpy()
    X_oof = np.load(OUTPUT_DIR / "X_oof_ext.npy")
    X_test = np.load(OUTPUT_DIR / "X_test_ext.npy")
    names = pd.read_csv(OUTPUT_DIR / "model_names_ext.csv")["model"].tolist()
    new_oof, new_test, new_names = [], [], []

    libraries = [
        (EXTRA_DIR / "ps-s6e10-airline-satisfaction-artifacts", "sadam"),
        (EXTRA_DIR / "s6e10-stack-oof-predictions", "sachith"),
    ]
    for root, prefix in libraries:
        for path in sorted(root.glob("*.npz")):
            data = np.load(path)
            oof_key = "oof" if "oof" in data.files else ("train" if "train" in data.files else None)
            test_key = "test" if "test" in data.files else None
            if oof_key is None or test_key is None:
                print("skip keys", path.name, data.files)
                continue
            oof = data[oof_key]
            test_pred = data[test_key]
            if oof.shape != (len(y),) or test_pred.shape != (299844,):
                print("skip shape", path.name, oof.shape, test_pred.shape)
                continue
            o = to_logit(oof)
            t = to_logit(test_pred)
            if o is None or t is None:
                continue
            new_names.append(f"{prefix}_{path.stem}")
            new_oof.append(o)
            new_test.append(t)

    X_oof2 = np.column_stack([X_oof] + new_oof).astype(np.float32)
    X_test2 = np.column_stack([X_test] + new_test).astype(np.float32)
    all_names = names + new_names
    np.save(OUTPUT_DIR / "X_oof_ext2.npy", X_oof2)
    np.save(OUTPUT_DIR / "X_test_ext2.npy", X_test2)
    pd.Series(all_names, name="model").to_csv(
        OUTPUT_DIR / "model_names_ext2.csv", index=False
    )

    rows = [
        {"model": name, "oof_auc": roc_auc_score(y, X_oof2[:, i])}
        for i, name in enumerate(all_names)
    ]
    table = pd.DataFrame(rows).sort_values("oof_auc", ascending=False)
    table.to_csv(OUTPUT_DIR / "model_table_ext2.csv", index=False)
    print("old", X_oof.shape[1], "added", len(new_names), "total", X_oof2.shape[1])
    print(table.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
