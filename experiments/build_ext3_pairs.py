from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "public_stack"
EXTRA_DIR = PROJECT_DIR / "某个公开算法" / "extra_oof2"
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

    for root in sorted([p for p in EXTRA_DIR.iterdir() if p.is_dir()]):
        for oof_path in sorted(root.glob("oof_*.npy")):
            name = oof_path.stem[len("oof_"):]
            test_path = root / f"test_{name}.npy"
            if not test_path.exists():
                continue
            oof = np.load(oof_path)
            test_pred = np.load(test_path)
            if oof.shape != (len(y),) or test_pred.shape != (299844,):
                print("skip", root.name, name, oof.shape, test_pred.shape)
                continue
            new_names.append(f"x2_{root.name}_{name}")
            new_oof.append(to_logit(oof))
            new_test.append(to_logit(test_pred))

    all_oof = [X[:, i] for i in range(X.shape[1])] + new_oof
    all_test = [X_test[:, i] for i in range(X_test.shape[1])] + new_test
    all_names = names + new_names

    keep = []
    seen = set()
    for i, matrix in enumerate(all_oof):
        digest = hashlib.sha256(np.ascontiguousarray(matrix).view(np.uint8)).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        keep.append(i)

    X_final = np.column_stack([all_oof[i] for i in keep]).astype(np.float32)
    X_test_final = np.column_stack([all_test[i] for i in keep]).astype(np.float32)
    names_final = [all_names[i] for i in keep]
    np.save(OUTPUT_DIR / "X_oof_ext3u.npy", X_final)
    np.save(OUTPUT_DIR / "X_test_ext3u.npy", X_test_final)
    pd.Series(names_final, name="model").to_csv(
        OUTPUT_DIR / "model_names_ext3u.csv", index=False
    )
    rows = [
        {"model": name, "oof_auc": roc_auc_score(y, X_final[:, i])}
        for i, name in enumerate(names_final)
    ]
    table = pd.DataFrame(rows).sort_values("oof_auc", ascending=False)
    table.to_csv(OUTPUT_DIR / "model_table_ext3u.csv", index=False)
    print("base", X.shape[1], "added", len(new_names), "unique", X_final.shape[1])
    print(table.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
