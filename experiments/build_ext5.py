from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT = ROOT / "outputs" / "public_stack"
NEW = ROOT / "某个公开算法" / "extra_oof3"
TARGET = "satisfaction"


def logit(v):
    v = np.asarray(v)
    if v.min() < 0 or v.max() > 1:
        return np.clip(v, -30, 30).astype(np.float32)
    v = np.clip(v.astype(np.float64), 1e-15, 1 - 1e-15)
    return np.log(v / (1 - v)).astype(np.float32)


def main():
    y = pd.read_csv(DATA_DIR / "train.csv", usecols=[TARGET])[TARGET].astype(int).to_numpy()
    train_ids = pd.read_csv(DATA_DIR / "train.csv", usecols=["id"])["id"]
    test_ids = pd.read_csv(DATA_DIR / "test.csv", usecols=["id"])["id"]
    X = np.load(OUT / "X_oof_ext.npy")
    XT = np.load(OUT / "X_test_ext.npy")
    names = pd.read_csv(OUT / "model_names_ext.csv")["model"].tolist()
    cols_o, cols_t, new_names = [], [], []

    tab = NEW / "s6e10-tabpfn-member"
    for oof_path in sorted(tab.glob("oof_*.csv")):
        name = oof_path.stem[4:]
        test_path = tab / f"test_{name}.csv"
        od = pd.read_csv(oof_path)
        td = pd.read_csv(test_path)
        ocol = next(c for c in od.columns if c not in ["id", "fold"])
        tcol = next(c for c in td.columns if c not in ["id"])
        o = od.set_index("id")[ocol].reindex(train_ids).to_numpy()
        t = td.set_index("id")[tcol].reindex(test_ids).to_numpy()
        new_names.append(f"x3_tabpfn_{name}")
        cols_o.append(logit(o))
        cols_t.append(logit(t))

    atlas = NEW / "s6e10-lb-surface-atlas"
    new_names.append("x3_own_cv_stack181")
    cols_o.append(logit(np.load(atlas / "own_cv_stack_181_oof.npy")))
    cols_t.append(logit(np.load(atlas / "own_cv_stack_181_test.npy")))

    all_o = [X[:, i] for i in range(X.shape[1])] + cols_o
    all_t = [XT[:, i] for i in range(XT.shape[1])] + cols_t
    all_names = names + new_names
    keep, seen = [], set()
    for i, v in enumerate(all_o):
        h = hashlib.sha256(np.ascontiguousarray(v).view(np.uint8)).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        keep.append(i)
    X5 = np.column_stack([all_o[i] for i in keep]).astype(np.float32)
    XT5 = np.column_stack([all_t[i] for i in keep]).astype(np.float32)
    names5 = [all_names[i] for i in keep]
    np.save(OUT / "X_oof_ext5u.npy", X5)
    np.save(OUT / "X_test_ext5u.npy", XT5)
    pd.Series(names5, name="model").to_csv(OUT / "model_names_ext5u.csv", index=False)
    table = pd.DataFrame(
        [
            {"model": n, "oof_auc": roc_auc_score(y, X5[:, i])}
            for i, n in enumerate(names5)
        ]
    ).sort_values("oof_auc", ascending=False)
    table.to_csv(OUT / "model_table_ext5u.csv", index=False)
    print("added", len(new_names), "unique", X5.shape[1])
    print(table.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
