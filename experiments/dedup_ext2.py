from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "public_stack"


def main():
    X = np.load(OUTPUT_DIR / "X_oof_ext2.npy")
    X_test = np.load(OUTPUT_DIR / "X_test_ext2.npy")
    names = pd.read_csv(OUTPUT_DIR / "model_names_ext2.csv")["model"].tolist()
    keep = []
    seen = set()
    for i in range(X.shape[1]):
        digest = hashlib.sha256(np.ascontiguousarray(X[:, i]).view(np.uint8)).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        keep.append(i)
    X_u = X[:, keep]
    X_test_u = X_test[:, keep]
    names_u = [names[i] for i in keep]
    np.save(OUTPUT_DIR / "X_oof_ext2u.npy", X_u)
    np.save(OUTPUT_DIR / "X_test_ext2u.npy", X_test_u)
    pd.Series(names_u, name="model").to_csv(
        OUTPUT_DIR / "model_names_ext2u.csv", index=False
    )
    print("before", X.shape[1], "after", X_u.shape[1])


if __name__ == "__main__":
    main()
