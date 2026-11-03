from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "public_stack"
TARGET = "satisfaction"


def main():
    X = np.load(OUTPUT_DIR / "X_oof.npy", mmap_mode="r")
    y = pd.read_csv(DATA_DIR / "train.csv", usecols=[TARGET])[TARGET].astype(int).to_numpy()
    coef = pd.read_csv(OUTPUT_DIR / "coefficients.csv")
    coef = coef.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = np.zeros(len(y), dtype=int)
    for fold, (_, valid_idx) in enumerate(cv.split(np.zeros(len(y)), y)):
        folds[valid_idx] = fold
    train_mask = folds != 0
    valid_mask = folds == 0

    results = []
    candidates = []
    for k in [10, 20, 30, 40, 50, 60, 80, 100, 120, 135]:
        index = coef.head(k).index.to_numpy()
        candidates.append((f"top_abs_{k}", index))
    positive = coef[coef["coefficient"] > 0].index.to_numpy()
    candidates.append((f"positive_{len(positive)}", positive))

    for name, index in candidates:
        model = LogisticRegression(
            penalty="l2",
            C=0.03,
            solver="lbfgs",
            max_iter=400,
            tol=1e-4,
        )
        model.fit(X[train_mask][:, index], y[train_mask])
        prediction = model.predict_proba(X[valid_mask][:, index])[:, 1]
        row = {
            "name": name,
            "n_features": int(len(index)),
            "auc": roc_auc_score(y[valid_mask], prediction),
        }
        results.append(row)
        print(row, flush=True)

    pd.DataFrame(results).sort_values("auc", ascending=False).to_csv(
        OUTPUT_DIR / "logistic_coef_screen.csv", index=False
    )


if __name__ == "__main__":
    main()
