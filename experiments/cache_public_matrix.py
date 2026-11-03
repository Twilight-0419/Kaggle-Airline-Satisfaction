from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
PUBLIC_DIR = PROJECT_DIR / "某个公开算法"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "public_stack"
TARGET = "satisfaction"

PRED_DIRS = [
    PUBLIC_DIR / "S6E10 124 Public OOF and Test Predictions",
    PUBLIC_DIR / "S6E10 TFM OOF and Test Predictions",
]


def to_logit(values):
    values = np.clip(values.astype(np.float64), 1e-15, 1 - 1e-15)
    return np.log(values / (1 - values))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv", usecols=["id", TARGET])
    test = pd.read_csv(DATA_DIR / "test.csv", usecols=["id"])

    names = []
    oofs = []
    tests = []
    for root in PRED_DIRS:
        for oof_path in sorted(root.glob("oof_*.npy")):
            name = oof_path.stem[len("oof_"):]
            test_path = oof_path.with_name(f"test_{name}.npy")
            if not test_path.exists():
                continue
            oof = np.load(oof_path)
            test_pred = np.load(test_path)
            if oof.shape != (len(train),) or test_pred.shape != (len(test),):
                continue
            already_logit = bool(
                oof.min() < 0 or oof.max() > 1
                or test_pred.min() < 0 or test_pred.max() > 1
            )
            if not already_logit:
                oof = to_logit(oof)
                test_pred = to_logit(test_pred)
            names.append(f"{root.name}:{name}")
            oofs.append(np.clip(oof, -30, 30).astype(np.float32))
            tests.append(np.clip(test_pred, -30, 30).astype(np.float32))

    X_oof = np.stack(oofs, axis=1)
    X_test = np.stack(tests, axis=1)
    np.save(OUTPUT_DIR / "X_oof.npy", X_oof)
    np.save(OUTPUT_DIR / "X_test.npy", X_test)
    pd.Series(names, name="model").to_csv(
        OUTPUT_DIR / "model_names.csv", index=False
    )
    print("saved", X_oof.shape, X_test.shape)


if __name__ == "__main__":
    main()
