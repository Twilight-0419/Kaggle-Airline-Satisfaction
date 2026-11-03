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
ID = "id"


def to_logit(values):
    values = np.asarray(values, dtype=np.float64)
    if values.min() < 0 or values.max() > 1:
        return np.clip(values, -30, 30).astype(np.float32)
    values = np.clip(values, 1e-15, 1 - 1e-15)
    return np.log(values / (1 - values)).astype(np.float32)


def numeric_prediction_column(df):
    candidates = [
        col for col in df.columns
        if col not in {ID, "fold", TARGET, "Unnamed: 0"}
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    if (
        not candidates
        and TARGET in df.columns
        and pd.api.types.is_numeric_dtype(df[TARGET])
    ):
        candidates.append(TARGET)
    if not candidates:
        raise ValueError(f"No prediction column in {df.columns.tolist()}")
    return candidates[-1]


def add_csv_pair(name, oof_path, test_path):
    oof_df = pd.read_csv(oof_path)
    test_df = pd.read_csv(test_path)
    oof_col = numeric_prediction_column(oof_df)
    test_col = numeric_prediction_column(test_df)
    oof = oof_df.set_index(ID)[oof_col].reindex(train_ids).to_numpy()
    test_pred = test_df.set_index(ID)[test_col].reindex(test_ids).to_numpy()
    if np.isnan(oof).any() or np.isnan(test_pred).any():
        raise ValueError(f"Alignment failed for {name}")
    return name, to_logit(oof), to_logit(test_pred)


train_ids = pd.read_csv(DATA_DIR / "train.csv", usecols=[ID])[ID]
test_ids = pd.read_csv(DATA_DIR / "test.csv", usecols=[ID])[ID]
y = pd.read_csv(DATA_DIR / "train.csv", usecols=[TARGET])[TARGET].astype(int).to_numpy()

X_oof = np.load(OUTPUT_DIR / "X_oof.npy")
X_test = np.load(OUTPUT_DIR / "X_test.npy")
names = pd.read_csv(OUTPUT_DIR / "model_names.csv")["model"].tolist()

new_oof = []
new_test = []
new_names = []

# Five direct CSV pairs.
for oof_path in sorted((EXTRA_DIR / "oof-dataset").glob("*_oof.csv")):
    name = oof_path.stem[:-4]
    test_path = EXTRA_DIR / "oof-dataset" / f"{name}_test.csv"
    n, o, t = add_csv_pair(f"abdullah_{name}", oof_path, test_path)
    new_names.append(n)
    new_oof.append(o)
    new_test.append(t)

# Najiama blend pairs.
for oof_path in sorted((EXTRA_DIR / "s6e10-oof").glob("*_blend_oof.csv")):
    stem = oof_path.stem.replace("_blend_oof", "")
    test_path = EXTRA_DIR / "s6e10-oof" / f"{stem}_submission.csv"
    n, o, t = add_csv_pair(f"najiama_{stem}", oof_path, test_path)
    new_names.append(n)
    new_oof.append(o)
    new_test.append(t)

oof_path = EXTRA_DIR / "s6e10-oof" / "07_XGBoost_oof.csv"
test_path = EXTRA_DIR / "s6e10-oof" / "07_submission.csv"
n, o, t = add_csv_pair("najiama_07", oof_path, test_path)
new_names.append(n)
new_oof.append(o)
new_test.append(t)

# Parquet OOF libraries.
for library_name, tag in [
    ("s6e10-oof-library", "megayak"),
    ("s6e10-oof-test-predictions", "arhancanli"),
]:
    root = EXTRA_DIR / library_name
    oof_df = pd.read_parquet(root / "oof_predictions.parquet")
    test_df = pd.read_parquet(root / "test_predictions.parquet")
    model_cols = [
        col for col in test_df.columns
        if col not in {ID, "fold", TARGET}
        and pd.api.types.is_numeric_dtype(test_df[col])
    ]
    for col in model_cols:
        oof = oof_df.set_index(ID)[col].reindex(train_ids).to_numpy()
        test_pred = test_df.set_index(ID)[col].reindex(test_ids).to_numpy()
        if np.isnan(oof).any() or np.isnan(test_pred).any():
            continue
        new_names.append(f"{tag}_{col}")
        new_oof.append(to_logit(oof))
        new_test.append(to_logit(test_pred))

# Busyaprime npz files.
busy_dir = EXTRA_DIR / "s6e10-busyaprime-five-members-oof"
for npz_path in sorted(busy_dir.glob("*.npz")):
    data = np.load(npz_path)
    new_names.append(f"busyaprime_{npz_path.stem}")
    new_oof.append(to_logit(data["oof"]))
    new_test.append(to_logit(data["test"]))

X_oof_ext = np.column_stack([X_oof] + new_oof).astype(np.float32)
X_test_ext = np.column_stack([X_test] + new_test).astype(np.float32)
all_names = names + new_names

np.save(OUTPUT_DIR / "X_oof_ext.npy", X_oof_ext)
np.save(OUTPUT_DIR / "X_test_ext.npy", X_test_ext)
pd.Series(all_names, name="model").to_csv(
    OUTPUT_DIR / "model_names_ext.csv", index=False
)

rows = []
for index, name in enumerate(all_names):
    rows.append(
        {
            "model": name,
            "oof_auc": roc_auc_score(y, X_oof_ext[:, index]),
        }
    )
model_table = pd.DataFrame(rows).sort_values("oof_auc", ascending=False)
model_table.to_csv(OUTPUT_DIR / "model_table_ext.csv", index=False)
print("old", X_oof.shape[1], "new", len(new_names), "total", X_oof_ext.shape[1])
print(model_table.head(25).to_string(index=False))
