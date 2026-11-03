"""Leakage-safe CV experiment for Flight Distance condition target encodings.

Variants: A=best existing raw feature configuration; B=exact-distance pairwise
encodings; C=10-unit distance-bin pairwise encodings. Training-row encodings are
inner-OOF within each outer training fold. Outer validation and test encodings
use only statistics from that outer training fold.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "distance_condition_encoding"
TARGET = "satisfaction"
ID = "id"
DIST = "Flight Distance"
CLASS = "Class"
TRAVEL = "Type of Travel"
SMOOTHS = (20, 100, 300)
OUTER_FOLDS = 5
OUTER_SEED = 42
INNER_FOLDS = 5
INNER_SEED_BASE = 2026


def target_encode_group(
    train_frame: pd.DataFrame,
    y: np.ndarray,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    key_columns: list[str],
    smooth_values: tuple[int, ...],
    inner_seed: int,
):
    """Return inner-OOF train encodings and outer-fit valid/test encodings."""
    # Factorization sees feature keys only; no validation/test labels are read.
    combined = pd.concat(
        [train_frame[key_columns], valid_frame[key_columns], test_frame[key_columns]],
        ignore_index=True,
    )
    codes, uniques = pd.factorize(pd.MultiIndex.from_frame(combined), sort=False)
    n_train, n_valid = len(train_frame), len(valid_frame)
    train_codes = codes[:n_train]
    valid_codes = codes[n_train:n_train + n_valid]
    test_codes = codes[n_train + n_valid:]
    n_groups = len(uniques)
    alphas = np.asarray(smooth_values, dtype=np.float64)

    train_encoded = np.empty((n_train, len(alphas)), dtype=np.float32)
    inner_cv = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=inner_seed)
    for inner_fit, inner_hold in inner_cv.split(np.zeros(n_train), y):
        prior = float(y[inner_fit].mean())
        sums = np.bincount(train_codes[inner_fit], weights=y[inner_fit], minlength=n_groups)
        counts = np.bincount(train_codes[inner_fit], minlength=n_groups)
        hold_codes = train_codes[inner_hold]
        train_encoded[inner_hold] = (
            sums[hold_codes, None] + prior * alphas[None, :]
        ) / (counts[hold_codes, None] + alphas[None, :])

    # All outer-train labels may be used for outer validation and test encodings.
    outer_prior = float(y.mean())
    sums = np.bincount(train_codes, weights=y, minlength=n_groups)
    counts = np.bincount(train_codes, minlength=n_groups)
    valid_encoded = (
        sums[valid_codes, None] + outer_prior * alphas[None, :]
    ) / (counts[valid_codes, None] + alphas[None, :])
    test_encoded = (
        sums[test_codes, None] + outer_prior * alphas[None, :]
    ) / (counts[test_codes, None] + alphas[None, :])
    return train_encoded, valid_encoded.astype(np.float32), test_encoded.astype(np.float32)


def encode_variant(train_frame, y, valid_frame, test_frame, variant, inner_seed):
    if variant == "B_exact_distance":
        train_source, valid_source, test_source = train_frame, valid_frame, test_frame
        distance_name = DIST
    elif variant == "C_distance_bin10":
        train_source = train_frame.copy()
        valid_source = valid_frame.copy()
        test_source = test_frame.copy()
        for frame in (train_source, valid_source, test_source):
            # Existing public feature recipe documents Flight Distance // 10.
            frame["Flight Distance Bin10"] = np.floor_divide(frame[DIST].to_numpy(dtype=np.int64), 10)
        distance_name = "Flight Distance Bin10"
    else:
        raise ValueError(variant)

    train_parts, valid_parts, test_parts = [], [], []
    key_definitions = [
        ([distance_name, CLASS], "distance_x_class"),
        ([distance_name, TRAVEL], "distance_x_travel"),
    ]
    for key_columns, name in key_definitions:
        a, b, c = target_encode_group(
            train_source, y, valid_source, test_source, key_columns, SMOOTHS, inner_seed
        )
        train_parts.append(a)
        valid_parts.append(b)
        test_parts.append(c)
    # Shape is rows x (two condition keys x three smoothing values).
    return (
        np.concatenate(train_parts, axis=1),
        np.concatenate(valid_parts, axis=1),
        np.concatenate(test_parts, axis=1),
        [f"{variant}_{key}_smooth_{s}" for key in ["distance_x_class", "distance_x_travel"] for s in SMOOTHS],
    )


def fit_xgb(x_train, y_train, seed):
    # Reuses the existing 5-fold XGBoost feature-screen parameters unchanged.
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=10,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_lambda=2.0,
        tree_method="hist",
        device="cpu",
        n_jobs=8,
        random_state=seed,
    ).fit(x_train, y_train, verbose=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    # Test labels are deliberately excluded from the read.
    test = pd.read_csv(DATA / "test.csv", usecols=lambda col: col != TARGET)
    required = {ID, TARGET, DIST, CLASS, TRAVEL}
    if missing := required - set(train.columns):
        raise ValueError(f"Training data missing required columns: {sorted(missing)}")
    if missing := ({ID, DIST, CLASS, TRAVEL} - set(test.columns)):
        raise ValueError(f"Test data missing required columns: {sorted(missing)}")
    if not pd.api.types.is_integer_dtype(train[DIST]):
        raise TypeError(f"{DIST} must be integer for the documented //10 bin")
    if train[[DIST, CLASS, TRAVEL, TARGET]].isna().any().any() or test[[DIST, CLASS, TRAVEL]].isna().any().any():
        raise ValueError("Missing values found in target-encoding keys or labels")
    if not train[ID].is_unique or not test[ID].is_unique:
        raise ValueError("Train/test IDs are not unique")

    # Select the existing feature configuration with the best recorded 5-fold OOF
    # mean in outputs/oof_feature_results.csv. It is the raw feature set.
    feature_history = pd.read_csv(ROOT / "outputs" / "oof_feature_results.csv")
    baseline_name = str(feature_history.sort_values("oof_auc", ascending=False).iloc[0]["name"])
    if baseline_name != "raw":
        raise ValueError(f"Expected documented best OOF baseline 'raw', found {baseline_name!r}")
    raw_columns = [
        "Age", DIST, "Inflight wifi service", "Departure/Arrival time convenient",
        "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
        "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service",
        "Baggage handling", "Checkin service", "Cleanliness", "Departure Delay in Minutes",
        "Arrival Delay in Minutes", "Gender", "Customer Type", TRAVEL, CLASS,
    ]
    categorical = ["Gender", "Customer Type", TRAVEL, CLASS]
    if missing := set(raw_columns) - set(train.columns):
        raise ValueError(f"Raw baseline feature columns missing: {sorted(missing)}")
    if missing := set(raw_columns) - set(test.columns):
        raise ValueError(f"Test raw feature columns missing: {sorted(missing)}")
    y_all = train[TARGET].astype(np.int8).to_numpy()
    base_train = train[raw_columns].copy()
    base_test = test[raw_columns].copy()

    # Existing source README describes the prior coarse distance key as floor(distance / 10).
    bin_summary = {
        "raw_feature_config": baseline_name,
        "distance_column": DIST,
        "distance_dtype": str(train[DIST].dtype),
        "distance_range": [int(train[DIST].min()), int(train[DIST].max())],
        "existing_documented_binning": "Flight Distance // 10; remainder/digit keys also appear in public OOF recipes",
        "C_definition": "floor(Flight Distance / 10)",
        "combination_keys": [[DIST, CLASS], [DIST, TRAVEL]],
    }
    config = {
        "baseline_feature_set": baseline_name,
        "baseline_source": "outputs/oof_feature_results.csv best recorded 5-fold OOF row; experiments/feature_screen.py raw config",
        "n_train": len(train), "n_test": len(test),
        "outer_cv": {"type": "StratifiedKFold", "n_splits": OUTER_FOLDS, "shuffle": True, "random_state": OUTER_SEED},
        "inner_target_encoding_cv": {"type": "StratifiedKFold", "n_splits": INNER_FOLDS, "shuffle": True, "random_state_per_outer_fold": "2026 + outer_fold_number"},
        "smoothing_values": list(SMOOTHS),
        "new_features_per_B_or_C_variant": 2,
        "runtime_metric": "fold_results.fit_predict_seconds and summary.total_fit_predict_seconds cover XGBoost fit+predict time; target-encoding feature construction is excluded.",
        "target_encoding_formula": "(group_target_sum + smoothing * fold_prior) / (group_count + smoothing)",
        "model": {"class": "xgboost.XGBClassifier", "objective": "binary:logistic", "eval_metric": "auc", "n_estimators": 300, "learning_rate": 0.05, "max_depth": 7, "min_child_weight": 10, "subsample": 0.85, "colsample_bytree": 0.7, "reg_lambda": 2.0, "tree_method": "hist", "device": "cpu", "n_jobs": 8, "random_state": 42},
        "distance_inspection": bin_summary,
        "leakage_controls": ["Outer folds fixed and shared across all variants.", "Training-row target encodings are generated inner-OOF within the outer training fold.", "Outer validation/test encodings use only the outer training fold labels/statistics.", "Test satisfaction is never loaded or used."],
    }
    (OUT / "experiment_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print("columns verified", sorted(required), "distance range", bin_summary["distance_range"], "baseline", baseline_name, flush=True)

    outer_cv = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=OUTER_SEED)
    folds = list(outer_cv.split(np.zeros(len(y_all)), y_all))
    variants = ["A_existing_raw"] + [f"B_exact_distance_smooth_{s}" for s in SMOOTHS] + [f"C_distance_bin10_smooth_{s}" for s in SMOOTHS]
    oof = {name: np.full(len(train), np.nan, dtype=np.float32) for name in variants}
    test_sum = {name: np.zeros(len(test), dtype=np.float64) for name in variants}
    fold_rows = []
    results_path = OUT / "fold_results.csv"

    for fold_no, (outer_train_idx, outer_valid_idx) in enumerate(folds, start=1):
        y_tr = y_all[outer_train_idx]
        tr_df = base_train.iloc[outer_train_idx].reset_index(drop=True)
        va_df = base_train.iloc[outer_valid_idx].reset_index(drop=True)
        te_df = base_test.reset_index(drop=True)
        xtr_df = pd.get_dummies(tr_df[raw_columns], columns=categorical, dtype=np.int8)
        xva_df = pd.get_dummies(va_df[raw_columns], columns=categorical, dtype=np.int8).reindex(columns=xtr_df.columns, fill_value=0)
        xte_df = pd.get_dummies(te_df[raw_columns], columns=categorical, dtype=np.int8).reindex(columns=xtr_df.columns, fill_value=0)
        xtr_base = xtr_df.to_numpy(dtype=np.float32, copy=False)
        xva_base = xva_df.to_numpy(dtype=np.float32, copy=False)
        xte_base = xte_df.to_numpy(dtype=np.float32, copy=False)

        jobs = [("A_existing_raw", None, None)]
        for variant in ["B_exact_distance", "C_distance_bin10"]:
            enc_train, enc_valid, enc_test, enc_names = encode_variant(
                tr_df, y_tr, va_df, te_df, variant, INNER_SEED_BASE + fold_no
            )
            for smooth_idx, smooth in enumerate(SMOOTHS):
                name = f"{variant}_smooth_{smooth}"
                col_idx = [smooth_idx, len(SMOOTHS) + smooth_idx]
                jobs.append((name, (enc_train[:, col_idx], enc_valid[:, col_idx], enc_test[:, col_idx]), enc_names))

        for name, enc, _ in jobs:
            started = time.perf_counter()
            if enc is None:
                xtr, xva, xte = xtr_base, xva_base, xte_base
            else:
                etr, eva, ete = enc
                xtr = np.column_stack([xtr_base, etr])
                xva = np.column_stack([xva_base, eva])
                xte = np.column_stack([xte_base, ete])
            model = fit_xgb(xtr, y_tr, OUTER_SEED)
            va_pred = model.predict_proba(xva)[:, 1].astype(np.float32)
            te_pred = model.predict_proba(xte)[:, 1].astype(np.float32)
            auc = roc_auc_score(y_all[outer_valid_idx], va_pred)
            oof[name][outer_valid_idx] = va_pred
            test_sum[name] += te_pred / OUTER_FOLDS
            row = {
                "variant": name,
                "fold": fold_no,
                "auc": float(auc),
                "train_rows": len(outer_train_idx),
                "valid_rows": len(outer_valid_idx),
                "n_features": xtr.shape[1],
                "fit_predict_seconds": time.perf_counter() - started,
                "smooth": 0 if enc is None else int(name.rsplit("_", 1)[1]),
            }
            fold_rows.append(row)
            pd.DataFrame(fold_rows).to_csv(results_path, index=False)
            print(f"fold={fold_no}/{OUTER_FOLDS} variant={name} auc={auc:.6f} seconds={row['fit_predict_seconds']:.1f}", flush=True)
            del model, xtr, xva, xte, va_pred, te_pred
            gc.collect()

    summary_rows = []
    baseline_by_fold = {
        row["fold"]: row["auc"]
        for row in fold_rows
        if row["variant"] == "A_existing_raw"
    }
    for row in fold_rows:
        row["delta_auc_vs_A_same_fold"] = row["auc"] - baseline_by_fold[row["fold"]]
    pd.DataFrame(fold_rows).to_csv(results_path, index=False)
    for name in variants:
        scores = [r["auc"] for r in fold_rows if r["variant"] == name]
        summary_rows.append({
            "variant": name,
            "fold_mean_auc": float(np.mean(scores)),
            "fold_std_auc_population": float(np.std(scores, ddof=0)),
            "fold_std_auc_sample": float(np.std(scores, ddof=1)),
            "total_fit_predict_seconds": float(sum(r["fit_predict_seconds"] for r in fold_rows if r["variant"] == name)),
            "fold_aucs": ";".join(f"{v:.9f}" for v in scores),
            "delta_mean_vs_A": float(np.mean(scores) - np.mean([r["auc"] for r in fold_rows if r["variant"] == "A_existing_raw"])),
        })
        if not np.isfinite(oof[name]).all():
            raise ValueError(f"OOF predictions incomplete for {name}")
        np.save(OUT / f"{name}_oof.npy", oof[name])
        np.save(OUT / f"{name}_test_mean5fold.npy", test_sum[name].astype(np.float32))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    test_output = {ID: test[ID].to_numpy()}
    test_output.update({name: test_sum[name].astype(np.float32) for name in variants})
    pd.DataFrame(test_output).to_csv(OUT / "test_predictions.csv", index=False)
    print("\nSUMMARY\n", summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
