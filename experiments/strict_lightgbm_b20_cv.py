"""Strict outer-CV LightGBM baseline vs exact Flight Distance B20 encoding.

Writes only to outputs/strict_lightgbm_b20_cv. It does not read test labels,
save test predictions, or modify existing models, OOF files, or submissions.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "strict_lightgbm_b20_cv"
TARGET, ID = "satisfaction", "id"
DISTANCE, CLASS, TRAVEL = "Flight Distance", "Class", "Type of Travel"
OUTER_FOLDS, OUTER_SEED = 5, 42
INNER_FOLDS, INNER_SEED_BASE = 5, 2026
SMOOTH = 20
SAMPLE_ROWS, SAMPLE_SEED = 280_000, 42
PARAMS = {
    "objective": "binary", "metric": "auc", "n_estimators": 600,
    "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 20,
    "subsample": 0.85, "subsample_freq": 1, "colsample_bytree": 0.7,
    "reg_lambda": 2.0, "n_jobs": 8, "random_state": 42, "verbosity": -1,
}


def target_encoding(train: pd.DataFrame, y: np.ndarray, valid: pd.DataFrame, seed: int):
    """Two exact-distance pair encodings; train values are inner-OOF only."""
    train_result = np.empty((len(train), 2), dtype=np.float32)
    valid_result = np.empty((len(valid), 2), dtype=np.float32)
    inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed)

    for col_no, key_cols in enumerate(([DISTANCE, CLASS], [DISTANCE, TRAVEL])):
        joined_keys = pd.concat(
            [train.loc[:, key_cols], valid.loc[:, key_cols]], ignore_index=True
        )
        codes, _ = pd.factorize(pd.MultiIndex.from_frame(joined_keys), sort=False)
        train_codes, valid_codes = codes[:len(train)], codes[len(train):]
        n_groups = int(codes.max()) + 1

        for fit_idx, hold_idx in inner.split(np.zeros(len(y)), y):
            prior = float(y[fit_idx].mean())
            sums = np.bincount(train_codes[fit_idx], weights=y[fit_idx], minlength=n_groups)
            counts = np.bincount(train_codes[fit_idx], minlength=n_groups)
            hold_codes = train_codes[hold_idx]
            train_result[hold_idx, col_no] = (
                sums[hold_codes] + SMOOTH * prior
            ) / (counts[hold_codes] + SMOOTH)

        prior = float(y.mean())
        sums = np.bincount(train_codes, weights=y, minlength=n_groups)
        counts = np.bincount(train_codes, minlength=n_groups)
        valid_result[:, col_no] = (
            sums[valid_codes] + SMOOTH * prior
        ) / (counts[valid_codes] + SMOOTH)
    return train_result, valid_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print the fixed plan without fitting")
    args = parser.parse_args()
    plan = {
        "cohort": {"rows": SAMPLE_ROWS, "method": "stratified train_test_split", "random_state": SAMPLE_SEED},
        "outer_cv": {"splitter": "StratifiedKFold", "n_splits": OUTER_FOLDS, "shuffle": True, "random_state": OUTER_SEED},
        "inner_target_encoding_cv": {"splitter": "StratifiedKFold", "n_splits": INNER_FOLDS, "shuffle": True, "seed_per_outer_fold": "2026 + fold number"},
        "A": "raw features only",
        "B": "same raw features plus exact Flight Distance × Class and Flight Distance × Type of Travel target encodings",
        "smoothing": SMOOTH,
        "model": "LightGBM, fixed 600 rounds; no early stopping; parameters copied from notebook/main_lightgbm_backup.ipynb",
        "outer_validation_labels": "used only for final fold AUC; excluded from feature encoding, early stopping, feature selection and parameter selection",
        "fits": OUTER_FOLDS * 2,
        "output_directory": str(OUT),
        "test_data": "not read; no test predictions generated",
        "overwrite_policy": "write new files only under outputs/strict_lightgbm_b20_cv",
        "runtime_reference_seconds": {"historic_raw_lgb_280k_5fold": 312.4, "estimated_per_variant_280k_fixed600": 360, "estimated_total_seconds": 720},
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    started_all = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    if train[ID].duplicated().any() or train[TARGET].isna().any():
        raise ValueError("Invalid train IDs or labels")
    if SAMPLE_ROWS < len(train):
        train, _ = train_test_split(train, train_size=SAMPLE_ROWS, stratify=train[TARGET], random_state=SAMPLE_SEED)
        train = train.reset_index(drop=True)
    y = train[TARGET].astype(np.int8).to_numpy()

    from feature_screen import feature_configs

    feature_names, categorical = feature_configs()["raw"]
    raw = train[feature_names].copy()
    required = {DISTANCE, CLASS, TRAVEL}
    if not required.issubset(raw.columns):
        raise ValueError(f"Raw feature set is missing B20 columns: {sorted(required-set(raw.columns))}")

    splits = list(StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=OUTER_SEED).split(np.zeros(len(y)), y))
    fold_rows, frames = [], []
    for fold_no, (tr_idx, va_idx) in enumerate(splits, start=1):
        started_fold = time.perf_counter()
        tr_raw = raw.iloc[tr_idx].reset_index(drop=True)
        va_raw = raw.iloc[va_idx].reset_index(drop=True)
        y_tr, y_va = y[tr_idx], y[va_idx]
        xtr_base_df = pd.get_dummies(tr_raw, columns=categorical, dtype=np.int8)
        xva_base_df = pd.get_dummies(va_raw, columns=categorical, dtype=np.int8).reindex(columns=xtr_base_df.columns, fill_value=0)
        xtr_base = xtr_base_df.to_numpy(dtype=np.float32, copy=False)
        xva_base = xva_base_df.to_numpy(dtype=np.float32, copy=False)

        p_fold = {}
        times = {}
        for variant in ("A_raw", "B_exact_distance_smooth_20"):
            if variant == "A_raw":
                xtr, xva = xtr_base, xva_base
            else:
                te_tr, te_va = target_encoding(tr_raw, y_tr, va_raw, INNER_SEED_BASE + fold_no)
                xtr = np.column_stack([xtr_base, te_tr])
                xva = np.column_stack([xva_base, te_va])

            started_fit = time.perf_counter()
            model = lgb.LGBMClassifier(**PARAMS)
            model.fit(xtr, y_tr)
            p = model.predict_proba(xva)[:, 1].astype(np.float32)
            times[variant] = time.perf_counter() - started_fit
            p_fold[variant] = p
            auc = roc_auc_score(y_va, p)
            fold_rows.append({
                "fold": fold_no, "variant": variant, "auc": float(auc),
                "train_rows": len(tr_idx), "valid_rows": len(va_idx),
                "n_features": xtr.shape[1], "fit_predict_seconds": times[variant],
            })
            print(f"fold={fold_no}/{OUTER_FOLDS} {variant} auc={auc:.9f} seconds={times[variant]:.1f}", flush=True)
            del model, xtr, xva
            gc.collect()

        delta = p_fold["B_exact_distance_smooth_20"] - p_fold["A_raw"]
        frames.append(pd.DataFrame({
            ID: train.iloc[va_idx][ID].to_numpy(), "fold": fold_no, TARGET: y_va,
            "A_raw_probability": p_fold["A_raw"],
            "B20_probability": p_fold["B_exact_distance_smooth_20"],
            "B20_minus_A_probability": delta,
        }))
        del tr_raw, va_raw, xtr_base, xva_base, xtr_base_df, xva_base_df, p_fold
        gc.collect()
        fold_wall = time.perf_counter() - started_fold
        fold_rows[-2]["fold_wall_seconds"] = fold_wall
        fold_rows[-1]["fold_wall_seconds"] = fold_wall
        print(f"outer fold {fold_no} elapsed={fold_wall:.1f}s", flush=True)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT / "fold_results.csv", index=False)
    pred_df = pd.concat(frames, ignore_index=True)
    pred_df.to_csv(OUT / "outer_oof_predictions.csv", index=False)
    piv = fold_df.pivot(index="fold", columns="variant", values="auc")
    piv["delta_B20_minus_A"] = piv["B_exact_distance_smooth_20"] - piv["A_raw"]
    piv.to_csv(OUT / "paired_fold_auc.csv")
    summary = []
    for variant in ("A_raw", "B_exact_distance_smooth_20"):
        vals = fold_df.loc[fold_df.variant == variant, "auc"].to_numpy()
        summary.append({"variant": variant, "mean_auc": float(vals.mean()),
                        "std_population": float(vals.std(ddof=0)), "std_sample": float(vals.std(ddof=1)),
                        "fold_aucs": ";".join(f"{x:.9f}" for x in vals),
                        "total_fit_predict_seconds": float(fold_df.loc[fold_df.variant == variant, "fit_predict_seconds"].sum())})
    deltas = piv["delta_B20_minus_A"].to_numpy()
    summary.append({"variant": "B20_minus_A", "mean_auc": float(deltas.mean()),
                    "std_population": float(deltas.std(ddof=0)), "std_sample": float(deltas.std(ddof=1)),
                    "fold_aucs": ";".join(f"{x:.9f}" for x in deltas),
                    "total_fit_predict_seconds": float(fold_df.fit_predict_seconds.sum())})
    pd.DataFrame(summary).to_csv(OUT / "cv_summary.csv", index=False)
    plan["actual_total_fit_predict_seconds"] = float(fold_df.fit_predict_seconds.sum())
    plan["actual_total_wall_seconds"] = float(time.perf_counter() - started_all)
    (OUT / "experiment_config.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print("CV summary\n", pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
