"""Standalone diagnostics and blend-weight experiment for existing predictions.

All predictions are loaded from existing project artifacts. No model is trained here.
Full-OOF grid scores are selection scores, not independent validation scores.
Outer-CV estimates the weight-selection step on a fixed prediction pool, but is not
fully nested because base OOF models are not retrained inside every outer split.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "fusion_study"
STACK = ROOT / "outputs" / "public_stack"
PUBLIC = next(ROOT.glob("某个公开算法*")) / "extra_oof" / "s6e10-oof"
TARGET = "satisfaction"
ID = "id"

# These are existing prediction artifacts with training scripts/notebook sources.
# NumPy artifacts were written in data/train.csv and data/test.csv row order.
ARRAY_MODELS = {
    "ext_l2": ("ext_l2_oof.npy", "ext_l2_test.npy", "experiments/stack_ext_l2.py"),
    "xgb_meta_tuned": ("xgboost_meta_tuned_oof.npy", "xgboost_meta_tuned_test.npy", "experiments/xgb_meta_tuned.py"),
    "lgb_meta": ("lightgbm_meta_oof.npy", "lightgbm_meta_test.npy", "experiments/lgbm_meta_screen.py"),
    "cat_meta_public24": ("catboost_meta_public24_oof.npy", "catboost_meta_public24_test.npy", "experiments/catboost_meta_public24.py"),
    "public24_stack": ("public24_stack_oof.npy", "public24_stack_test.npy", "experiments/stack_public24.py"),
    "external_lgb": ("external_lgb_oof.npy", "external_lgb_test.npy", "experiments/external_lgbm_full.py"),
    "lgb_aux_route": ("lgb_aux_route_oof.npy", "lgb_aux_route_test.npy", "experiments/lgb_aux_route.py"),
    "cat_external": ("catboost_external_oof.npy", "catboost_external_test.npy", "experiments/catboost_external.py"),
    "xgb_meta": ("xgboost_meta_oof.npy", "xgboost_meta_test.npy", "experiments/xgb_meta_screen.py"),
    "tfm_stack": ("tfm_stack_oof.npy", "tfm_stack_test.npy", "experiments/tfm_stack.py"),
    "level2": ("level2_oof.npy", "level2_test.npy", "experiments/meta_level2.py"),
    "refine_ext3": ("refine_ext3_oof.npy", "refine_ext3_test.npy", "experiments/refine_ext3.py"),
}
# Predeclared pool for simplex search and outer-fold weight selection.
# Keeping this fixed avoids selecting the pool itself on the same OOF labels.
SEARCH_POOL = ["ext_l2", "xgb_meta_tuned", "lgb_meta", "cat_meta_public24", "najiama_05"]
FULL_GRID_STEP = 0.1
OUTER_GRID_STEP = 0.2
OUTER_SEED = 31415
OUTER_FOLDS = 5


def _check_probability(name: str, arr: np.ndarray, expected: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if len(arr) != expected:
        raise ValueError(f"{name}: {len(arr)} predictions; expected {expected}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}: contains NaN or infinite values")
    if ((arr < 0) | (arr > 1)).any():
        raise ValueError(f"{name}: probabilities outside [0, 1]")
    return arr


def _load_id_csv(path: Path, expected_ids: pd.Series, name: str) -> np.ndarray:
    frame = pd.read_csv(path)
    if ID not in frame or TARGET not in frame:
        raise ValueError(f"{path}: expected columns {ID!r} and {TARGET!r}")
    if not frame[ID].is_unique:
        raise ValueError(f"{path}: duplicate IDs")
    if len(frame) != len(expected_ids):
        raise ValueError(f"{path}: IDs do not match the competition rows")
    positions = pd.Index(frame[ID].to_numpy()).get_indexer(expected_ids.to_numpy())
    if (positions < 0).any():
        raise ValueError(f"{path}: missing predictions after ID alignment")
    return _check_probability(name, frame[TARGET].to_numpy()[positions], len(expected_ids))


def load_predictions(train: pd.DataFrame, test: pd.DataFrame):
    models: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    sources: dict[str, str] = {}
    n_train, n_test = len(train), len(test)
    for name, (oof_file, test_file, source) in ARRAY_MODELS.items():
        op, tp = STACK / oof_file, STACK / test_file
        if not (op.exists() and tp.exists()):
            continue
        oof = _check_probability(name + " OOF", np.load(op), n_train)
        pred = _check_probability(name + " test", np.load(tp), n_test)
        models[name] = (oof, pred)
        sources[name] = source + " (row order checked against generation code; arrays have no IDs)"
        print(f"loaded array pair {name}", flush=True)

    # Public blend predictions are CSVs with IDs, so explicitly reindex both splits.
    for code in ["04", "05", "06"]:
        oof_path, test_path = PUBLIC / f"{code}_blend_oof.csv", PUBLIC / f"{code}_submission.csv"
        if oof_path.exists() and test_path.exists():
            name = f"najiama_{code}"
            models[name] = (
                _load_id_csv(oof_path, train[ID], name + " OOF"),
                _load_id_csv(test_path, test[ID], name + " test"),
            )
            sources[name] = str(oof_path.relative_to(ROOT)) + " (ID-aligned)"
            print(f"loaded ID-aligned pair {name}", flush=True)
    if not models:
        raise RuntimeError("No complete OOF/test prediction pairs found")
    return models, sources


def simplex_grid(n_models: int, step: float):
    units = round(1 / step)
    if not math.isclose(units * step, 1.0, abs_tol=1e-9):
        raise ValueError("Grid step must divide 1 exactly")
    def rec(left: int, n: int, prefix=()):
        if n == 1:
            yield np.asarray(prefix + (left,), dtype=np.float64) / units
            return
        for v in range(left + 1):
            yield from rec(left - v, n - 1, prefix + (v,))
    return list(rec(units, n_models))


def score_grid(P: np.ndarray, y: np.ndarray, indices: np.ndarray, weights: list[np.ndarray]):
    best_auc, best_w = -1.0, None
    for w in weights:
        auc = roc_auc_score(y[indices], P[indices] @ w)
        if auc > best_auc:
            best_auc, best_w = float(auc), w.copy()
    return best_auc, best_w


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv", usecols=[ID, TARGET])
    test = pd.read_csv(DATA / "test.csv", usecols=[ID])
    if train[ID].duplicated().any() or test[ID].duplicated().any():
        raise ValueError("Competition train/test IDs are not unique")
    if train[TARGET].isna().any() or not set(train[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Labels are missing or not binary")
    y = train[TARGET].astype(np.int8).to_numpy()
    models, sources = load_predictions(train, test)

    # Persist provenance, order, and validation caveats alongside the results.
    config = {
        "n_train": len(train), "n_test": len(test), "train_order": "data/train.csv row order",
        "test_order": "data/test.csv row order", "search_pool": SEARCH_POOL,
        "full_grid_step": FULL_GRID_STEP, "outer_grid_step": OUTER_GRID_STEP,
        "outer_cv": {"splitter": "StratifiedKFold", "n_splits": OUTER_FOLDS, "shuffle": True, "random_state": OUTER_SEED},
        "same_oof_warning": "Full-OOF grid_auc is a weight-selection score and is not independent validation.",
        "outer_cv_caveat": "Fusion weights are selected on outer-training OOF rows and evaluated on held-out rows. Base OOF models were not retrained inside each outer split, so this is a fusion-level estimate, not fully nested stacking CV; base-model split overlap may cause some optimism.",
        "prediction_sources": sources,
    }
    missing_pool = [n for n in SEARCH_POOL if n not in models]
    if missing_pool:
        raise FileNotFoundError(f"Search pool prediction files missing: {missing_pool}")
    (OUT / "experiment_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    names = list(models)
    P_all = np.column_stack([models[n][0] for n in names])
    T_all = np.column_stack([models[n][1] for n in names])
    single_rows = []
    for j, name in enumerate(names):
        single_rows.append({"model": name, "oof_auc": roc_auc_score(y, P_all[:, j]), "source": sources[name]})
    pd.DataFrame(single_rows).sort_values("oof_auc", ascending=False).to_csv(OUT / "single_model_metrics.csv", index=False)

    pair_rows = []
    for i, j in combinations(range(len(names)), 2):
        p = (P_all[:, i] + P_all[:, j]) / 2
        pair_rows.append({
            "model_a": names[i], "model_b": names[j],
            "pearson_oof_correlation": np.corrcoef(P_all[:, i], P_all[:, j])[0, 1],
            "equal_weight_oof_auc": roc_auc_score(y, p),
        })
    pd.DataFrame(pair_rows).sort_values("equal_weight_oof_auc", ascending=False).to_csv(OUT / "pairwise_correlation_equal_auc.csv", index=False)

    pool = SEARCH_POOL
    P = np.column_stack([models[n][0] for n in pool])
    T = np.column_stack([models[n][1] for n in pool])
    equal_w = np.full(len(pool), 1 / len(pool))
    full_weights = simplex_grid(len(pool), FULL_GRID_STEP)
    full_rows = []
    for w in full_weights:
        full_rows.append({"oof_auc_same_data_search": roc_auc_score(y, P @ w), **{f"w_{n}": v for n, v in zip(pool, w)}})
    full_df = pd.DataFrame(full_rows).sort_values("oof_auc_same_data_search", ascending=False).reset_index(drop=True)
    full_df.to_csv(OUT / "full_oof_simplex_grid.csv", index=False)
    best_row = full_df.iloc[0]
    best_w = np.asarray([best_row[f"w_{n}"] for n in pool], dtype=np.float64)
    equal_auc = roc_auc_score(y, P @ equal_w)

    # Outer folds evaluate the *weight-selection procedure*, optimizing only on outer-train rows.
    folds = list(StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=OUTER_SEED).split(np.zeros(len(y)), y))
    outer_weights = simplex_grid(len(pool), OUTER_GRID_STEP)
    fold_rows, cv_oof = [], np.full(len(y), np.nan, dtype=np.float64)
    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        train_search_auc, w = score_grid(P, y, tr_idx, outer_weights)
        equal_pred = P[va_idx] @ equal_w
        tuned_pred = P[va_idx] @ w
        cv_oof[va_idx] = tuned_pred
        row = {
            "fold": fold,
            "equal_weight_auc": roc_auc_score(y[va_idx], equal_pred),
            "tuned_weight_auc": roc_auc_score(y[va_idx], tuned_pred),
            "inner_train_best_auc_selection_only": train_search_auc,
            "n_outer_train": len(tr_idx), "n_outer_valid": len(va_idx),
        }
        row.update({f"w_{n}": v for n, v in zip(pool, w)})
        fold_rows.append(row)
        print(f"outer fold {fold}/{OUTER_FOLDS}: equal={row['equal_weight_auc']:.7f}, tuned={row['tuned_weight_auc']:.7f}, weights={w.round(2).tolist()}", flush=True)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT / "outer_cv_fold_results.csv", index=False)
    summary = []
    for metric in ["equal_weight_auc", "tuned_weight_auc"]:
        summary.append({"method": metric, "mean_fold_auc": fold_df[metric].mean(), "std_fold_auc": fold_df[metric].std(ddof=1), "fold_aucs": ";".join(f"{x:.9f}" for x in fold_df[metric])})
    pd.DataFrame(summary).to_csv(OUT / "outer_cv_summary.csv", index=False)
    pd.DataFrame({ID: train[ID], "outer_cv_tuned_oof_prediction": cv_oof}).to_csv(OUT / "outer_cv_tuned_oof_predictions.csv", index=False)

    # New submissions are saved only under fusion_study; the best submission is untouched.
    test_grid = T @ best_w
    test_equal = T @ equal_w
    mean_outer_w = fold_df[[f"w_{n}" for n in pool]].mean().to_numpy(dtype=np.float64)
    test_outer = T @ mean_outer_w
    for filename, pred in [
        ("submission_full_oof_grid_best.csv", test_grid),
        ("submission_equal_weight.csv", test_equal),
        ("submission_outer_cv_mean_weights.csv", test_outer),
    ]:
        if len(pred) != len(test) or not np.isfinite(pred).all():
            raise ValueError(f"Invalid generated test prediction: {filename}")
        pd.DataFrame({ID: test[ID], TARGET: pred}).to_csv(OUT / filename, index=False)

    comparison = pd.DataFrame([
        {"method": "equal_weight_fixed_pool", "oof_auc_same_data": equal_auc, "score_is_independent": False, **{f"w_{n}": v for n, v in zip(pool, equal_w)}},
        {"method": "simplex_grid_best_full_oof", "oof_auc_same_data": float(best_row["oof_auc_same_data_search"]), "score_is_independent": False, **{f"w_{n}": v for n, v in zip(pool, best_w)}},
        {"method": "outer_cv_tuned_weights", "oof_auc_same_data": float(roc_auc_score(y, cv_oof)), "score_is_independent": False, "outer_cv_mean_auc": float(fold_df["tuned_weight_auc"].mean()), "outer_cv_std_auc": float(fold_df["tuned_weight_auc"].std(ddof=1)), **{f"w_{n}": v for n, v in zip(pool, mean_outer_w)}},
    ])
    comparison.to_csv(OUT / "blend_method_comparison.csv", index=False)
    print("\nSearch pool:", pool)
    print("Loaded models:", len(names), "Full-grid candidates:", len(full_weights), "Outer-grid candidates:", len(outer_weights))
    print("Equal-weight OOF AUC (descriptive):", f"{equal_auc:.9f}")
    print("Best full-OOF grid score (selection only):", f"{best_row['oof_auc_same_data_search']:.9f}", "weights", best_w.round(3).tolist())
    print("Outer-CV tuned mean/std:", f"{fold_df['tuned_weight_auc'].mean():.9f}", f"{fold_df['tuned_weight_auc'].std(ddof=1):.9f}")
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
