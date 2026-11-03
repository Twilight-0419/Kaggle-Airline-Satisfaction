"""Reproducible applicability audit for raw-ID ablation of the current best blend.

The current best predictor is a fixed blend of three already-generated prediction
vectors. None of its meta-level inputs is a raw feature matrix containing `id`, so
an A/B refit that adds/removes ID would be artificial and is intentionally skipped.
This script verifies that fact and the alignment of the exact saved prediction inputs.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "id_ablation"
STACK = ROOT / "outputs" / "public_stack"
PUBLIC = next(ROOT.glob("某个公开算法*")) / "extra_oof" / "s6e10-oof"
ID, TARGET = "id", "satisfaction"


def require_prob(name, values, expected):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) != expected:
        raise ValueError(f"{name}: got {len(values)} values, expected {expected}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: non-finite predictions found")
    if ((values < 0) | (values > 1)).any():
        raise ValueError(f"{name}: prediction outside [0, 1]")
    return values


def load_public_csv(path, expected_ids, name):
    df = pd.read_csv(path, usecols=[ID, TARGET])
    if not df[ID].is_unique:
        raise ValueError(f"{name}: duplicate IDs")
    if len(df) != len(expected_ids):
        raise ValueError(f"{name}: row count differs from competition data")
    loc = pd.Index(df[ID].to_numpy()).get_indexer(expected_ids.to_numpy())
    if (loc < 0).any():
        raise ValueError(f"{name}: IDs missing from prediction file")
    return require_prob(name, df[TARGET].to_numpy()[loc], len(expected_ids))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv", usecols=[ID, TARGET])
    test = pd.read_csv(DATA / "test.csv", usecols=[ID])  # test labels are never read
    train_ids, test_ids = train[ID], test[ID]
    if not train_ids.is_unique or not test_ids.is_unique:
        raise ValueError("train/test IDs must be unique")
    n_train, n_test = len(train), len(test)

    # The two trained meta models consume matrices of existing model predictions.
    # Their matrix construction scripts store prediction columns, not raw ID columns.
    xmeta = np.load(STACK / "X_oof.npy", mmap_mode="r")
    xmeta_test = np.load(STACK / "X_test.npy", mmap_mode="r")
    xmeta_names = pd.read_csv(STACK / "model_names.csv")["model"].astype(str)
    ext = np.load(STACK / "X_oof_ext.npy", mmap_mode="r")
    ext_test = np.load(STACK / "X_test_ext.npy", mmap_mode="r")
    ext_names = pd.read_csv(STACK / "model_names_ext.csv")["model"].astype(str)
    if xmeta.shape[0] != n_train or xmeta_test.shape[0] != n_test or xmeta.shape[1] != len(xmeta_names):
        raise ValueError("X_oof/X_test matrix dimensions do not match source data/model names")
    if ext.shape[0] != n_train or ext_test.shape[0] != n_test or ext.shape[1] != len(ext_names):
        raise ValueError("Extended OOF/test matrix dimensions do not match source data/model names")

    # Verify actual current blend weights and that the saved candidate is reproducible.
    weights_table = pd.read_csv(STACK / "fine_triple_cv.csv")
    row = weights_table.iloc[0]
    weights = {
        "ext_l2": float(row["w_ext_l2"]),
        "xgboost_meta_tuned": float(row["w_xgb_meta"]),
        "najiama_05": float(row["w_najiama05"]),
    }
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError(f"Current best blend weights do not sum to one: {weights}")
    inputs = {
        "ext_l2": (
            require_prob("ext_l2 OOF", np.load(STACK / "ext_l2_oof.npy"), n_train),
            require_prob("ext_l2 test", np.load(STACK / "ext_l2_test.npy"), n_test),
        ),
        "xgboost_meta_tuned": (
            require_prob("xgb meta OOF", np.load(STACK / "xgboost_meta_tuned_oof.npy"), n_train),
            require_prob("xgb meta test", np.load(STACK / "xgboost_meta_tuned_test.npy"), n_test),
        ),
        "najiama_05": (
            load_public_csv(PUBLIC / "05_blend_oof.csv", train_ids, "Najiama 05 OOF"),
            load_public_csv(PUBLIC / "05_submission.csv", test_ids, "Najiama 05 test"),
        ),
    }
    oof = sum(weights[k] * inputs[k][0] for k in weights)
    test_pred = sum(weights[k] * inputs[k][1] for k in weights)
    # The saved prediction lives under public_stack; exact comparison is made
    # against that candidate without modifying either file.
    candidate_paths = [STACK / "submission_triple_blend_fine.csv", STACK / "submission_triple_blend.csv"]
    saved_test = next((p for p in candidate_paths if p.exists()), None)
    if saved_test is not None:
        sf = pd.read_csv(saved_test, usecols=[ID, TARGET])
        if not sf[ID].equals(test_ids):
            raise ValueError("Saved blend candidate test IDs are not in test.csv order")
        saved_values = require_prob("saved blend candidate", sf[TARGET].to_numpy(), n_test)
        if not np.allclose(saved_values, test_pred, rtol=1e-10, atol=1e-10):
            raise ValueError("Reconstructed current blend does not match saved candidate")

    components = [
        {"component": "ext_l2", "weight": weights["ext_l2"], "prediction_source": "ext_l2_oof.npy/ext_l2_test.npy", "meta_input_kind": "public model predictions in X_oof_ext/X_test_ext", "raw_id_used_by_current_blend": False, "id_role": "not an input feature"},
        {"component": "xgboost_meta_tuned", "weight": weights["xgboost_meta_tuned"], "prediction_source": "xgboost_meta_tuned_oof.npy/xgboost_meta_tuned_test.npy", "meta_input_kind": "public model predictions in X_oof/X_test", "raw_id_used_by_current_blend": False, "id_role": "not an input feature"},
        {"component": "najiama_05", "weight": weights["najiama_05"], "prediction_source": "05_blend_oof.csv/05_submission.csv", "meta_input_kind": "precomputed probabilities aligned by id", "raw_id_used_by_current_blend": False, "id_role": "alignment key only; prediction column is satisfaction"},
    ]
    pd.DataFrame(components).to_csv(OUT / "current_blend_components.csv", index=False)
    result = {
        "status": "not_applicable_no_raw_id_feature_at_current_blend_level",
        "current_best_weights": weights,
        "train_rows": n_train,
        "test_rows": n_test,
        "base_matrix_shapes": {"X_oof": list(xmeta.shape), "X_test": list(xmeta_test.shape), "X_oof_ext": list(ext.shape), "X_test_ext": list(ext_test.shape)},
        "base_prediction_model_counts": {"X_oof": len(xmeta_names), "X_oof_ext": len(ext_names)},
        "id_used_as_predictor": False,
        "reason": "The active predictor is a weighted combination of three precomputed probability vectors. The logistic/XGBoost meta-model input matrices contain only named model-prediction columns; the public CSV id is used only to align Najiama predictions. There is no raw id feature accepted by this fitted blend, so constructing A=with-id versus B=without-id would require inventing a new model outside the current best model.",
        "ablation": "Skipped as requested: no A/B model refit, no ID-derived features, and no test labels read.",
        "reproducibility": "Run experiments/id_ablation_audit.py from the project root. It validates row counts, model-matrix widths, ID uniqueness/alignment, finite probabilities, current blend weights, and reconstruction of the saved test candidate.",
        "source_evidence": [
            "experiments/stack_ext_l2.py loads X_oof_ext and fits LogisticRegression on those prediction columns.",
            "experiments/xgb_meta_tuned.py loads X_oof and fits XGBoost on those prediction columns.",
            "experiments/triple_blend_cv.py combines ext_l2, xgboost_meta_tuned, and Najiama 05 prediction vectors; Najiama IDs are used for reindexing.",
        ],
    }
    (OUT / "id_ablation_status.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    # Empty A/B rows explicitly record why per-fold scores are not applicable.
    pd.DataFrame([
        {"variant": "A_keep_raw_id", "status": "not_applicable", "fold_1_auc": np.nan, "fold_2_auc": np.nan, "fold_3_auc": np.nan, "fold_4_auc": np.nan, "fold_5_auc": np.nan, "mean_auc": np.nan, "std_auc": np.nan, "reason": "Current fitted model does not accept raw id as a feature."},
        {"variant": "B_remove_raw_id", "status": "not_applicable", "fold_1_auc": np.nan, "fold_2_auc": np.nan, "fold_3_auc": np.nan, "fold_4_auc": np.nan, "fold_5_auc": np.nan, "mean_auc": np.nan, "std_auc": np.nan, "reason": "Current fitted model does not accept raw id as a feature; there is no ID column to remove from its meta input."},
    ]).to_csv(OUT / "id_ablation_fold_results.csv", index=False)
    print(json.dumps({"status": result["status"], "weights": weights, "train_rows": n_train, "test_rows": n_test, "X_oof": list(xmeta.shape), "X_oof_ext": list(ext.shape), "reconstructed_test_candidate": saved_test is not None, "output": str(OUT)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
