"""Generate a fresh submission from the best validated simple blend.

Rejected stacking/ID/distance/pseudo-label experiments are deliberately not
included; this writes a new candidate and preserves every existing submission.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STACK = ROOT / "outputs" / "public_stack"
OUT = ROOT / "outputs" / "final_candidates"
PUBLIC = next(ROOT.glob("某个公开算法*")) / "extra_oof" / "s6e10-oof"
ID, TARGET = "id", "satisfaction"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(DATA / "sample_submission.csv")
    test = pd.read_csv(DATA / "test.csv", usecols=[ID])
    if not sample[ID].is_unique or not test[ID].is_unique:
        raise ValueError("Duplicate test IDs")
    if not sample[ID].reset_index(drop=True).equals(test[ID].reset_index(drop=True)):
        raise ValueError("sample_submission and test IDs/order differ")

    weight_row = pd.read_csv(STACK / "fine_triple_cv.csv").iloc[0]
    components = {
        "ext_l2": np.load(STACK / "ext_l2_test.npy").astype(np.float64),
        "xgboost_meta_tuned": np.load(STACK / "xgboost_meta_tuned_test.npy").astype(np.float64),
    }
    public = pd.read_csv(PUBLIC / "05_submission.csv", usecols=[ID, TARGET])
    if not public[ID].is_unique:
        raise ValueError("Public prediction component has duplicate IDs")
    positions = pd.Index(public[ID].to_numpy()).get_indexer(sample[ID].to_numpy())
    if (positions < 0).any():
        raise ValueError("Public prediction component is missing test IDs")
    components["najiama_05"] = public[TARGET].to_numpy(dtype=np.float64)[positions]
    for name, p in components.items():
        if p.shape != (len(sample),) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError(f"Invalid component prediction: {name}")

    weights = {
        "ext_l2": float(weight_row["w_ext_l2"]),
        "xgboost_meta_tuned": float(weight_row["w_xgb_meta"]),
        "najiama_05": float(weight_row["w_najiama05"]),
    }
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError(f"Blend weights do not sum to one: {weights}")
    probability = sum(weights[k] * components[k] for k in weights)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Final predictions are invalid")

    submission = pd.DataFrame({ID: sample[ID].to_numpy(), TARGET: probability})
    if submission.columns.tolist() != sample.columns.tolist():
        raise ValueError("Submission columns differ from sample_submission")
    output = OUT / "submission_best_after_experiments.csv"
    submission.to_csv(output, index=False)

    meta = {
        "output": str(output),
        "rows": len(submission),
        "columns": submission.columns.tolist(),
        "weights": weights,
        "oof_blend_cv_mean": float(weight_row["mean_cv"]),
        "oof_blend_cv_std": float(weight_row["std_cv"]),
        "oof_blend_auc": float(weight_row["oof_auc"]),
        "prediction_range": [float(probability.min()), float(probability.max())],
        "pseudo_labeling_used": False,
        "existing_submissions_overwritten": False,
        "selection_basis": "Highest existing fine_triple_cv.csv row; downstream stacking and pseudo-label CV had no stable gain; ID/distance experiments retained only when validated and are not components of this saved simple blend.",
    }
    (OUT / "submission_best_after_experiments.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
