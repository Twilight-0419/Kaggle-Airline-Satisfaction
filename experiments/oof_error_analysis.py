"""Error analysis for the current best blend using train-only OOF predictions."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STACK = ROOT / "outputs" / "public_stack"
PUBLIC = next(ROOT.glob("某个公开算法*")) / "extra_oof" / "s6e10-oof"
OUT = ROOT / "outputs" / "oof_error_analysis"
ID, TARGET = "id", "satisfaction"


def safe_auc(y, p):
    y = np.asarray(y)
    counts = np.bincount(y.astype(np.int8), minlength=2)
    # Avoid presenting a noisy within-bin AUC when either class has tiny support.
    return float(roc_auc_score(y, p)) if np.unique(y).size == 2 and counts.min() >= 20 else np.nan


def group_row(frame, mask, group_kind, group_name, fold="all"):
    d = frame.loc[mask]
    if d.empty:
        return None
    y, p = d[TARGET].to_numpy(), d["oof_prediction"].to_numpy()
    pred_label = (p >= 0.5).astype(np.int8)
    return {
        "fold": fold,
        "group_type": group_kind,
        "group": str(group_name),
        "n": len(d),
        "true_satisfied_rate": float(y.mean()),
        "mean_predicted_probability": float(p.mean()),
        "mean_residual_y_minus_p": float(np.mean(y - p)),
        "auc_within_group": safe_auc(y, p),
        "threshold_0_5_error_rate": float(np.mean(pred_label != y)),
        "false_negative_rate_within_satisfied": float(np.mean(pred_label[y == 1] == 0)) if np.any(y == 1) else np.nan,
        "false_positive_rate_within_dissatisfied": float(np.mean(pred_label[y == 0] == 1)) if np.any(y == 0) else np.nan,
        "brier_score": float(brier_score_loss(y, p)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Intentionally read only training labels and the columns needed for diagnosis.
    cols = [ID, TARGET, "Class", "Type of Travel", "Flight Distance"]
    train = pd.read_csv(DATA / "train.csv", usecols=cols)
    n = len(train)
    if not train[ID].is_unique or train[TARGET].isna().any():
        raise ValueError("Train IDs/labels invalid")

    weight_row = pd.read_csv(STACK / "fine_triple_cv.csv").iloc[0]
    weights = {
        "ext_l2": float(weight_row["w_ext_l2"]),
        "xgboost_meta_tuned": float(weight_row["w_xgb_meta"]),
        "najiama_05": float(weight_row["w_najiama05"]),
    }
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError(f"Blend weights must sum to 1: {weights}")

    p_ext = np.asarray(np.load(STACK / "ext_l2_oof.npy"), dtype=np.float64)
    p_xgb = np.asarray(np.load(STACK / "xgboost_meta_tuned_oof.npy"), dtype=np.float64)
    csv = pd.read_csv(PUBLIC / "05_blend_oof.csv", usecols=[ID, TARGET])
    if not csv[ID].is_unique or len(csv) != n:
        raise ValueError("Najiama OOF has duplicate/mismatched IDs")
    loc = pd.Index(csv[ID].to_numpy()).get_indexer(train[ID].to_numpy())
    if (loc < 0).any():
        raise ValueError("Najiama OOF IDs do not cover train IDs")
    p_naji = csv[TARGET].to_numpy(dtype=np.float64)[loc]
    for name, p in [("ext_l2", p_ext), ("xgboost_meta_tuned", p_xgb), ("najiama_05", p_naji)]:
        if p.shape != (n,) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError(f"Invalid {name} OOF predictions")
    p = weights["ext_l2"] * p_ext + weights["xgboost_meta_tuned"] * p_xgb + weights["najiama_05"] * p_naji
    y = train[TARGET].to_numpy(dtype=np.int8)
    frame = train.copy()
    frame["oof_prediction"] = p
    frame["residual_y_minus_p"] = y - p
    frame["abs_error"] = np.abs(y - p)
    frame["predicted_label_0_5"] = (p >= 0.5).astype(np.int8)
    frame["is_error_at_0_5"] = frame["predicted_label_0_5"].to_numpy() != y
    fold_id = np.full(n, -1, dtype=np.int8)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_splits = list(cv.split(np.zeros(n), y))
    for fold, (_, va) in enumerate(fold_splits, start=1):
        fold_id[va] = fold
    frame["oof_fold"] = fold_id

    overall = {
        "n": n,
        "oof_auc": safe_auc(y, p),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "threshold_0_5_error_rate": float(frame["is_error_at_0_5"].mean()),
        "weights": weights,
        "weight_source": "outputs/public_stack/fine_triple_cv.csv top row (5-seed CV winner)",
    }
    (OUT / "overall_metrics.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    frame.to_csv(OUT / "row_level_oof_analysis.csv", index=False)

    # Calibration/confidence bins. Within-bin AUC is local ranking discrimination;
    # it is NA when the bin has only one class or too few rows.
    edges = [0, .01, .025, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, .95, .975, .99, 1.0000001]
    frame["probability_bin"] = pd.cut(frame["oof_prediction"], bins=edges, right=False, include_lowest=True)
    prob_rows = []
    for interval, d in frame.groupby("probability_bin", observed=False):
        if d.empty:
            continue
        yb, pb = d[TARGET].to_numpy(), d["oof_prediction"].to_numpy()
        prob_rows.append({
            "probability_interval": str(interval), "n": len(d),
            "true_satisfied_rate": float(yb.mean()),
            "mean_predicted_probability": float(pb.mean()),
            "calibration_gap_true_minus_pred": float(yb.mean() - pb.mean()),
            "within_interval_auc": safe_auc(yb, pb),
            "threshold_0_5_error_rate": float(np.mean((pb >= .5) != yb)),
            "brier_score": float(brier_score_loss(yb, pb)),
        })
    pd.DataFrame(prob_rows).to_csv(OUT / "probability_interval_metrics.csv", index=False)

    # Confidence/error sets and a compact summary; no test labels are read.
    high_wrong = frame[((p >= .90) & (y == 0)) | ((p <= .10) & (y == 1))].copy()
    high_wrong["error_kind"] = np.where(high_wrong["satisfaction"].eq(0), "confident_false_positive", "confident_false_negative")
    high_wrong.sort_values("abs_error", ascending=False).to_csv(OUT / "high_confidence_errors.csv", index=False)
    low_conf = frame[(p >= .45) & (p <= .55)].copy()
    low_conf.sort_values("abs_error", ascending=False).to_csv(OUT / "low_confidence_samples.csv", index=False)
    conf_sets = [
        ("confident_false_positive_p_ge_0.90", (p >= .90) & (y == 0)),
        ("confident_false_negative_p_le_0.10", (p <= .10) & (y == 1)),
        ("very_confident_false_positive_p_ge_0.975", (p >= .975) & (y == 0)),
        ("very_confident_false_negative_p_le_0.025", (p <= .025) & (y == 1)),
        ("low_confidence_0.45_to_0.55", (p >= .45) & (p <= .55)),
    ]
    confidence_summary = pd.DataFrame([
        {"set": name, "n": int(mask.sum()), "share_of_all": float(mask.mean()),
         "true_satisfied_rate": float(y[mask].mean()) if mask.any() else np.nan,
         "mean_probability": float(p[mask].mean()) if mask.any() else np.nan,
         "threshold_0_5_error_rate": float(np.mean((p[mask] >= .5) != y[mask])) if mask.any() else np.nan}
        for name, mask in conf_sets
    ])
    confidence_summary.to_csv(OUT / "confidence_summary.csv", index=False)

    # Subgroup discrimination/calibration/error metrics, plus foldwise stability.
    distance_bins = pd.qcut(frame["Flight Distance"], q=10, duplicates="drop")
    frame["flight_distance_decile"] = distance_bins.astype(str)
    group_rows, fold_group_rows = [], []
    for col, group_type in [("Class", "Class"), ("Type of Travel", "Type of Travel"), ("flight_distance_decile", "Flight Distance decile")]:
        for value in frame[col].drop_duplicates().tolist():
            mask = frame[col].eq(value).to_numpy()
            row = group_row(frame, mask, group_type, value)
            if row:
                group_rows.append(row)
            for fold in range(1, 6):
                foldmask = mask & (fold_id == fold)
                frow = group_row(frame, foldmask, group_type, value, fold)
                if frow:
                    fold_group_rows.append(frow)
    pd.DataFrame(group_rows).to_csv(OUT / "subgroup_error_metrics.csv", index=False)
    fold_group_df = pd.DataFrame(fold_group_rows)
    fold_group_df.to_csv(OUT / "subgroup_error_metrics_by_fold.csv", index=False)
    stability_rows = []
    for (kind, group), d in fold_group_df.groupby(["group_type", "group"]):
        residual = d["mean_residual_y_minus_p"].to_numpy()
        stability_rows.append({
            "group_type": kind, "group": group,
            "mean_residual_across_folds": float(np.mean(residual)),
            "std_residual_across_folds": float(np.std(residual, ddof=1)),
            "min_fold_residual": float(np.min(residual)),
            "max_fold_residual": float(np.max(residual)),
            "dominant_residual_sign_folds": int(max(np.sum(residual >= 0), np.sum(residual < 0))),
            "fold_count": len(d),
            "mean_group_auc": float(d["auc_within_group"].mean()),
            "std_group_auc": float(d["auc_within_group"].std(ddof=1)),
            "mean_error_rate": float(d["threshold_0_5_error_rate"].mean()),
            "std_error_rate": float(d["threshold_0_5_error_rate"].std(ddof=1)),
        })
    pd.DataFrame(stability_rows).to_csv(OUT / "subgroup_fold_stability.csv", index=False)

    # Fold-level overall results provide a reference for later experiments.
    fold_rows = []
    for fold, (_, va) in enumerate(fold_splits, start=1):
        fold_rows.append({"fold": fold, "n": len(va), "auc": safe_auc(y[va], p[va]), "brier_score": brier_score_loss(y[va], p[va]), "threshold_0_5_error_rate": np.mean((p[va] >= .5) != y[va])})
    pd.DataFrame(fold_rows).to_csv(OUT / "overall_fold_metrics.csv", index=False)
    print(json.dumps(overall, indent=2))
    print("probability intervals:\n", pd.DataFrame(prob_rows).to_string(index=False))
    print("subgroups:\n", pd.DataFrame(group_rows).to_string(index=False))
    print("confidence sets:\n", confidence_summary.to_string(index=False))
    print("saved", OUT)


if __name__ == "__main__":
    main()
