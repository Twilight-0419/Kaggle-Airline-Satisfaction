"""One motivated second-level experiment based only on genuine first-layer OOF scores.

Motivation: subgroup OOF AUC differs consistently by Type of Travel, while mean
calibration residuals are near zero. Test whether small Class/Travel-specific score
slopes improve held-out ranking. No test data or labels are read by this script.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STACK = ROOT / "outputs" / "public_stack"
PUBLIC = next(ROOT.glob("某个公开算法*")) / "extra_oof" / "s6e10-oof"
OUT = ROOT / "outputs" / "oof_error_analysis"
ID, TARGET = "id", "satisfaction"
OUTER_SEED = 2026
FOLDS = 5
C = 0.1  # fixed before evaluation; no hyperparameter search


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan


def make_features(frame, score):
    z = logit(np.clip(np.asarray(score, dtype=np.float64), 1e-6, 1 - 1e-6))
    cats = pd.get_dummies(frame[["Class", "Type of Travel"]], drop_first=True, dtype=np.float64)
    x = pd.DataFrame({"logit_oof_score": z})
    for col in cats:
        x[col] = cats[col].to_numpy()
        x[f"logit_x_{col}"] = z * cats[col].to_numpy()
    return x.to_numpy(dtype=np.float64), list(x.columns)


def main():
    train = pd.read_csv(DATA / "train.csv", usecols=[ID, TARGET, "Class", "Type of Travel"])
    if not train[ID].is_unique or train[TARGET].isna().any():
        raise ValueError("Invalid training IDs/labels")
    y = train[TARGET].astype(np.int8).to_numpy()
    n = len(train)
    best = pd.read_csv(STACK / "fine_triple_cv.csv").iloc[0]
    weights = {
        "ext_l2": float(best["w_ext_l2"]),
        "xgboost_meta_tuned": float(best["w_xgb_meta"]),
        "najiama_05": float(best["w_najiama05"]),
    }
    p0 = np.asarray(np.load(STACK / "ext_l2_oof.npy"), dtype=np.float64)
    p1 = np.asarray(np.load(STACK / "xgboost_meta_tuned_oof.npy"), dtype=np.float64)
    public = pd.read_csv(PUBLIC / "05_blend_oof.csv", usecols=[ID, TARGET])
    if not public[ID].is_unique or len(public) != n:
        raise ValueError("Public OOF rows/IDs invalid")
    locations = pd.Index(public[ID].to_numpy()).get_indexer(train[ID].to_numpy())
    if (locations < 0).any():
        raise ValueError("Public OOF predictions do not cover train IDs")
    p2 = public[TARGET].to_numpy(dtype=np.float64)[locations]
    if any(a.shape != (n,) or not np.isfinite(a).all() for a in [p0, p1, p2]):
        raise ValueError("First-layer OOF predictions have invalid shape or values")
    score = weights["ext_l2"] * p0 + weights["xgboost_meta_tuned"] * p1 + weights["najiama_05"] * p2
    X, feature_names = make_features(train, score)
    folds = list(StratifiedKFold(FOLDS, shuffle=True, random_state=OUTER_SEED).split(np.zeros(n), y))

    oof_level2 = np.full(n, np.nan, dtype=np.float64)
    fold_rows, group_rows = [], []
    for fold, (tr, va) in enumerate(folds, start=1):
        model = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=500, random_state=42)
        model.fit(X[tr], y[tr])  # training inputs are first-layer OOF predictions only
        p_meta = model.predict_proba(X[va])[:, 1]
        oof_level2[va] = p_meta
        base_auc = safe_auc(y[va], score[va])
        meta_auc = safe_auc(y[va], p_meta)
        fold_rows.append({
            "fold": fold, "n_train": len(tr), "n_valid": len(va),
            "baseline_oof_auc": base_auc, "level2_oof_auc": meta_auc,
            "delta_auc": meta_auc - base_auc,
            "baseline_brier": float(np.mean((y[va] - score[va]) ** 2)),
            "level2_brier": float(np.mean((y[va] - p_meta) ** 2)),
            "C": C, "n_meta_features": X.shape[1],
        })
        for travel in train["Type of Travel"].drop_duplicates():
            m = va[train["Type of Travel"].to_numpy()[va] == travel]
            if len(m):
                group_rows.append({
                    "fold": fold, "group": str(travel), "n": len(m),
                    "baseline_auc": safe_auc(y[m], score[m]),
                    "level2_auc": safe_auc(y[m], p_meta[np.isin(va, m)]),
                })
        print(f"fold={fold} baseline_auc={base_auc:.7f} level2_auc={meta_auc:.7f} delta={meta_auc-base_auc:+.7f}", flush=True)

    if not np.isfinite(oof_level2).all():
        raise ValueError("Incomplete outer-CV predictions")
    fold_df = pd.DataFrame(fold_rows)
    group_df = pd.DataFrame(group_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(OUT / "level2_outer_cv_folds.csv", index=False)
    group_df.to_csv(OUT / "level2_outer_cv_travel_groups.csv", index=False)
    np.save(OUT / "level2_outer_cv_oof.npy", oof_level2.astype(np.float32))
    summary = {
        "experiment": "logistic score calibration with Class/Type of Travel intercepts and score interactions",
        "first_layer_prediction": "current best blend reconstructed exclusively from genuine OOF predictions",
        "first_layer_weights": weights,
        "outer_cv": {"n_splits": FOLDS, "shuffle": True, "random_state": OUTER_SEED},
        "level2": {"model": "LogisticRegression", "C": C, "features": feature_names},
        "baseline_mean_fold_auc": float(fold_df.baseline_oof_auc.mean()),
        "baseline_std_fold_auc": float(fold_df.baseline_oof_auc.std(ddof=1)),
        "level2_mean_fold_auc": float(fold_df.level2_oof_auc.mean()),
        "level2_std_fold_auc": float(fold_df.level2_oof_auc.std(ddof=1)),
        "mean_delta_auc": float(fold_df.delta_auc.mean()),
        "fold_wins": int((fold_df.delta_auc > 0).sum()),
        "limitation": "Outer CV trains/evaluates the level-2 process using first-layer OOF artifacts. The original base predictors are not refit inside each outer fold, so this is not a fully nested retraining of every upstream public model.",
    }
    (OUT / "level2_outer_cv_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
