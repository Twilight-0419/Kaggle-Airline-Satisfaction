"""Strict pseudo-labeling simulation and isolated best-model test diagnostics.

No original labels, saved model artifacts, or baseline submissions are changed.
Outer validation labels are used only for scoring after predictions are fixed.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from feature_screen import feature_configs  # noqa: E402
from nested_stacking_oof import fit_base  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "outputs" / "pseudo_label"
STACK = ROOT / "outputs" / "public_stack"
PUBLIC_TEST = next(ROOT.glob("某个公开算法*")) / "extra_oof" / "s6e10-oof" / "05_submission.csv"
TARGET, ID = "satisfaction", "id"
MODELS = ["xgboost", "lightgbm", "catboost"]
N_SAMPLE, SEED, FOLDS = 280_000, 42, 5
P_POS, P_NEG, PSEUDO_WEIGHT = 0.99, 0.01, 0.10


def matrix(frame, columns, categorical):
    d = pd.get_dummies(frame[columns], columns=categorical, dtype=np.int8)
    return d.to_numpy(dtype=np.float32, copy=False), list(d.columns)


def pred_ensemble(x, y, x_eval, weights=None):
    ps = []
    for name in MODELS:
        model = fit_base(name, x, y)
        ps.append(model.predict_proba(x_eval)[:, 1].astype(np.float64))
        del model
        gc.collect()
    return np.column_stack(ps) @ (np.ones(3) / 3 if weights is None else weights)


def run_cv():
    started = time.perf_counter()
    train = pd.read_csv(DATA / "train.csv")
    sample, _ = train_test_split(train, train_size=N_SAMPLE, stratify=train[TARGET], random_state=SEED)
    sample = sample.reset_index(drop=True)
    y = sample[TARGET].astype(np.int8).to_numpy()
    cols, cats = feature_configs()["raw"]
    raw = sample[cols].copy()
    outer = list(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    rows, count_rows, pred_frames = [], [], []
    for fold, (tr_idx, va_idx) in enumerate(outer):
        pool_fold = (fold + 1) % FOLDS
        # Partition the outer-train indices into four balanced subfolds. One is
        # pseudo-label pool; the remaining three are the only labeled fit set.
        outer_tr_y = y[tr_idx]
        inner_parts = list(StratifiedKFold(FOLDS - 1, shuffle=True, random_state=SEED + 100 + fold).split(np.zeros(len(tr_idx)), outer_tr_y))
        # Use one 25%-sized held-aside pool, with 75% labeled fit data.
        local_train, local_pool = inner_parts[1][0], inner_parts[1][1]
        fit_idx, pool_idx = tr_idx[local_train], tr_idx[local_pool]
        y_fit = y[fit_idx]
        x_fit, dummy_cols = matrix(raw.iloc[fit_idx], cols, cats)
        def aligned(indices):
            df = pd.get_dummies(raw.iloc[indices][cols], columns=cats, dtype=np.int8).reindex(columns=dummy_cols, fill_value=0)
            return df.to_numpy(dtype=np.float32, copy=False)
        x_pool, x_val = aligned(pool_idx), aligned(va_idx)
        # Teacher never sees labels in either the pseudo pool or outer validation.
        p_pool = pred_ensemble(x_fit, y_fit, x_pool)
        p_val_teacher = pred_ensemble(x_fit, y_fit, x_val)
        selected = (p_pool >= P_POS) | (p_pool <= P_NEG)
        pseudo_y = (p_pool[selected] >= P_POS).astype(np.int8)
        pseudo_pos = int(pseudo_y.sum())
        pseudo_neg = int(len(pseudo_y) - pseudo_pos)
        n_pool = len(pool_idx)

        if selected.any():
            x_aug = np.concatenate([x_fit, x_pool[selected]], axis=0)
            y_aug = np.concatenate([y_fit, pseudo_y])
            w_aug = np.concatenate([np.ones(len(y_fit), dtype=np.float32), np.full(len(pseudo_y), PSEUDO_WEIGHT, dtype=np.float32)])
            ps = []
            for name in MODELS:
                model = fit_base(name, x_aug, y_aug, sample_weight=w_aug)
                ps.append(model.predict_proba(x_val)[:, 1].astype(np.float64))
                del model
                gc.collect()
            p_val_student = np.column_stack(ps).mean(axis=1)
        else:
            p_val_student = p_val_teacher.copy()
        # Access validation labels only after both predictions are finalized.
        auc_teacher = roc_auc_score(y[va_idx], p_val_teacher)
        auc_student = roc_auc_score(y[va_idx], p_val_student)
        rows.append({"outer_fold": fold + 1, "n_fit_labeled": len(fit_idx), "n_pseudo_pool": n_pool,
                     "n_outer_valid": len(va_idx), "teacher_auc": auc_teacher, "pseudo_student_auc": auc_student,
                     "delta_auc": auc_student - auc_teacher, "selected_pseudo": len(pseudo_y),
                     "pseudo_positive": pseudo_pos, "pseudo_negative": pseudo_neg,
                     "pseudo_positive_fraction": pseudo_pos / len(pseudo_y) if len(pseudo_y) else np.nan,
                     "pseudo_weight_each": PSEUDO_WEIGHT, "pseudo_weight_total": len(pseudo_y) * PSEUDO_WEIGHT})
        count_rows.append({"outer_fold": fold + 1, "pool_size": n_pool, "selected": len(pseudo_y),
                           "selected_fraction": len(pseudo_y) / n_pool, "positive": pseudo_pos, "negative": pseudo_neg,
                           "positive_fraction": pseudo_pos / len(pseudo_y) if len(pseudo_y) else np.nan,
                           "negative_fraction": pseudo_neg / len(pseudo_y) if len(pseudo_y) else np.nan,
                           "weight_each": PSEUDO_WEIGHT, "weight_total": len(pseudo_y) * PSEUDO_WEIGHT})
        pred_frames.append(pd.DataFrame({ID: sample.iloc[va_idx][ID].to_numpy(), "fold": fold + 1,
                                         "y_true": y[va_idx], "teacher_probability": p_val_teacher,
                                         "pseudo_student_probability": p_val_student}))
        print(f"fold={fold+1}/{FOLDS} teacher={auc_teacher:.6f} student={auc_student:.6f} delta={auc_student-auc_teacher:+.6f} pseudo={len(pseudo_y)}/{n_pool} (+{pseudo_pos}/-{pseudo_neg})", flush=True)
    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(OUT / "cv_fold_results.csv", index=False)
    pd.DataFrame(count_rows).to_csv(OUT / "cv_pseudo_label_counts.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(OUT / "cv_outer_predictions.csv", index=False)
    summary = []
    for col, name in [("teacher_auc", "teacher_no_pseudolabel"), ("pseudo_student_auc", "pseudo_label_student"), ("delta_auc", "student_minus_teacher")]:
        a = fold_df[col].to_numpy()
        summary.append({"method": name, "mean": float(a.mean()), "std_population": float(a.std(ddof=0)),
                        "std_sample": float(a.std(ddof=1)), "fold_values": ";".join(f"{v:.9f}" for v in a)})
    pd.DataFrame(summary).to_csv(OUT / "cv_summary.csv", index=False)
    return fold_df, summary, time.perf_counter() - started


def make_test_artifacts():
    test = pd.read_csv(DATA / "test.csv", usecols=lambda c: c == ID)
    if not test[ID].is_unique:
        raise ValueError("Test IDs are not unique")
    w = pd.read_csv(STACK / "fine_triple_cv.csv").iloc[0]
    p0 = np.load(STACK / "ext_l2_test.npy").astype(np.float64)
    p1 = np.load(STACK / "xgboost_meta_tuned_test.npy").astype(np.float64)
    pub = pd.read_csv(PUBLIC_TEST, usecols=[ID, TARGET])
    if not pub[ID].is_unique:
        raise ValueError("Public component test IDs are not unique")
    ix = pd.Index(pub[ID].to_numpy()).get_indexer(test[ID].to_numpy())
    if (ix < 0).any():
        raise ValueError("Public component predictions do not cover test IDs")
    p2 = pub[TARGET].to_numpy(dtype=np.float64)[ix]
    if any(len(p) != len(test) for p in [p0, p1, p2]):
        raise ValueError("Test prediction lengths do not match test IDs")
    p = float(w["w_ext_l2"]) * p0 + float(w["w_xgb_meta"]) * p1 + float(w["w_najiama05"]) * p2
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Best-model test probabilities are invalid")
    pos, neg = p >= P_POS, p <= P_NEG
    allpred = pd.DataFrame({ID: test[ID], "satisfaction_probability": p})
    allpred.to_csv(OUT / "test_best_model_probabilities.csv", index=False)
    selected = pd.DataFrame({ID: test.loc[pos | neg, ID].to_numpy(), "pseudo_label": (p[pos | neg] >= P_POS).astype(np.int8),
                             "satisfaction_probability": p[pos | neg], "pseudo_weight": PSEUDO_WEIGHT})
    selected.to_csv(OUT / "test_pseudo_labels_candidate.csv", index=False)
    summary = {"prediction_source": "current best simple blend: ext_l2 + xgboost_meta_tuned + public Najiama 05 submission, weights from fine_triple_cv.csv",
               "weights": {"ext_l2": float(w["w_ext_l2"]), "xgboost_meta_tuned": float(w["w_xgb_meta"]), "najiama_05": float(w["w_najiama05"])},
               "threshold_positive_ge": P_POS, "threshold_negative_le": P_NEG, "pseudo_weight_each": PSEUDO_WEIGHT,
               "test_rows": len(test), "selected_total": int((pos | neg).sum()), "selected_fraction": float((pos | neg).mean()),
               "positive_count": int(pos.sum()), "negative_count": int(neg.sum()),
               "positive_fraction_of_selected": float(pos.sum() / (pos | neg).sum()) if (pos | neg).sum() else None,
               "negative_fraction_of_selected": float(neg.sum() / (pos | neg).sum()) if (pos | neg).sum() else None,
               "weight_total": float((pos | neg).sum() * PSEUDO_WEIGHT),
               "baseline_submission_preserved": True, "model_artifacts_modified": False}
    (OUT / "test_pseudo_label_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = {"thresholds_fixed_before_cv": {"positive": ">=0.99", "negative": "<=0.01"},
              "pseudo_sample_weight": PSEUDO_WEIGHT, "outer_cv": "5-fold stratified, shuffle=True, random_state=42",
              "partition_per_outer_fold": "outer training split into labeled fit and separate 25% pseudo pool; outer validation held out",
              "teacher_and_student_models": MODELS, "model_params_source": "experiments/nested_stacking_oof.py fit_base, fixed 300 rounds, seed 42",
              "teacher": "equal mean of XGBoost, LightGBM, CatBoost probabilities fit only on labeled fit partition",
              "student": "same three models refit on labeled fit plus confident pool rows; original labels untouched; pseudo rows weight 0.1",
              "validation_labels": "not used for thresholding, pseudo-labeling, preprocessing choice, or fitting; used only for final fold AUC",
              "limitations": "Strict pseudo-labeling CV proxy uses existing XGB/LGB/CatBoost models; exact best blend includes pretrained/external components not refit in this nested protocol. Test pseudo probabilities are generated from the current best saved simple blend.",
              "test_labels_used": False, "baseline_submission_overwritten": False, "original_model_overwritten": False}
    (OUT / "experiment_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    fold_df, cv_summary, cv_seconds = run_cv()
    test_summary = make_test_artifacts()
    delta = fold_df["delta_auc"].to_numpy()
    reliable = bool((delta > 0).sum() >= 4 and delta.mean() > 0.0001)
    decision = {"reliable_cv_gain": reliable, "positive_folds": int((delta > 0).sum()), "mean_delta_auc": float(delta.mean()),
                "decision": "diagnostic_only_stop_no_gain" if not reliable else "candidate_gain_requires_separate_review",
                "submission_changed": False, "cv_runtime_seconds": cv_seconds, "test_pseudo_label_summary": test_summary}
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    print("CV SUMMARY", json.dumps(cv_summary, ensure_ascii=False), flush=True)
    print("TEST PSEUDO SUMMARY", json.dumps(test_summary, ensure_ascii=False), flush=True)
    print("DECISION", json.dumps(decision, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
