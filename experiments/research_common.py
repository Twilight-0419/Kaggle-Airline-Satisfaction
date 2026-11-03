from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUN = ROOT / "outputs" / "research_run_001"
TARGET = "satisfaction"
ID = "id"
DISTANCE = "Flight Distance"
CLASS = "Class"
TRAVEL = "Type of Travel"
RAW_FEATURES = [
    "Age", "Flight Distance", "Inflight wifi service",
    "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness",
    "Departure Delay in Minutes", "Arrival Delay in Minutes",
    "Gender", "Customer Type", "Type of Travel", "Class",
]
CATEGORICAL = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUMERIC = [c for c in RAW_FEATURES if c not in CATEGORICAL]
LOCK_SEED = 20261030
OUTER_SEED = 42
INNER_SEED_BASE = 2026
SMOOTH = 20
THREADS = 8

BASE_PARAMS: dict[str, dict[str, Any]] = {
    "lightgbm": {
        "objective": "binary", "metric": "auc", "n_estimators": 600,
        "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 20,
        "subsample": 0.85, "subsample_freq": 1, "colsample_bytree": 0.7,
        "reg_lambda": 2.0, "n_jobs": THREADS, "random_state": 42,
        "verbosity": -1,
    },
    "xgboost": {
        "objective": "binary:logistic", "eval_metric": "auc",
        "n_estimators": 300, "learning_rate": 0.05, "max_depth": 7,
        "min_child_weight": 10, "subsample": 0.85, "colsample_bytree": 0.7,
        "reg_lambda": 2.0, "tree_method": "hist", "device": "cpu",
        "n_jobs": THREADS, "random_state": 42,
    },
    "catboost": {
        "iterations": 300, "learning_rate": 0.03, "depth": 7,
        "l2_leaf_reg": 3.0, "loss_function": "Logloss", "eval_metric": "AUC",
        "random_seed": 42, "thread_count": THREADS, "verbose": False,
        "allow_writing_files": False,
    },
}


def ensure_run() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "predictions").mkdir(exist_ok=True)
    (RUN / "models").mkdir(exist_ok=True)
    config_path = RUN / "experiment_config.json"
    if not config_path.exists():
        config = {
            "run_name": RUN.name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "root": str(ROOT),
            "python_executable": sys.executable,
            "target": TARGET,
            "id_column": ID,
            "train_file_sha256": sha256(DATA / "train.csv"),
            "test_file_sha256": sha256(DATA / "test.csv"),
            "lock": {"fraction": 0.10, "stratified": True, "random_state": LOCK_SEED},
            "development_cv": {"type": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": OUTER_SEED},
            "target_encoding": {
                "keys": [[DISTANCE, CLASS], [DISTANCE, TRAVEL]],
                "smoothing": SMOOTH,
                "formula": "(group_target_sum + smoothing * fold_prior) / (group_count + smoothing)",
                "train_encoding": "inner five-fold OOF within each outer-training partition",
                "validation_encoding": "statistics from outer-training labels only",
                "locked_encoding": "statistics from complete development labels only",
            },
            "missing_value_policy": "Arrival Delay in Minutes mean imputed from the fitting partition only",
            "early_stopping": "disabled for all development OOF and locked fits",
            "test_policy": "schema/row audit only; no test predictions or submissions generated",
            "base_parameters": BASE_PARAMS,
            "overwrite_policy": "new files only under outputs/research_run_001; historical files are never modified",
            "known_validation_limitation": "The new lock is untouched by this run's selection, but historical project experiments previously used the full training set; it is not guaranteed to be globally unseen across prior work.",
        }
        write_json_atomic(config_path, config)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def log(message: str) -> None:
    ensure_run()
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}"
    print(line, flush=True)
    with (RUN / "run.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def record_experiment(row: dict[str, Any]) -> None:
    ensure_run()
    path = RUN / "experiment_records.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    key = (row.get("experiment_id"), row.get("fold"), row.get("variant"))
    rows = [r for r in rows if (r.get("experiment_id"), r.get("fold"), r.get("variant")) != key]
    rows.append(row)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in rows) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(RUN / "experiments.csv", index=False)


def load_train() -> pd.DataFrame:
    df = pd.read_csv(DATA / "train.csv")
    if df[ID].duplicated().any() or df[TARGET].isna().any():
        raise ValueError("Invalid training IDs or labels")
    return df


def prepare_splits() -> dict[str, np.ndarray]:
    ensure_run()
    split_file = RUN / "split_indices.npz"
    if split_file.exists():
        z = np.load(split_file)
        return {k: z[k] for k in z.files}
    df = load_train()
    y = df[TARGET].astype(np.int8).to_numpy()
    all_idx = np.arange(len(df))
    dev_idx, lock_idx = train_test_split(
        all_idx, test_size=0.10, stratify=y, random_state=LOCK_SEED
    )
    dev_idx = np.sort(dev_idx)
    lock_idx = np.sort(lock_idx)
    arrays: dict[str, np.ndarray] = {
        "dev_idx": dev_idx.astype(np.int32),
        "locked_idx": lock_idx.astype(np.int32),
    }
    y_dev = y[dev_idx]
    skf = StratifiedKFold(5, shuffle=True, random_state=OUTER_SEED)
    membership = np.full(len(df), -1, dtype=np.int8)
    membership[dev_idx] = 0
    membership[lock_idx] = 1
    fold_no = np.full(len(df), -1, dtype=np.int8)
    for fold, (tr_local, va_local) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=OUTER_SEED).split(np.zeros(len(y_dev)), y_dev),
        start=1,
    ):
        arrays[f"fold_{fold}_train_idx"] = dev_idx[tr_local].astype(np.int32)
        arrays[f"fold_{fold}_valid_idx"] = dev_idx[va_local].astype(np.int32)
        fold_no[dev_idx[va_local]] = fold
    np.savez_compressed(split_file, **arrays)
    pd.DataFrame({
        ID: df[ID].to_numpy(),
        "row_index": all_idx,
        "role": np.where(membership == 1, "locked", "development"),
        "development_outer_fold": fold_no,
    }).to_csv(RUN / "split_membership.csv", index=False)
    write_json_atomic(RUN / "split_manifest.json", {
        "train_rows": len(df),
        "development_rows": len(dev_idx),
        "locked_rows": len(lock_idx),
        "development_positive_rate": float(y_dev.mean()),
        "locked_positive_rate": float(y[lock_idx].mean()),
        "lock_seed": LOCK_SEED,
        "outer_cv_seed": OUTER_SEED,
        "fold_valid_rows": [int(len(arrays[f"fold_{f}_valid_idx"])) for f in range(1, 6)],
        "locked_label_policy": "not read by any selection stage; only stage lock evaluates it after candidates are frozen",
        "historical_caveat": "This run-level lock was not isolated from prior project experiments that may have used the full training set.",
    })
    log(f"prepared splits dev={len(dev_idx)} locked={len(lock_idx)}")
    return arrays


def load_split_arrays() -> dict[str, np.ndarray]:
    z = np.load(RUN / "split_indices.npz")
    return {k: z[k] for k in z.files}


def audit_project() -> None:
    ensure_run()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    vm = ctypes_memory()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True) if shutil.which("nvidia-smi") else None
    audit = {
        "project_root": str(ROOT),
        "python": {"executable": sys.executable, "version": sys.version},
        "platform": platform.platform(),
        "cpu_logical": os.cpu_count(),
        "memory": vm,
        "disk_free_gb": round(shutil.disk_usage(ROOT).free / 1e9, 2),
        "gpu": (gpu.stdout.strip() if gpu and gpu.returncode == 0 else "unavailable"),
        "library_versions": {
            "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
            "lightgbm": lgb.__version__, "xgboost": xgb.__version__, "catboost": cb.__version__,
        },
        "train": {
            "path": str(DATA / "train.csv"), "rows": len(train), "columns": list(train.columns),
            "id_column": ID, "target_column": TARGET,
            "id_unique": bool(train[ID].is_unique), "target_dtype": str(train[TARGET].dtype),
            "target_counts": {str(k): int(v) for k, v in train[TARGET].value_counts(dropna=False).items()},
            "missing": {str(k): int(v) for k, v in train.isna().sum().items() if v},
        },
        "test": {
            "path": str(DATA / "test.csv"), "rows": len(test), "columns": list(test.columns),
            "id_column": ID, "id_unique": bool(test[ID].is_unique),
            "target_present": TARGET in test.columns,
            "missing": {str(k): int(v) for k, v in test.isna().sum().items() if v},
        },
        "b20_source": {
            "path": str(ROOT / "experiments" / "strict_lightgbm_b20_cv.py"),
            "sha256": sha256(ROOT / "experiments" / "strict_lightgbm_b20_cv.py"),
            "keys": [[DISTANCE, CLASS], [DISTANCE, TRAVEL]],
            "smoothing": SMOOTH,
            "formula": "(group_target_sum + smoothing * fold_prior) / (group_count + smoothing)",
            "inner_oof": "StratifiedKFold(5, shuffle=True, random_state=2026 + outer_fold_number)",
            "missing_key_policy": "keys have no missing values in audited training data",
        },
    }
    write_json_atomic(RUN / "environment_audit.json", audit)
    log("project and environment audit complete")


def ctypes_memory() -> dict[str, float]:
    import ctypes
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return {"total_gb": round(stat.ullTotalPhys / 1e9, 2), "available_gb": round(stat.ullAvailPhys / 1e9, 2)}


class Preprocessor:
    def __init__(self, frame: pd.DataFrame):
        self.means = {c: float(frame[c].mean()) for c in NUMERIC}
        self.categories = {
            c: sorted(frame[c].astype(str).dropna().unique().tolist()) for c in CATEGORICAL
        }
        self.feature_names = list(NUMERIC)
        for c in CATEGORICAL:
            self.feature_names.extend([f"{c}={v}" for v in self.categories[c]])

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(frame), len(self.feature_names)), dtype=np.float32)
        for j, c in enumerate(NUMERIC):
            out[:, j] = frame[c].fillna(self.means[c]).to_numpy(dtype=np.float32)
        offset = len(NUMERIC)
        values = frame.copy()
        for c in CATEGORICAL:
            vals = values[c].astype(str).to_numpy()
            for level in self.categories[c]:
                out[:, offset] = (vals == level).astype(np.float32)
                offset += 1
        return out


def encode_pair(
    train_frame: pd.DataFrame,
    y: np.ndarray,
    eval_frame: pd.DataFrame,
    key_cols: list[str],
    smooth: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    keys = pd.concat([train_frame[key_cols], eval_frame[key_cols]], ignore_index=True)
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(keys), sort=False)
    train_codes = codes[:len(train_frame)]
    eval_codes = codes[len(train_frame):]
    n_groups = int(codes.max()) + 1
    train_result = np.empty(len(train_frame), dtype=np.float32)
    inner = StratifiedKFold(5, shuffle=True, random_state=seed)
    for fit_idx, hold_idx in inner.split(np.zeros(len(y)), y):
        prior = float(y[fit_idx].mean())
        sums = np.bincount(train_codes[fit_idx], weights=y[fit_idx], minlength=n_groups)
        counts = np.bincount(train_codes[fit_idx], minlength=n_groups)
        hc = train_codes[hold_idx]
        train_result[hold_idx] = (sums[hc] + smooth * prior) / (counts[hc] + smooth)
    prior = float(y.mean())
    sums = np.bincount(train_codes, weights=y, minlength=n_groups)
    counts = np.bincount(train_codes, minlength=n_groups)
    eval_result = (sums[eval_codes] + smooth * prior) / (counts[eval_codes] + smooth)
    return train_result, eval_result.astype(np.float32)


def build_features(
    train_frame: pd.DataFrame,
    y: np.ndarray,
    eval_frame: pd.DataFrame,
    variant: str,
    inner_seed: int,
) -> tuple[np.ndarray, np.ndarray, Preprocessor]:
    prep = Preprocessor(train_frame)
    xtr = prep.transform(train_frame)
    xev = prep.transform(eval_frame)
    if variant not in {"raw", "b20", "b20_distance_class", "b20_distance_travel"}:
        raise ValueError(variant)
    if variant != "raw":
        pairs = []
        if variant in {"b20", "b20_distance_class"}:
            pairs.append([DISTANCE, CLASS])
        if variant in {"b20", "b20_distance_travel"}:
            pairs.append([DISTANCE, TRAVEL])
        enc_tr, enc_ev = [], []
        for key_cols in pairs:
            a, b = encode_pair(train_frame, y, eval_frame, key_cols, SMOOTH, inner_seed)
            enc_tr.append(a)
            enc_ev.append(b)
        xtr = np.column_stack([xtr] + enc_tr).astype(np.float32, copy=False)
        xev = np.column_stack([xev] + enc_ev).astype(np.float32, copy=False)
    return xtr, xev, prep


def make_model(model_name: str, params: dict[str, Any], seed: int):
    p = dict(params)
    if "random_state" in p:
        p["random_state"] = seed
    if "random_seed" in p:
        p["random_seed"] = seed
    if model_name == "lightgbm":
        return lgb.LGBMClassifier(**p)
    if model_name == "xgboost":
        return xgb.XGBClassifier(**p)
    if model_name == "catboost":
        return cb.CatBoostClassifier(**p)
    raise ValueError(model_name)


def fit_predict(model_name: str, params: dict[str, Any], xtr: np.ndarray, ytr: np.ndarray, xev: np.ndarray, seed: int) -> np.ndarray:
    model = make_model(model_name, params, seed)
    model.fit(xtr, ytr)
    return model.predict_proba(xev)[:, 1].astype(np.float32)


def run_cv_experiment(
    experiment_id: str,
    model_name: str,
    params: dict[str, Any],
    feature_variant: str,
    cohort_idx: np.ndarray,
    fold_indices: list[tuple[np.ndarray, np.ndarray]],
    stage: str,
    save_models: bool = False,
) -> dict[str, Any]:
    ensure_run()
    df = load_train()
    y_all = df[TARGET].astype(np.int8).to_numpy()
    cohort = df.iloc[cohort_idx].reset_index(drop=True)
    y = y_all[cohort_idx]
    out_dir = RUN / stage / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    oof = np.full(len(cohort), np.nan, dtype=np.float32)
    fold_rows = []
    started_all = time.perf_counter()
    for fold, (tr_local, va_local) in enumerate(fold_indices, start=1):
        pred_path = out_dir / f"fold_{fold}_prediction.npy"
        meta_path = out_dir / f"fold_{fold}_meta.json"
        if pred_path.exists() and meta_path.exists():
            pred = np.load(pred_path)
            oof[va_local] = pred
            fold_rows.append(json.loads(meta_path.read_text(encoding="utf-8")))
            log(f"resume skip {experiment_id} fold={fold}")
            continue
        tr_frame = cohort.iloc[tr_local]
        va_frame = cohort.iloc[va_local]
        ytr = y[tr_local]
        started = time.perf_counter()
        xtr, xva, prep = build_features(tr_frame, ytr, va_frame, feature_variant, INNER_SEED_BASE + fold)
        built = time.perf_counter()
        pred = fit_predict(model_name, params, xtr, ytr, xva, seed=42 + fold)
        finished = time.perf_counter()
        auc = float(roc_auc_score(y[va_local], pred))
        row = {
            "experiment_id": experiment_id, "stage": stage, "model": model_name,
            "variant": feature_variant, "fold": fold, "auc": auc,
            "train_rows": int(len(tr_local)), "valid_rows": int(len(va_local)),
            "n_features": int(xtr.shape[1]), "feature_build_seconds": built - started,
            "fit_predict_seconds": finished - built, "fold_wall_seconds": finished - started,
            "parameters": params,
        }
        np.save(pred_path, pred)
        write_json_atomic(meta_path, row)
        record_experiment(row)
        fold_rows.append(row)
        oof[va_local] = pred
        log(f"{experiment_id} fold={fold} auc={auc:.9f} features={xtr.shape[1]} seconds={finished-started:.1f}")
        del xtr, xva, prep
    valid = np.isfinite(oof)
    if not valid.all():
        raise RuntimeError(f"Incomplete OOF for {experiment_id}: {np.isnan(oof).sum()} missing")
    fold_labels = np.full(len(cohort), -1, dtype=np.int8)
    for fold, (_, va_local) in enumerate(fold_indices, start=1):
        fold_labels[va_local] = fold
    pred_df = pd.DataFrame({
        ID: cohort[ID].to_numpy(), "row_index": cohort_idx,
        TARGET: y, "fold": fold_labels, "prediction": oof,
    })
    pred_df.to_csv(out_dir / "oof_predictions.csv", index=False)
    fold_aucs = [r["auc"] for r in fold_rows]
    summary = {
        "experiment_id": experiment_id, "stage": stage, "model": model_name,
        "variant": feature_variant, "cohort_rows": int(len(cohort_idx)),
        "n_features": int(fold_rows[0]["n_features"]), "fold_aucs": fold_aucs,
        "mean_auc": float(np.mean(fold_aucs)), "std_population": float(np.std(fold_aucs, ddof=0)),
        "std_sample": float(np.std(fold_aucs, ddof=1)), "oof_auc": float(roc_auc_score(y, oof)),
        "feature_build_seconds": float(sum(r["feature_build_seconds"] for r in fold_rows)),
        "fit_predict_seconds": float(sum(r["fit_predict_seconds"] for r in fold_rows)),
        "wall_seconds": float(time.perf_counter() - started_all), "parameters": params,
    }
    write_json_atomic(out_dir / "summary.json", summary)
    log(f"complete {experiment_id} mean_auc={summary['mean_auc']:.9f} oof_auc={summary['oof_auc']:.9f}")
    return summary
