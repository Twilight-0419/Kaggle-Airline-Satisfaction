from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import research_common as rc


PROGRESS = rc.RUN / "progress.json"


def update_progress(stage: str, status: str, **details) -> None:
    rc.ensure_run()
    current = {}
    if PROGRESS.exists():
        current = json.loads(PROGRESS.read_text(encoding="utf-8"))
    current.setdefault("stages", {})[stage] = {
        "status": status,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        **details,
    }
    rc.write_json_atomic(PROGRESS, current)


def get_folds():
    arrays = rc.load_split_arrays()
    dev_idx = arrays["dev_idx"]
    global_to_local = np.full(int(dev_idx.max()) + 1, -1, dtype=np.int64)
    global_to_local[dev_idx] = np.arange(len(dev_idx), dtype=np.int64)
    return [
        (
            global_to_local[arrays[f"fold_{fold}_train_idx"]],
            global_to_local[arrays[f"fold_{fold}_valid_idx"]],
        )
        for fold in range(1, 6)
    ]


def stage_audit() -> None:
    update_progress("stage1_audit", "running")
    rc.audit_project()
    update_progress(
        "stage1_audit",
        "complete",
        audit_file=str(rc.RUN / "environment_audit.json"),
    )


def stage_prepare() -> None:
    update_progress("stage2_splits", "running")
    arrays = rc.prepare_splits()
    update_progress(
        "stage2_splits",
        "complete",
        development_rows=int(len(arrays["dev_idx"])),
        locked_rows=int(len(arrays["locked_idx"])),
        split_file=str(rc.RUN / "split_indices.npz"),
    )


def stage4(models: list[str]) -> None:
    arrays = rc.load_split_arrays()
    dev_idx = arrays["dev_idx"]
    folds = get_folds()
    model_specs = {
        "lightgbm": ("B1_lgb_raw", "B2_lgb_b20"),
        "xgboost": ("B3_xgb_raw", "B4_xgb_b20"),
        "catboost": ("B5_cat_raw", "B6_cat_b20"),
    }
    all_summaries = []
    for model_name in models:
        raw_id, b20_id = model_specs[model_name]
        for experiment_id, variant in (
            (raw_id, "raw"),
            (b20_id, "b20"),
        ):
            update_progress(
                "stage4_full_dev_ab",
                "running",
                current_experiment=experiment_id,
            )
            summary = rc.run_cv_experiment(
                experiment_id=experiment_id,
                model_name=model_name,
                params=rc.BASE_PARAMS[model_name],
                feature_variant=variant,
                cohort_idx=dev_idx,
                fold_indices=folds,
                stage="stage4_full_dev_ab",
                save_models=False,
            )
            all_summaries.append(summary)

    summary_path = rc.RUN / "stage4_full_dev_ab" / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if summary_path.exists():
        existing = pd.read_csv(summary_path).to_dict("records")
    frame = pd.DataFrame(existing + all_summaries)
    frame = frame.drop_duplicates(subset=["experiment_id"], keep="last")
    frame.to_csv(summary_path, index=False)
    update_progress(
        "stage4_full_dev_ab",
        "complete",
        models=models,
        summary_file=str(summary_path),
    )


def stage_status() -> None:
    rc.ensure_run()
    print("Run:", rc.RUN)
    if PROGRESS.exists():
        print(PROGRESS.read_text(encoding="utf-8"))
    else:
        print("No progress.json yet")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    sub.add_parser("prepare")
    stage4_parser = sub.add_parser("stage4")
    stage4_parser.add_argument(
        "--models",
        nargs="+",
        choices=["lightgbm", "xgboost", "catboost"],
        default=["lightgbm"],
    )
    sub.add_parser("status")
    args = parser.parse_args()

    rc.ensure_run()
    if args.command == "audit":
        stage_audit()
    elif args.command == "prepare":
        stage_prepare()
    elif args.command == "stage4":
        stage4(args.models)
    else:
        stage_status()


if __name__ == "__main__":
    main()
