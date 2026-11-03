from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "train.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

TARGET = "satisfaction"
ID_COL = "id"

RATING_COLS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
]

BASE_CATEGORICAL = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-rows", type=int, default=280000)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Average Ranking"] = out[RATING_COLS].mean(axis=1)
    out["Class_Travel"] = (
        out["Class"].astype(str) + "_" + out["Type of Travel"].astype(str)
    )
    out["Online_x_Wifi"] = (
        out["Online boarding"] * out["Inflight wifi service"]
    )
    out["Online_x_Entertainment"] = (
        out["Online boarding"] * out["Inflight entertainment"]
    )
    out["Total Delay"] = (
        out["Departure Delay in Minutes"] + out["Arrival Delay in Minutes"]
    )
    out["Delay Difference"] = (
        out["Arrival Delay in Minutes"] - out["Departure Delay in Minutes"]
    ).abs()
    out["Delayed 15"] = (out["Total Delay"] >= 15).astype("int8")
    out["Rating Min"] = out[RATING_COLS].min(axis=1)
    out["Rating Max"] = out[RATING_COLS].max(axis=1)
    out["Rating Std"] = out[RATING_COLS].std(axis=1)
    out["Low Ratings"] = (out[RATING_COLS] <= 2).sum(axis=1)
    out["High Ratings"] = (out[RATING_COLS] >= 4).sum(axis=1)
    return out


def feature_configs() -> dict[str, tuple[list[str], list[str]]]:
    raw = [
        "Age",
        "Flight Distance",
        "Inflight wifi service",
        "Departure/Arrival time convenient",
        "Ease of Online booking",
        "Gate location",
        "Food and drink",
        "Online boarding",
        "Seat comfort",
        "Inflight entertainment",
        "On-board service",
        "Leg room service",
        "Baggage handling",
        "Checkin service",
        "Cleanliness",
        "Departure Delay in Minutes",
        "Arrival Delay in Minutes",
        "Gender",
        "Customer Type",
        "Type of Travel",
        "Class",
    ]
    cats = list(BASE_CATEGORICAL)
    avg = ["Average Ranking"]
    combo = ["Class_Travel"]
    wifi = ["Online_x_Wifi"]
    entertainment = ["Online_x_Entertainment"]
    delay = ["Total Delay", "Delay Difference", "Delayed 15"]
    ratings = [
        "Rating Min",
        "Rating Max",
        "Rating Std",
        "Low Ratings",
        "High Ratings",
    ]

    return {
        "raw": (raw, cats),
        "raw_avg": (raw + avg, cats),
        "avg_combo": (raw + avg + combo, cats + combo),
        "avg_wifi": (raw + avg + wifi, cats),
        "avg_entertainment": (raw + avg + entertainment, cats),
        "avg_delay": (raw + avg + delay, cats),
        "avg_rating_stats": (raw + avg + ratings, cats),
        "avg_combo_wifi": (raw + avg + combo + wifi, cats + combo),
        "avg_combo_entertainment": (
            raw + avg + combo + entertainment,
            cats + combo,
        ),
        "current": (
            raw + avg + combo + wifi + entertainment + delay,
            cats + combo,
        ),
        "current_no_combo": (
            raw + avg + wifi + entertainment + delay,
            cats,
        ),
        "current_no_delay": (
            raw + avg + combo + wifi + entertainment,
            cats + combo,
        ),
    }


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_PATH)
    train["Arrival Delay in Minutes"] = train[
        "Arrival Delay in Minutes"
    ].fillna(train["Arrival Delay in Minutes"].mean())

    if args.sample_rows and args.sample_rows < len(train):
        train, _ = train_test_split(
            train,
            train_size=args.sample_rows,
            stratify=train[TARGET],
            random_state=args.seed,
        )
    train = train.reset_index(drop=True)

    y = train[TARGET].astype("int8")
    data = add_engineered_features(train.drop(columns=[ID_COL, TARGET]))

    train_idx, valid_idx = train_test_split(
        np.arange(len(data)),
        test_size=0.2,
        stratify=y,
        random_state=args.seed,
    )

    results: list[dict[str, object]] = []
    results_path = OUTPUT_DIR / "feature_screen_results.csv"

    print(
        f"rows={len(data):,} train={len(train_idx):,} valid={len(valid_idx):,}",
        flush=True,
    )

    for name, (features, categorical) in feature_configs().items():
        started = time.perf_counter()

        x_train = data.iloc[train_idx][features]
        x_valid = data.iloc[valid_idx][features]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        x_train = pd.get_dummies(
            x_train, columns=categorical, dtype=np.int8
        )
        x_valid = pd.get_dummies(
            x_valid, columns=categorical, dtype=np.int8
        ).reindex(columns=x_train.columns, fill_value=0)

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=args.rounds,
            learning_rate=0.05,
            max_depth=7,
            min_child_weight=10,
            subsample=0.85,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            tree_method="hist",
            device="cpu",
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )
        model.fit(x_train, y_train, verbose=False)

        valid_pred = model.predict_proba(x_valid)[:, 1]
        train_pred = model.predict_proba(x_train)[:, 1]
        valid_auc = roc_auc_score(y_valid, valid_pred)
        train_auc = roc_auc_score(y_train, train_pred)
        elapsed = time.perf_counter() - started

        row = {
            "name": name,
            "valid_auc": valid_auc,
            "train_auc": train_auc,
            "n_features": x_train.shape[1],
            "seconds": elapsed,
        }
        results.append(row)
        pd.DataFrame(results).sort_values(
            "valid_auc", ascending=False
        ).to_csv(results_path, index=False)

        print(
            f"{name:26s} valid_auc={valid_auc:.6f} "
            f"train_auc={train_auc:.6f} features={x_train.shape[1]} "
            f"time={elapsed:.1f}s",
            flush=True,
        )

        del model, x_train, x_valid, train_pred, valid_pred
        gc.collect()

    print(results_path, flush=True)


if __name__ == "__main__":
    main()
