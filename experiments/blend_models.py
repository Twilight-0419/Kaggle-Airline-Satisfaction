from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
ENSEMBLE_DIR = PROJECT_DIR / "outputs" / "ensemble"
TARGET = "satisfaction"
ID_COL = "id"
MODELS = ["xgboost", "lightgbm", "catboost"]


def rank_normalize(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / len(values)


def search_weights(
    oof_matrix: np.ndarray,
    y: np.ndarray,
    transform: str,
) -> tuple[tuple[float, ...], float]:
    if transform == "probability":
        transformed = oof_matrix
        inverse = lambda x: x
    elif transform == "logit":
        transformed = logit(np.clip(oof_matrix, 1e-7, 1 - 1e-7))
        inverse = expit
    elif transform == "rank":
        transformed = np.column_stack(
            [rank_normalize(oof_matrix[:, i]) for i in range(oof_matrix.shape[1])]
        )
        inverse = lambda x: x
    else:
        raise ValueError(transform)

    best_weights = (1.0, 0.0, 0.0)
    best_auc = -1.0
    for w0 in np.arange(0.0, 1.0001, 0.05):
        for w1 in np.arange(0.0, 1.0001 - w0, 0.05):
            w2 = 1.0 - w0 - w1
            weights = np.array([w0, w1, w2])
            prediction = inverse(transformed @ weights)
            auc = roc_auc_score(y, prediction)
            if auc > best_auc:
                best_auc = float(auc)
                best_weights = tuple(float(w) for w in weights)
    return best_weights, best_auc


def apply_blend(
    matrix: np.ndarray,
    weights: tuple[float, ...],
    transform: str,
) -> np.ndarray:
    weights_array = np.asarray(weights)
    if transform == "probability":
        return matrix @ weights_array
    if transform == "logit":
        return expit(
            logit(np.clip(matrix, 1e-7, 1 - 1e-7)) @ weights_array
        )
    if transform == "rank":
        ranked = np.column_stack(
            [rank_normalize(matrix[:, i]) for i in range(matrix.shape[1])]
        )
        return ranked @ weights_array
    raise ValueError(transform)


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv", usecols=[ID_COL, TARGET])
    test_ids = pd.read_csv(DATA_DIR / "sample_submission.csv", usecols=[ID_COL])

    oof = train[[ID_COL, TARGET]].copy()
    test = test_ids.copy()
    for model in MODELS:
        model_oof = pd.read_csv(ENSEMBLE_DIR / f"{model}_oof.csv")
        model_test = pd.read_csv(ENSEMBLE_DIR / f"{model}_test.csv")
        oof = oof.merge(
            model_oof[[ID_COL, TARGET]].rename(
                columns={TARGET: f"{model}_oof"}
            ),
            on=ID_COL,
            how="inner",
        )
        test = test.merge(
            model_test[[ID_COL, TARGET]].rename(
                columns={TARGET: f"{model}_test"}
            ),
            on=ID_COL,
            how="inner",
        )

    y = oof[TARGET].to_numpy()
    oof_matrix = oof[[f"{model}_oof" for model in MODELS]].to_numpy()
    test_matrix = test[[f"{model}_test" for model in MODELS]].to_numpy()

    results: list[dict[str, object]] = []
    for model_index, model in enumerate(MODELS):
        results.append(
            {
                "name": model,
                "transform": "single",
                "w_xgb": float(model_index == 0),
                "w_lgb": float(model_index == 1),
                "w_cat": float(model_index == 2),
                "oof_auc": roc_auc_score(y, oof_matrix[:, model_index]),
            }
        )

    best_overall = None
    for transform in ["probability", "logit", "rank"]:
        weights, auc = search_weights(oof_matrix, y, transform)
        row = {
            "name": f"{transform}_blend",
            "transform": transform,
            "w_xgb": weights[0],
            "w_lgb": weights[1],
            "w_cat": weights[2],
            "oof_auc": auc,
        }
        results.append(row)
        if best_overall is None or auc > best_overall["oof_auc"]:
            best_overall = row

    result_frame = pd.DataFrame(results).sort_values(
        "oof_auc", ascending=False
    )
    result_frame.to_csv(ENSEMBLE_DIR / "blend_results.csv", index=False)

    transform = str(best_overall["transform"])
    weights = (
        float(best_overall["w_xgb"]),
        float(best_overall["w_lgb"]),
        float(best_overall["w_cat"]),
    )
    oof_blend = apply_blend(oof_matrix, weights, transform)
    test_blend = apply_blend(test_matrix, weights, transform)

    pd.DataFrame(
        {ID_COL: oof[ID_COL], TARGET: oof_blend}
    ).to_csv(ENSEMBLE_DIR / "blend_oof.csv", index=False)
    pd.DataFrame(
        {ID_COL: test[ID_COL], TARGET: test_blend}
    ).to_csv(ENSEMBLE_DIR / "blend_test.csv", index=False)

    print(result_frame.to_string(index=False), flush=True)
    print(f"best={transform} weights={weights}", flush=True)


if __name__ == "__main__":
    main()
