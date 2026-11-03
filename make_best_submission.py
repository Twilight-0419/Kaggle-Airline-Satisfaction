from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
PUBLIC_STACK_DIR = PROJECT_DIR / "outputs" / "public_stack"
EXTRA_OOF_DIR = PROJECT_DIR / "某个公开算法" / "extra_oof"
TARGET = "satisfaction"
ID = "id"


def main() -> None:
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    ext_l2_test = np.load(PUBLIC_STACK_DIR / "ext_l2_test.npy")
    najiama = pd.read_csv(
        EXTRA_OOF_DIR / "s6e10-oof" / "05_submission.csv"
    ).set_index(ID)[TARGET].reindex(sample[ID]).to_numpy()

    if ext_l2_test.shape != (len(sample),):
        raise ValueError(
            f"Unexpected ext_l2_test shape: {ext_l2_test.shape}"
        )
    if np.isnan(najiama).any():
        raise ValueError("Could not align najiama_05 predictions.")

    # Validated OOF AUC: 0.962270465.
    blended = expit(
        0.87 * logit(np.clip(ext_l2_test, 1e-7, 1 - 1e-7))
        + 0.13 * logit(np.clip(najiama, 1e-7, 1 - 1e-7))
    )
    submission = pd.DataFrame(
        {
            ID: sample[ID].to_numpy(),
            TARGET: np.clip(blended, 0.0, 1.0),
        }
    )

    assert submission.columns.tolist() == sample.columns.tolist()
    assert len(submission) == len(sample)
    assert np.array_equal(
        submission[ID].to_numpy(), sample[ID].to_numpy()
    )
    assert submission[TARGET].between(0.0, 1.0).all()

    output_path = DATA_DIR / "submission.csv"
    submission.to_csv(output_path, index=False)
    print("Saved:", output_path)
    print("Rows:", len(submission))
    print(
        "Probability range:",
        float(submission[TARGET].min()),
        float(submission[TARGET].max()),
    )


if __name__ == "__main__":
    main()
