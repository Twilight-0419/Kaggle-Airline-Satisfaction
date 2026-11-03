from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def main() -> None:
    sample = pd.read_csv(DATA / "sample_submission.csv")
    public_root = next(ROOT.glob("某个公开算法*")) / "extra_oof" / "ps-s6e10-02"
    source = public_root / "0.96213d.txt"
    candidate = pd.read_csv(source).set_index("id").reindex(sample["id"])
    if candidate["satisfaction"].isna().any():
        raise ValueError("Could not align source predictions with sample IDs")

    submission = pd.DataFrame(
        {
            "id": sample["id"].to_numpy(),
            "satisfaction": candidate["satisfaction"].to_numpy(),
        }
    )
    if not submission["satisfaction"].between(0.0, 1.0).all():
        raise ValueError("Predictions must be probabilities in [0, 1]")
    submission.to_csv(DATA / "submission.csv", index=False)
    print(f"Saved {DATA / 'submission.csv'} ({len(submission):,} rows)")


if __name__ == "__main__":
    main()
