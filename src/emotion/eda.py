from pathlib import Path

from .data import prepare_goemotions_mood_dataset
from .labels import ID_TO_MOOD


def run_eda(cache_dir: str | None = None,
            output_dir: str = "data/processed/emotion") -> None:
    train_df, val_df, test_df = prepare_goemotions_mood_dataset(cache_dir=cache_dir)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=== Class distribution: TRAIN ===")
    print(train_df["mood"].value_counts())
    print("\n=== Class distribution (proportions): TRAIN ===")
    print(train_df["mood"].value_counts(normalize=True))

    print("\n=== Class distribution: VALIDATION ===")
    print(val_df["mood"].value_counts())
    print("\n=== Class distribution (proportions): VALIDATION ===")
    print(val_df["mood"].value_counts(normalize=True))

    print("\n=== Class distribution: TEST ===")
    print(test_df["mood"].value_counts())
    print("\n=== Class distribution (proportions): TEST ===")
    print(test_df["mood"].value_counts(normalize=True))

    print("\n=== Sample examples per mood (TRAIN) ===")
    for mood_id, mood_name in ID_TO_MOOD.items():
        print(f"\n--- Mood: {mood_name} ---")
        subset = train_df[train_df["mood"] == mood_name].head(5)
        for i, row in subset.iterrows():
            print(f"[{i}] {row['text']!r}")

    train_csv = out_path / "goemotions_mood_train.csv"
    val_csv = out_path / "goemotions_mood_val.csv"
    test_csv = out_path / "goemotions_mood_test.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"\nSaved train to: {train_csv}")
    print(f"Saved val   to: {val_csv}")
    print(f"Saved test  to: {test_csv}")


if __name__ == "__main__":
    run_eda()
