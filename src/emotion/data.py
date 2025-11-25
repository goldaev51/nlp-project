from typing import Tuple

import pandas as pd
from datasets import load_dataset, DatasetDict

from .labels import choose_mood_for_emotions, MOOD_TO_ID


def load_raw_goemotions_simplified(cache_dir: str | None = None) -> DatasetDict:
    ds = load_dataset("go_emotions", "simplified", cache_dir=cache_dir)
    return ds


def prepare_goemotions_mood_dataset(
    cache_dir: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ds = load_raw_goemotions_simplified(cache_dir=cache_dir)

    label_names = ds["train"].features["labels"].feature.names

    def map_example(example):
        emotions = [label_names[i] for i in example["labels"]]
        mood = choose_mood_for_emotions(emotions)
        if mood is None:
            return {"use_sample": False, "mood": None, "mood_id": -1}
        return {
            "use_sample": True,
            "mood": mood,
            "mood_id": MOOD_TO_ID[mood],
        }

    mapped = {}
    for split in ["train", "validation", "test"]:
        with_mood = ds[split].map(map_example)
        with_mood = with_mood.filter(lambda ex: ex["use_sample"])
        df = with_mood.to_pandas()[["text", "mood", "mood_id"]]
        mapped[split] = df

    return mapped["train"], mapped["validation"], mapped["test"]
