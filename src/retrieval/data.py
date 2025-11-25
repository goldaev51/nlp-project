from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_KB_PATH = Path("data/raw/kb/kb_securepass.csv")


def load_kb(path: Optional[str | Path] = None) -> pd.DataFrame:
    if path is None:
        path = DEFAULT_KB_PATH

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"KB file not found: {path}")

    df = pd.read_csv(path)

    required_cols = {"id", "section", "question", "answer"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"KB file {path} is missing columns: {missing}")

    if "tags" not in df.columns:
        df["tags"] = ""

    return df
