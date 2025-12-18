from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, sep="\t", index=False)


def write_series_tsv(s: pd.Series, path: Path, *, name: str = "value") -> None:
    ensure_dir(path.parent)
    s.to_frame(name=name).to_csv(path, sep="\t", index=True, header=True)

