from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def slugify_dataset_name(dataset: str) -> str:
    return dataset.replace("/", "__")


def stable_hash(items: list[str]) -> str:
    h = hashlib.md5()
    for item in items:
        h.update(item.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:12]


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)


def read_pickle(path: Path) -> pd.DataFrame:
    return pd.read_pickle(path)


def write_pickle(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
