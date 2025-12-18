from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xenaPython
from tqdm import tqdm

from tcga_tp53.cache import (
    read_parquet,
    read_pickle,
    slugify_dataset_name,
    stable_hash,
    write_parquet,
    write_pickle,
)
from tcga_tp53.config import XenaDatasets


def _drop_sampleid_field(fields: list[str]) -> list[str]:
    return [f for f in fields if f != "sampleID"]


def _as_float32(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype("float32")


def _as_int8(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype("int8")


def _chunks(seq: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


@dataclass
class XenaClient:
    hub: str = XenaDatasets.HUB
    cache_dir: Path = Path("cache")
    overwrite_cache: bool = False
    show_progress: bool = True

    def dataset_samples(self, dataset: str, limit: int = 1_000_000) -> list[str]:
        return xenaPython.dataset_samples(self.hub, dataset, limit)

    def dataset_fields(self, dataset: str) -> list[str]:
        return xenaPython.dataset_field(self.hub, dataset)

    def fetch_clinical(self, dataset: str, *, cache_key: str = "all") -> pd.DataFrame:
        logger = logging.getLogger(__name__)
        slug = slugify_dataset_name(dataset)
        cache_path = self.cache_dir / slug / f"{cache_key}.pkl"
        if cache_path.exists() and not self.overwrite_cache:
            df = read_pickle(cache_path)
            logger.debug("cache hit: clinical %s key=%s shape=%s", dataset, cache_key, df.shape)
            return df

        samples = self.dataset_samples(dataset)
        fields = self.dataset_fields(dataset)
        values = xenaPython.dataset_fetch(self.hub, dataset, samples, fields)
        df = pd.DataFrame(values, index=fields, columns=samples).T
        # clinical matrices often contain mixed dtypes (e.g., 'NaN' strings), which can break parquet writes.
        write_pickle(df, cache_path)
        logger.info("loaded clinical %s: %d samples x %d fields", dataset, df.shape[0], df.shape[1])
        return df

    def decode_field_codes(self, dataset: str, df: pd.DataFrame, *, fields: list[str]) -> pd.DataFrame:
        """
        Decode Xena clinicalMatrix categorical fields that are stored as integer codes (0,1,2,...).
        """
        out = df.copy()
        code_entries = xenaPython.field_codes(self.hub, dataset, fields)
        code_map: dict[str, list[str]] = {}
        for entry in code_entries:
            name = entry.get("name")
            code = entry.get("code", "")
            if not name:
                continue
            code_map[name] = str(code).split("\t")

        def decode_val(v: object, labels: list[str]) -> object:
            if v is None:
                return np.nan
            s = str(v).strip()
            if not s or s.lower() == "nan":
                return np.nan
            try:
                i = int(float(s))
            except Exception:
                return v
            if 0 <= i < len(labels):
                lab = labels[i]
                return (lab if lab else np.nan)
            return np.nan

        for field, labels in code_map.items():
            if field not in out.columns:
                continue
            out[field] = out[field].apply(lambda v: decode_val(v, labels))

        return out

    def fetch_dense_matrix(
        self,
        dataset: str,
        *,
        samples: list[str],
        probes: list[str],
        cache_key: str,
        probe_chunk_size: int = 500,
        dtype: str = "float32",
    ) -> pd.DataFrame:
        logger = logging.getLogger(__name__)
        slug = slugify_dataset_name(dataset)
        cache_path = self.cache_dir / slug / f"{cache_key}.parquet"
        if cache_path.exists() and not self.overwrite_cache:
            df = read_parquet(cache_path)
            logger.debug("cache hit: %s key=%s shape=%s", dataset, cache_key, df.shape)
            return df

        probes = _drop_sampleid_field(probes)
        parts: list[pd.DataFrame] = []
        for chunk in tqdm(
            list(_chunks(probes, probe_chunk_size)),
            desc=f"Xena fetch {slug}",
            disable=not self.show_progress,
        ):
            rows = xenaPython.dataset_fetch(self.hub, dataset, samples, chunk)
            arr = np.array(rows, dtype=dtype)
            parts.append(pd.DataFrame(arr, index=chunk, columns=samples))
        df = pd.concat(parts, axis=0)

        if dtype.startswith("float"):
            df = _as_float32(df)
        elif dtype.startswith("int"):
            df = _as_int8(df)

        write_parquet(df, cache_path)
        logger.info("loaded %s: %d probes x %d samples", dataset, df.shape[0], df.shape[1])
        return df

    def fetch_tp53_sparse_mutations(self, *, samples: list[str]) -> pd.DataFrame:
        dataset = XenaDatasets.MUT_VEC
        slug = slugify_dataset_name(dataset)
        cache_key = f"TP53__n{len(samples)}__{stable_hash(sorted(samples))}"
        cache_path = self.cache_dir / slug / f"{cache_key}.parquet"
        if cache_path.exists() and not self.overwrite_cache:
            return read_parquet(cache_path)

        res = xenaPython.sparse_data(self.hub, dataset, samples, ["TP53"])
        rows = res.get("rows") or {}
        df = pd.DataFrame(rows)
        # Flatten list-valued genes (e.g. ["TP53"]) to first element for convenience
        if "genes" in df.columns:
            df["genes"] = df["genes"].apply(lambda x: x[0] if isinstance(x, list) and x else x)
        write_parquet(df, cache_path)
        return df

    def fetch_methylation_probes_for_genes(
        self,
        *,
        samples: list[str],
        genes: list[str],
        cache_key: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        dataset = XenaDatasets.METH450
        slug = slugify_dataset_name(dataset)
        cache_path = self.cache_dir / slug / f"{cache_key}.parquet"
        if cache_path.exists() and not self.overwrite_cache:
            probe_by_sample = read_parquet(cache_path)
            probe_meta = read_parquet(cache_path.with_suffix(".meta.parquet"))
            return probe_by_sample, probe_meta

        position, values = xenaPython.dataset_gene_probes_values(self.hub, dataset, samples, genes)
        probe_names = position["name"]
        probe_by_sample = pd.DataFrame(np.array(values, dtype="float32"), index=probe_names, columns=samples)

        # fetch probe -> genes mapping from probemap (server-side filter by genes to keep query compact)
        probemap = XenaDatasets.METH450_PROBEMAP
        genes_str = " ".join(json.dumps(g) for g in genes)
        q = (
            f'(xena-query {{:select ["name" "genes" "position"] '
            f':from ["{probemap}"] :where [:in :any "genes" [{genes_str}]]}})'
        )
        mapping_json = xenaPython.xenaQuery.post(self.hub, q)
        mapping = json.loads(mapping_json)
        probe_meta = pd.DataFrame(mapping)

        write_parquet(probe_by_sample, cache_path)
        write_parquet(probe_meta, cache_path.with_suffix(".meta.parquet"))
        return probe_by_sample, probe_meta
