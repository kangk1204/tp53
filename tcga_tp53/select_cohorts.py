from __future__ import annotations

import numpy as np
import pandas as pd

from tcga_tp53.survival import km_median_time
from tcga_tp53.tcga_ids import dedupe_by_patient, is_primary_tumor


def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def prepare_survival_table(surv: pd.DataFrame) -> pd.DataFrame:
    df = surv.copy()
    # Normalize key columns
    for col in ["OS", "OS.time", "PFI", "PFI.time", "DSS", "DSS.time"]:
        if col in df.columns and col.endswith(".time"):
            df[col] = _to_numeric(df[col])
        elif col in df.columns:
            df[col] = _to_numeric(df[col]).astype("Int64")
    if "cancer type abbreviation" in df.columns:
        df["cancer"] = df["cancer type abbreviation"].astype(str)
    return df


def select_worst_os_cancers(
    surv: pd.DataFrame,
    *,
    top_n: int,
    min_samples: int,
    min_events: int,
    primary_tumor_only: bool = True,
    dedupe_patients: bool = True,
) -> pd.DataFrame:
    df = prepare_survival_table(surv)
    if primary_tumor_only:
        df = df[df.index.to_series().map(is_primary_tumor)]
    if dedupe_patients:
        kept = set(dedupe_by_patient(df.index.tolist()))
        df = df[df.index.isin(kept)]

    rows: list[dict] = []
    for cancer, sub in df.groupby("cancer"):
        n = int(sub.shape[0])
        events = int(sub["OS"].fillna(0).astype(int).sum())
        if n < min_samples or events < min_events:
            continue
        med_os = km_median_time(sub["OS.time"], sub["OS"])
        med_pfi = km_median_time(sub["PFI.time"], sub["PFI"]) if "PFI" in sub else None
        med_dss = km_median_time(sub["DSS.time"], sub["DSS"]) if "DSS" in sub else None
        rows.append(
            {
                "cancer": cancer,
                "n": n,
                "os_events": events,
                "os_median_days": med_os,
                "pfi_median_days": med_pfi,
                "dss_median_days": med_dss,
            }
        )

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["os_median_days"]).sort_values("os_median_days", ascending=True)
    out = out.head(top_n).reset_index(drop=True)
    return out


def cohort_samples(
    surv: pd.DataFrame,
    *,
    cancer: str,
    primary_tumor_only: bool = True,
    dedupe_patients: bool = True,
) -> list[str]:
    df = prepare_survival_table(surv)
    df = df[df["cancer"] == cancer]
    if primary_tumor_only:
        df = df[df.index.to_series().map(is_primary_tumor)]
    ids = df.index.tolist()
    if dedupe_patients:
        ids = dedupe_by_patient(ids)
    return ids

