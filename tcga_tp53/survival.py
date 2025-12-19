from __future__ import annotations

import re

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, proportional_hazard_test


def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def parse_stage_to_int(stage: str | float | None) -> int | None:
    if stage is None or (isinstance(stage, float) and np.isnan(stage)):
        return None
    s = str(stage).strip().upper()
    if not s or s in {"NA", "NAN"}:
        return None
    # Common patterns: "Stage II", "STAGE IIA", "Stage IVB", "stage x"
    m = re.search(r"STAGE\s*([IVX]+)", s)
    if not m:
        # Fallbacks: sometimes stage is provided without the "Stage" prefix or as numeric.
        m = re.search(r"^([IVX]+)\b", s)
    if m:
        roman = m.group(1)
        mapping = {"I": 1, "II": 2, "III": 3, "IV": 4}
        return mapping.get(roman)
    m_num = re.search(r"STAGE\s*(\d+)", s)
    if m_num:
        try:
            v = int(m_num.group(1))
        except Exception:
            return None
        return v if 1 <= v <= 4 else None
    return None


def km_median_time(time: pd.Series, event: pd.Series) -> float | None:
    df = pd.DataFrame({"time": _to_numeric(time), "event": _to_numeric(event)})
    df = df.dropna()
    if df.empty:
        return None
    km = KaplanMeierFitter()
    km.fit(df["time"], event_observed=df["event"])
    median = km.median_survival_time_
    if median is None or (isinstance(median, float) and np.isnan(median)):
        return None
    return float(median)


def km_logrank_p(
    time: pd.Series, event: pd.Series, group: pd.Series
) -> float | None:
    df = pd.DataFrame({"time": _to_numeric(time), "event": _to_numeric(event), "group": group})
    df = df.dropna()
    if df["group"].nunique() != 2:
        return None
    g1, g0 = sorted(df["group"].unique())
    a = df[df["group"] == g1]
    b = df[df["group"] == g0]
    if a.empty or b.empty:
        return None
    res = logrank_test(a["time"], b["time"], event_observed_A=a["event"], event_observed_B=b["event"])
    return float(res.p_value)


def fit_cox_tp53(
    df: pd.DataFrame,
    *,
    time_col: str,
    event_col: str,
    tp53_col: str = "TP53_mut",
    covariates: list[str] | None = None,
    penalizer: float = 0.1,
) -> pd.DataFrame:
    covariates = covariates or []
    cols = [time_col, event_col, tp53_col, *covariates]
    if any(c not in df.columns for c in cols):
        return pd.DataFrame()

    model_df = df[cols].copy()
    model_df[time_col] = _to_numeric(model_df[time_col])
    model_df[event_col] = _to_numeric(model_df[event_col])
    model_df[tp53_col] = _to_numeric(model_df[tp53_col])

    # Basic type cleanup
    for c in covariates:
        if model_df[c].dtype == "object":
            continue
        model_df[c] = _to_numeric(model_df[c])

    # Encode categoricals (e.g., gender)
    cat_cols = [c for c in covariates if model_df[c].dtype == "object"]
    if cat_cols:
        # pd.get_dummies drops NaNs; make this explicit for reproducibility.
        model_df = model_df.dropna(subset=cat_cols)
        model_df = pd.get_dummies(model_df, columns=cat_cols, drop_first=True)

    # Complete-case: lifelines CoxPHFitter does not accept NaNs.
    model_df = model_df.dropna()
    if model_df.empty or float(model_df[event_col].sum()) < 5:
        return pd.DataFrame()
    if model_df[tp53_col].nunique(dropna=True) < 2:
        return pd.DataFrame()

    # Drop constant columns (lifelines fails on them)
    for c in list(model_df.columns):
        if c in {time_col, event_col}:
            continue
        if model_df[c].nunique(dropna=True) <= 1:
            model_df = model_df.drop(columns=[c])
    if tp53_col not in model_df.columns:
        return pd.DataFrame()

    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(model_df, duration_col=time_col, event_col=event_col)
    except Exception:
        return pd.DataFrame()

    out = cph.summary.reset_index()
    if "covariate" in out.columns:
        out = out.rename(columns={"covariate": "term"})
    elif "index" in out.columns:
        out = out.rename(columns={"index": "term"})
    else:
        out = out.rename(columns={out.columns[0]: "term"})

    out["n"] = int(model_df.shape[0])
    out["events"] = int(pd.to_numeric(model_df[event_col], errors="coerce").fillna(0).astype(int).sum())

    # PH assumption check (Schoenfeld residual-based test)
    try:
        ph = proportional_hazard_test(cph, model_df, time_transform="rank").summary.reset_index()
        if "index" in ph.columns:
            ph = ph.rename(columns={"index": "term"})
        ph = ph.rename(columns={"test_statistic": "ph_test_statistic", "p": "ph_p"})
        out = out.merge(ph[["term", "ph_test_statistic", "ph_p"]], on="term", how="left")
    except Exception:
        pass

    return out


def fit_cox_stratified(
    df: pd.DataFrame,
    *,
    time_col: str,
    event_col: str,
    covariates: list[str],
    strata_col: str,
    penalizer: float = 0.1,
) -> pd.DataFrame:
    """
    Cox PH with strata (e.g., stratify by cancer type).
    """
    cols = [time_col, event_col, strata_col, *covariates]
    if any(c not in df.columns for c in cols):
        return pd.DataFrame()

    model_df = df[cols].copy()
    model_df[time_col] = _to_numeric(model_df[time_col])
    model_df[event_col] = _to_numeric(model_df[event_col])
    for c in covariates:
        if model_df[c].dtype == "object":
            continue
        model_df[c] = _to_numeric(model_df[c])

    cat_cols = [c for c in covariates if model_df[c].dtype == "object"]
    if cat_cols:
        model_df = model_df.dropna(subset=cat_cols)
        model_df = pd.get_dummies(model_df, columns=cat_cols, drop_first=True)

    model_df = model_df.dropna()
    if model_df.empty or float(model_df[event_col].sum()) < 10:
        return pd.DataFrame()

    # Drop constant columns (lifelines fails on them)
    for c in list(model_df.columns):
        if c in {time_col, event_col, strata_col}:
            continue
        if model_df[c].nunique(dropna=True) <= 1:
            model_df = model_df.drop(columns=[c])

    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(model_df, duration_col=time_col, event_col=event_col, strata=[strata_col])
    except Exception:
        return pd.DataFrame()

    out = cph.summary.reset_index()
    if "covariate" in out.columns:
        out = out.rename(columns={"covariate": "term"})
    elif "index" in out.columns:
        out = out.rename(columns={"index": "term"})
    else:
        out = out.rename(columns={out.columns[0]: "term"})
    out["n"] = int(model_df.shape[0])
    out["events"] = int(pd.to_numeric(model_df[event_col], errors="coerce").fillna(0).astype(int).sum())
    out["strata_col"] = strata_col
    return out


def fit_cox_categorical_group(
    df: pd.DataFrame,
    *,
    time_col: str,
    event_col: str,
    group_col: str,
    covariates: list[str] | None = None,
    categories: list[str] | None = None,
    penalizer: float = 0.1,
) -> pd.DataFrame:
    """
    Cox PH with a categorical group variable encoded via one-hot (baseline = first category).
    """
    covariates = covariates or []
    cols = [time_col, event_col, group_col, *covariates]
    if any(c not in df.columns for c in cols):
        return pd.DataFrame()

    model_df = df[cols].copy()
    model_df[time_col] = _to_numeric(model_df[time_col])
    model_df[event_col] = _to_numeric(model_df[event_col])
    model_df[group_col] = model_df[group_col].astype(str)

    for c in covariates:
        if model_df[c].dtype == "object":
            continue
        model_df[c] = _to_numeric(model_df[c])

    model_df = model_df.dropna(subset=[time_col, event_col, group_col])
    if model_df.empty or float(model_df[event_col].sum()) < 5:
        return pd.DataFrame()
    if model_df[group_col].nunique(dropna=True) < 2:
        return pd.DataFrame()

    if categories:
        model_df[group_col] = pd.Categorical(model_df[group_col], categories=categories, ordered=True)

    # Encode categoricals including group_col.
    cat_cols = [group_col] + [c for c in covariates if model_df[c].dtype == "object"]
    model_df = model_df.dropna(subset=cat_cols)
    model_df = pd.get_dummies(model_df, columns=cat_cols, drop_first=True)

    model_df = model_df.dropna()
    if model_df.empty or float(model_df[event_col].sum()) < 5:
        return pd.DataFrame()

    # Drop constant columns.
    for c in list(model_df.columns):
        if c in {time_col, event_col}:
            continue
        if model_df[c].nunique(dropna=True) <= 1:
            model_df = model_df.drop(columns=[c])

    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(model_df, duration_col=time_col, event_col=event_col)
    except Exception:
        return pd.DataFrame()

    out = cph.summary.reset_index()
    if "covariate" in out.columns:
        out = out.rename(columns={"covariate": "term"})
    elif "index" in out.columns:
        out = out.rename(columns={"index": "term"})
    else:
        out = out.rename(columns={out.columns[0]: "term"})
    out["n"] = int(model_df.shape[0])
    out["events"] = int(pd.to_numeric(model_df[event_col], errors="coerce").fillna(0).astype(int).sum())

    try:
        ph = proportional_hazard_test(cph, model_df, time_transform="rank").summary.reset_index()
        if "index" in ph.columns:
            ph = ph.rename(columns={"index": "term"})
        ph = ph.rename(columns={"test_statistic": "ph_test_statistic", "p": "ph_p"})
        out = out.merge(ph[["term", "ph_test_statistic", "ph_p"]], on="term", how="left")
    except Exception:
        pass

    return out


def summarize_endpoint_by_group(
    df: pd.DataFrame, *, time_col: str, event_col: str, group_col: str
) -> pd.DataFrame:
    rows: list[dict] = []
    for g, sub in df.groupby(group_col, dropna=False):
        rows.append(
            {
                "group": g,
                "n": int(sub.shape[0]),
                "events": int(_to_numeric(sub[event_col]).fillna(0).astype(int).sum()),
                "km_median_days": km_median_time(sub[time_col], sub[event_col]),
            }
        )
    return pd.DataFrame(rows)
