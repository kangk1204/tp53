from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from tcga_tp53.stats import fdr_bh


def cox_gene_tp53_interaction_screen(
    *,
    expr: pd.DataFrame,
    clinical: pd.DataFrame,
    time_col: str,
    event_col: str,
    tp53_col: str = "TP53_mut",
    genes: list[str],
    penalizer: float = 0.1,
    min_events: int = 20,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Per-gene Cox model with interaction:
      h(t) ~ x + TP53 + x*TP53

    Returns one row per gene with interaction p-value and implied group-specific HRs.
    Set n_jobs>1 to parallelize across genes (threading backend).
    """
    try:
        n_jobs = int(n_jobs)
    except Exception:
        n_jobs = 1
    n_jobs = max(1, n_jobs)

    common_samples = [s for s in clinical.index if s in expr.columns]
    if not common_samples:
        return pd.DataFrame()

    d0 = clinical.loc[common_samples, [time_col, event_col, tp53_col]].copy()
    d0[time_col] = pd.to_numeric(d0[time_col], errors="coerce")
    d0[event_col] = pd.to_numeric(d0[event_col], errors="coerce")
    d0[tp53_col] = pd.to_numeric(d0[tp53_col], errors="coerce")
    d0 = d0.dropna()
    if d0.empty:
        return pd.DataFrame()

    # Require both groups present
    if d0[tp53_col].nunique() != 2:
        return pd.DataFrame()

    if int(d0[event_col].sum()) < min_events:
        return pd.DataFrame()

    def fit_one_gene(gene: str) -> dict | None:
        if gene not in expr.index:
            return None
        x = pd.to_numeric(expr.loc[gene, d0.index], errors="coerce")
        # z-score for stability
        x = (x - x.mean()) / (x.std(ddof=0) + 1e-8)
        df = d0.copy()
        df["x"] = x
        df["x_tp53"] = df["x"] * df[tp53_col]
        df = df.dropna()
        if df.empty or int(df[event_col].sum()) < min_events:
            return None

        cph = CoxPHFitter(penalizer=penalizer)
        try:
            cph.fit(
                df[[time_col, event_col, "x", tp53_col, "x_tp53"]],
                duration_col=time_col,
                event_col=event_col,
            )
        except Exception:
            return None

        summ = cph.summary
        if "x_tp53" not in summ.index or "x" not in summ.index:
            return None

        coef_x = float(summ.loc["x", "coef"])
        coef_int = float(summ.loc["x_tp53", "coef"])
        p_int = float(summ.loc["x_tp53", "p"])

        hr_wt = float(np.exp(coef_x))
        hr_mut = float(np.exp(coef_x + coef_int))

        return {
            "gene": gene,
            "p_interaction": p_int,
            "coef_x": coef_x,
            "coef_x_tp53": coef_int,
            "hr_wt": hr_wt,
            "hr_mut": hr_mut,
            "n": int(df.shape[0]),
            "events": int(df[event_col].sum()),
        }

    if n_jobs > 1:
        try:
            from joblib import Parallel, delayed

            rows = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(fit_one_gene)(g) for g in genes)
            results = [r for r in rows if r is not None]
        except Exception:
            results = []
            for g in genes:
                row = fit_one_gene(g)
                if row is not None:
                    results.append(row)
    else:
        results = []
        for g in genes:
            row = fit_one_gene(g)
            if row is not None:
                results.append(row)

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    out["fdr_interaction"] = fdr_bh(out["p_interaction"].to_numpy())
    out = out.sort_values(["fdr_interaction", "p_interaction"]).reset_index(drop=True)
    return out
