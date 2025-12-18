from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection


def fdr_bh(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)
    _, q = fdrcorrection(p, alpha=0.05, method="indep")
    return q


def differential_ttest(
    matrix: pd.DataFrame, *, group: pd.Series, group1_label: str = "group1", group0_label: str = "group0"
) -> pd.DataFrame:
    """
    matrix: rows=features, cols=samples
    group: boolean-like Series indexed by sample id (True=group1, False=group0)
    """
    g = group.reindex(matrix.columns)
    keep = g.notna()
    matrix = matrix.loc[:, keep]
    g = g[keep].astype(bool)
    a = matrix.loc[:, g.values].to_numpy(dtype=float)
    b = matrix.loc[:, (~g.values)].to_numpy(dtype=float)

    if a.shape[1] < 3 or b.shape[1] < 3:
        return pd.DataFrame()

    t, p = stats.ttest_ind(a, b, axis=1, equal_var=False, nan_policy="omit")
    out = pd.DataFrame(
        {
            "feature": matrix.index,
            f"n_{group1_label}": int(a.shape[1]),
            f"n_{group0_label}": int(b.shape[1]),
            f"mean_{group1_label}": np.nanmean(a, axis=1),
            f"mean_{group0_label}": np.nanmean(b, axis=1),
            "diff_mean": np.nanmean(a, axis=1) - np.nanmean(b, axis=1),
            "t": t,
            "p": p,
        }
    )
    out["fdr"] = fdr_bh(out["p"].to_numpy())
    out = out.sort_values(["fdr", "p"], ascending=[True, True]).reset_index(drop=True)
    return out

