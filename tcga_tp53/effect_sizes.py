from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def _to_1d_float(x: object) -> np.ndarray:
    a = np.asarray(x, dtype=float).reshape(-1)
    return a[np.isfinite(a)]


def cohens_d(a: object, b: object) -> float | None:
    """
    Cohen's d using pooled SD (assumes roughly equal variances).
    Returns None if insufficient data.
    """
    x = _to_1d_float(a)
    y = _to_1d_float(b)
    if x.size < 2 or y.size < 2:
        return None
    nx = float(x.size)
    ny = float(y.size)
    vx = float(np.var(x, ddof=1))
    vy = float(np.var(y, ddof=1))
    sp2 = ((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1.0)
    if sp2 <= 0:
        return None
    return float((np.mean(x) - np.mean(y)) / np.sqrt(sp2))


def hedges_g(a: object, b: object) -> float | None:
    d = cohens_d(a, b)
    if d is None:
        return None
    x = _to_1d_float(a)
    y = _to_1d_float(b)
    n = x.size + y.size
    if n <= 3:
        return None
    j = 1.0 - (3.0 / (4.0 * n - 9.0))
    return float(j * d)


def mean_diff_ci_welch(a: object, b: object, *, alpha: float = 0.05) -> tuple[float, float, float] | None:
    """
    Mean difference and CI using Welch's t approximation.
    Returns (diff, lo, hi) where diff = mean(a) - mean(b).
    """
    x = _to_1d_float(a)
    y = _to_1d_float(b)
    if x.size < 2 or y.size < 2:
        return None
    mx = float(np.mean(x))
    my = float(np.mean(y))
    vx = float(np.var(x, ddof=1))
    vy = float(np.var(y, ddof=1))
    nx = float(x.size)
    ny = float(y.size)
    se2 = vx / nx + vy / ny
    if se2 <= 0:
        return None
    se = float(np.sqrt(se2))
    # Welch–Satterthwaite df
    df_num = se2 * se2
    df_den = (vx * vx) / (nx * nx * (nx - 1.0)) + (vy * vy) / (ny * ny * (ny - 1.0))
    if df_den <= 0:
        return None
    df = float(df_num / df_den)
    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    diff = mx - my
    lo = diff - tcrit * se
    hi = diff + tcrit * se
    return float(diff), float(lo), float(hi)


def cliffs_delta(a: object, b: object) -> float | None:
    """
    Cliff's delta (non-parametric effect size).
      -1: all a < b, +1: all a > b
    """
    x = _to_1d_float(a)
    y = _to_1d_float(b)
    if x.size == 0 or y.size == 0:
        return None
    # Use Mann–Whitney U based formula: delta = 2U/(nx*ny) - 1
    r = stats.rankdata(np.concatenate([x, y]), method="average")
    rx = float(np.sum(r[: x.size]))
    nx = float(x.size)
    ny = float(y.size)
    u = rx - nx * (nx + 1.0) / 2.0
    return float((2.0 * u) / max(nx * ny, 1.0) - 1.0)


def mannwhitney_u_p(a: object, b: object) -> float | None:
    x = _to_1d_float(a)
    y = _to_1d_float(b)
    if x.size == 0 or y.size == 0:
        return None
    try:
        res = stats.mannwhitneyu(x, y, alternative="two-sided")
        return float(res.pvalue)
    except Exception:
        return None


def odds_ratio_ci(
    a: int, b: int, c: int, d: int, *, correction: float = 0.5
) -> tuple[float, float, float] | None:
    """
    Odds ratio (a/b)/(c/d) with Wald CI on log(OR).
    Adds Haldane–Anscombe correction if any cell is 0.
    """
    if min(a, b, c, d) < 0:
        return None
    aa, bb, cc, dd = float(a), float(b), float(c), float(d)
    if min(a, b, c, d) == 0:
        aa += correction
        bb += correction
        cc += correction
        dd += correction
    if bb <= 0 or cc <= 0:
        return None
    or_ = (aa * dd) / (bb * cc)
    if or_ <= 0:
        return None
    se = float(np.sqrt(1.0 / aa + 1.0 / bb + 1.0 / cc + 1.0 / dd))
    z = float(stats.norm.ppf(0.975))
    lo = float(np.exp(np.log(or_) - z * se))
    hi = float(np.exp(np.log(or_) + z * se))
    return float(or_), lo, hi


@dataclass(frozen=True)
class ChiSquareResult:
    chi2: float
    p: float
    dof: int
    n: int
    cramers_v: float | None


def chi2_independence(table: np.ndarray) -> ChiSquareResult | None:
    """
    Pearson chi-square test + Cramér's V effect size.
    table: 2D contingency table (rows x cols), non-negative counts.
    """
    arr = np.asarray(table, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return None
    if np.any(arr < 0):
        return None
    n = int(arr.sum())
    if n == 0:
        return None
    try:
        chi2, p, dof, _ = stats.chi2_contingency(arr)
    except Exception:
        return None
    r, c = arr.shape
    denom = n * max(min(r - 1, c - 1), 1)
    v = float(np.sqrt(chi2 / denom)) if denom > 0 else None
    return ChiSquareResult(chi2=float(chi2), p=float(p), dof=int(dof), n=n, cramers_v=v)

