from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class RandomEffectsMeta:
    endpoint: str | None
    k: int
    log_effect: float
    se: float
    ci_lower: float
    ci_upper: float
    p: float
    tau2: float
    q: float
    q_p: float
    i2: float

    @property
    def effect(self) -> float:
        return float(np.exp(self.log_effect))

    @property
    def effect_ci_lower(self) -> float:
        return float(np.exp(self.ci_lower))

    @property
    def effect_ci_upper(self) -> float:
        return float(np.exp(self.ci_upper))


def log_hr_se_from_ci(hr: float, lo: float, hi: float) -> tuple[float, float] | None:
    if not np.isfinite(hr) or not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hr <= 0 or lo <= 0 or hi <= 0:
        return None
    if lo >= hi:
        return None
    log_hr = float(np.log(hr))
    se = float((np.log(hi) - np.log(lo)) / (2.0 * 1.96))
    if not np.isfinite(se) or se <= 0:
        return None
    return log_hr, se


def random_effects_meta(
    log_effects: np.ndarray,
    ses: np.ndarray,
    *,
    endpoint: str | None = None,
    alpha: float = 0.05,
) -> RandomEffectsMeta | None:
    """
    DerSimonian–Laird random-effects meta-analysis.
    """
    theta = np.asarray(log_effects, dtype=float)
    se = np.asarray(ses, dtype=float)
    ok = np.isfinite(theta) & np.isfinite(se) & (se > 0)
    theta = theta[ok]
    se = se[ok]
    k = int(theta.size)
    if k < 2:
        return None

    w = 1.0 / (se * se)
    theta_fixed = float(np.sum(w * theta) / np.sum(w))
    q = float(np.sum(w * (theta - theta_fixed) ** 2))
    df = k - 1
    q_p = float(1.0 - stats.chi2.cdf(q, df))
    c = float(np.sum(w) - (np.sum(w * w) / np.sum(w)))
    tau2 = float(max(0.0, (q - df) / c)) if c > 0 else 0.0

    w_re = 1.0 / (se * se + tau2)
    theta_re = float(np.sum(w_re * theta) / np.sum(w_re))
    se_re = float(np.sqrt(1.0 / np.sum(w_re)))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    ci_lo = float(theta_re - z * se_re)
    ci_hi = float(theta_re + z * se_re)
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(theta_re / se_re))))

    i2 = 0.0
    if q > 0 and df > 0:
        i2 = float(max(0.0, (q - df) / q) * 100.0)

    return RandomEffectsMeta(
        endpoint=endpoint,
        k=k,
        log_effect=theta_re,
        se=se_re,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        p=p,
        tau2=tau2,
        q=q,
        q_p=q_p,
        i2=i2,
    )

