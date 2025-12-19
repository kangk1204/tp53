from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test


def init_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    sns.set_palette("colorblind")


def savefig(fig: plt.Figure, path: Path, *, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def km_plot(
    df: pd.DataFrame,
    *,
    time_col: str,
    event_col: str,
    group_col: str,
    title: str,
    out_path: Path,
    group_labels: dict[object, str] | None = None,
    palette: dict[object, str] | None = None,
    show_risk_table: bool = True,
) -> None:
    init_style()
    d = df[[time_col, event_col, group_col]].copy()
    d = d.dropna()
    if d[group_col].nunique() != 2:
        return

    groups = list(sorted(d[group_col].unique()))
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    kmfs: list[KaplanMeierFitter] = []
    for g in groups:
        sub = d[d[group_col] == g]
        lab = group_labels.get(g, str(g)) if group_labels else str(g)
        kmf = KaplanMeierFitter()
        kmf.fit(sub[time_col], event_observed=sub[event_col], label=lab)
        kwargs: dict[str, object] = {"ax": ax, "ci_show": True, "linewidth": 2}
        if palette and g in palette:
            kwargs["color"] = palette[g]
        kmf.plot_survival_function(**kwargs)
        kmfs.append(kmf)

    a = d[d[group_col] == groups[0]]
    b = d[d[group_col] == groups[1]]
    p = logrank_test(
        a[time_col], b[time_col], event_observed_A=a[event_col], event_observed_B=b[event_col]
    ).p_value
    ax.set_title(f"{title}\nlog-rank p={p:.2e}")
    ax.set_xlabel("Days")
    ax.set_ylabel("Survival probability")
    ax.legend(title=None, frameon=True, loc="best")

    if show_risk_table and kmfs:
        try:
            from lifelines.plotting import add_at_risk_counts

            add_at_risk_counts(*kmfs, ax=ax)
        except Exception:
            pass
    savefig(fig, out_path)


def km_plot_multi(
    df: pd.DataFrame,
    *,
    time_col: str,
    event_col: str,
    group_col: str,
    title: str,
    out_path: Path,
    order: list[str] | None = None,
    palette: dict[str, str] | None = None,
    show_risk_table: bool = True,
) -> None:
    init_style()
    d = df[[time_col, event_col, group_col]].copy().dropna()
    if d.empty or d[group_col].nunique() < 2:
        return
    d[group_col] = d[group_col].astype(str)
    groups = order or sorted(d[group_col].unique().tolist())

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    kmfs: list[KaplanMeierFitter] = []
    for g in groups:
        sub = d[d[group_col] == g]
        if sub.empty:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub[time_col], event_observed=sub[event_col], label=str(g))
        kwargs: dict[str, object] = {"ax": ax, "ci_show": True, "linewidth": 2}
        if palette and g in palette:
            kwargs["color"] = palette[g]
        kmf.plot_survival_function(**kwargs)
        kmfs.append(kmf)

    try:
        p = float(multivariate_logrank_test(d[time_col], d[group_col], d[event_col]).p_value)
    except Exception:
        p = float("nan")
    if np.isfinite(p):
        ax.set_title(f"{title}\nlog-rank p={p:.2e}")
    else:
        ax.set_title(title)
    ax.set_xlabel("Days")
    ax.set_ylabel("Survival probability")
    ax.legend(title=None, frameon=True, loc="best")

    if show_risk_table and kmfs:
        try:
            from lifelines.plotting import add_at_risk_counts

            add_at_risk_counts(*kmfs, ax=ax)
        except Exception:
            pass

    savefig(fig, out_path, dpi=300)


def volcano_plot(
    df: pd.DataFrame,
    *,
    effect_col: str,
    p_col: str,
    label_col: str,
    title: str,
    out_path: Path,
    top_n_labels: int = 15,
    fdr_col: str | None = "fdr",
    fdr_threshold: float = 0.05,
) -> None:
    init_style()
    cols = [effect_col, p_col, label_col]
    if fdr_col and fdr_col in df.columns:
        cols.append(fdr_col)
    d = df[cols].copy().dropna(subset=[effect_col, p_col, label_col])
    if d.empty:
        return
    d[effect_col] = pd.to_numeric(d[effect_col], errors="coerce")
    d[p_col] = pd.to_numeric(d[p_col], errors="coerce")
    if fdr_col and fdr_col in d.columns:
        d[fdr_col] = pd.to_numeric(d[fdr_col], errors="coerce")
    d = d.dropna(subset=[effect_col, p_col])
    if d.empty:
        return

    d["neglog10p"] = -np.log10(d[p_col].clip(lower=1e-300))
    fig, ax = plt.subplots(figsize=(6, 5))
    is_sig = None
    if fdr_col and fdr_col in d.columns:
        is_sig = d[fdr_col] <= float(fdr_threshold)
    if is_sig is None:
        ax.scatter(d[effect_col], d["neglog10p"], s=10, alpha=0.6)
    else:
        ax.scatter(
            d.loc[~is_sig, effect_col],
            d.loc[~is_sig, "neglog10p"],
            s=10,
            alpha=0.35,
            color="gray",
        )
        ax.scatter(
            d.loc[is_sig, effect_col],
            d.loc[is_sig, "neglog10p"],
            s=12,
            alpha=0.8,
            color="#d62728",
        )
    ax.set_title(title)
    ax.set_xlabel(effect_col)
    ax.set_ylabel(f"-log10({p_col})")

    # Label top hits
    sort_cols = [p_col]
    if fdr_col and fdr_col in d.columns:
        sort_cols = [fdr_col, p_col]
    top = d.sort_values(sort_cols, ascending=True).head(top_n_labels)
    for _, r in top.iterrows():
        ax.text(r[effect_col], r["neglog10p"], str(r[label_col]), fontsize=7)

    savefig(fig, out_path)


def violin_box_plot(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    out_path: Path,
    order: list[str] | None = None,
    palette: dict[str, str] | None = None,
    ylabel: str | None = None,
) -> None:
    init_style()
    d = df[[x_col, y_col]].copy().dropna()
    if d.empty:
        return
    d[x_col] = d[x_col].astype(str)
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna(subset=[y_col])
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    sns.violinplot(
        data=d,
        x=x_col,
        y=y_col,
        order=order,
        palette=palette,
        inner=None,
        cut=0,
        linewidth=1,
        ax=ax,
    )
    sns.boxplot(
        data=d,
        x=x_col,
        y=y_col,
        order=order,
        width=0.35,
        showcaps=True,
        boxprops={"facecolor": "white", "zorder": 3},
        showfliers=False,
        whiskerprops={"linewidth": 1},
        medianprops={"color": "black", "linewidth": 2},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel or y_col)
    savefig(fig, out_path, dpi=300)


def top_barplot(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    out_path: Path,
    n: int = 20,
) -> None:
    init_style()
    d = df[[x_col, y_col]].copy().dropna()
    if d.empty:
        return
    d = d.sort_values(x_col, ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.25 * len(d) + 1)))
    sns.barplot(data=d, x=x_col, y=y_col, ax=ax, orient="h")
    ax.set_title(title)
    savefig(fig, out_path)
