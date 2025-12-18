from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def init_style() -> None:
    sns.set_theme(style="whitegrid")


def savefig(fig: plt.Figure, path: Path, *, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def km_plot(
    df: pd.DataFrame,
    *,
    time_col: str,
    event_col: str,
    group_col: str,
    title: str,
    out_path: Path,
) -> None:
    init_style()
    d = df[[time_col, event_col, group_col]].copy()
    d = d.dropna()
    if d[group_col].nunique() != 2:
        return

    groups = list(sorted(d[group_col].unique()))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    kmf = KaplanMeierFitter()
    for g in groups:
        sub = d[d[group_col] == g]
        kmf.fit(sub[time_col], event_observed=sub[event_col], label=str(g))
        kmf.plot_survival_function(ax=ax, ci_show=False)

    a = d[d[group_col] == groups[0]]
    b = d[d[group_col] == groups[1]]
    p = logrank_test(
        a[time_col], b[time_col], event_observed_A=a[event_col], event_observed_B=b[event_col]
    ).p_value
    ax.set_title(f"{title}\nlog-rank p={p:.2e}")
    ax.set_xlabel("Days")
    ax.set_ylabel("Survival probability")
    savefig(fig, out_path)


def volcano_plot(
    df: pd.DataFrame,
    *,
    effect_col: str,
    p_col: str,
    label_col: str,
    title: str,
    out_path: Path,
    top_n_labels: int = 15,
) -> None:
    init_style()
    d = df[[effect_col, p_col, label_col]].copy().dropna()
    if d.empty:
        return
    d["neglog10p"] = -np.log10(d[p_col].clip(lower=1e-300))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(d[effect_col], d["neglog10p"], s=8, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel(effect_col)
    ax.set_ylabel(f"-log10({p_col})")

    # Label top hits
    top = d.nsmallest(top_n_labels, p_col)
    for _, r in top.iterrows():
        ax.text(r[effect_col], r["neglog10p"], str(r[label_col]), fontsize=7)

    savefig(fig, out_path)


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

