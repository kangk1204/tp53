#!/usr/bin/env python
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcga_tp53.plots import savefig


def _guess_nes_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if str(c).strip().lower() == "nes":
            return str(c)
    return None


def _guess_term_col(df: pd.DataFrame) -> str | None:
    cols = {str(c): str(c).strip().lower() for c in df.columns}
    term = next((c for c, lc in cols.items() if lc == "term"), None)
    Term = next((c for c, lc in cols.items() if lc == "term" and c != term), None)
    # Some older outputs include both `term` (row index) and `Term` (gene set name).
    if term is not None:
        if "Term" in df.columns:
            try:
                numeric_ratio = float(pd.to_numeric(df[term], errors="coerce").notna().mean())
            except Exception:
                numeric_ratio = 0.0
            if numeric_ratio > 0.9:
                return "Term"
        return term
    if "Term" in df.columns:
        return "Term"
    # Fall back to a second "term" column name variant if present.
    if Term is not None:
        return Term
    return None


def plot_gsea_barplot_from_tsv(
    tsv_path: Path,
    *,
    out_path: Path,
    top_n: int,
    wrap_width: int,
    dpi: int,
) -> bool:
    df = pd.read_csv(tsv_path, sep="\t")
    if df.empty:
        return False

    nes_col = _guess_nes_col(df)
    term_col = _guess_term_col(df)
    if nes_col is None or term_col is None:
        return False

    d = df.copy()
    d[nes_col] = pd.to_numeric(d[nes_col], errors="coerce")
    d = d.dropna(subset=[nes_col, term_col])
    if d.empty:
        return False

    d_pos = d.sort_values(nes_col, ascending=False).head(top_n)
    d_neg = d.sort_values(nes_col, ascending=True).head(top_n)
    d_plot = pd.concat([d_pos, d_neg], axis=0)
    d_plot = d_plot.drop_duplicates(subset=[term_col]).sort_values(nes_col, ascending=True)
    if d_plot.empty:
        return False

    lib = tsv_path.stem.split("__gsea__", 1)[1] if "__gsea__" in tsv_path.stem else tsv_path.stem

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(7.5, max(4.0, 0.33 * len(d_plot) + 1.0)))
    colors = ["#2ca02c" if float(v) > 0 else "#1f77b4" for v in d_plot[nes_col].to_numpy(dtype=float)]
    labels = [textwrap.fill(str(x), width=int(wrap_width)) for x in d_plot[term_col].astype(str).tolist()]
    ax.barh(labels, d_plot[nes_col].to_numpy(dtype=float), color=colors)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_title(f"GSEA prerank ({lib})")
    ax.set_xlabel("Normalized Enrichment Score (NES)")
    ax.tick_params(axis="y", labelsize=8)
    savefig(fig, out_path, dpi=int(dpi))
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Regenerate GSEA barplots from saved TSV outputs.")
    p.add_argument("--out", type=Path, default=Path("results_full"), help="Results directory (e.g., results_full)")
    p.add_argument("--top-n", type=int, default=10, help="Top positive + top negative terms to plot")
    p.add_argument("--wrap-width", type=int, default=45, help="Wrap pathway labels at this character width")
    p.add_argument("--dpi", type=int, default=300, help="PNG DPI")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing PNGs")
    args = p.parse_args()

    out_dir: Path = args.out
    tsvs = sorted(out_dir.rglob("tables/gsea/*.tsv"))
    if not tsvs:
        print(f"No GSEA TSVs found under: {out_dir}")
        return

    n_ok = 0
    n_skip = 0
    for tsv in tsvs:
        # results/<CANCER>/tables/gsea/<file>.tsv -> results/<CANCER>/figures/gsea/<file>.png
        try:
            cancer_dir = tsv.parents[2]
        except Exception:
            n_skip += 1
            continue
        fig_path = cancer_dir / "figures" / "gsea" / f"{tsv.stem}.png"
        if fig_path.exists() and not args.overwrite:
            n_skip += 1
            continue
        ok = plot_gsea_barplot_from_tsv(
            tsv,
            out_path=fig_path,
            top_n=int(args.top_n),
            wrap_width=int(args.wrap_width),
            dpi=int(args.dpi),
        )
        if ok:
            n_ok += 1
        else:
            n_skip += 1

    print(f"Regenerated: {n_ok}, skipped/failed: {n_skip}")


if __name__ == "__main__":
    main()

