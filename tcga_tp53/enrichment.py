from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tcga_tp53.io import ensure_dir, write_tsv
from tcga_tp53.plots import savefig


@dataclass(frozen=True)
class GseaConfig:
    gene_sets: list[str]
    permutations: int = 200
    min_size: int = 10
    max_size: int = 500
    seed: int = 0


def rank_from_de_tstat(de: pd.DataFrame, *, gene_col: str = "gene", t_col: str = "t") -> pd.Series:
    d = de[[gene_col, t_col]].copy().dropna()
    d[gene_col] = d[gene_col].astype(str)
    d = d[~d[gene_col].isin(["", "nan", "None"])]
    d = d.drop_duplicates(subset=[gene_col], keep="first")
    r = pd.to_numeric(d[t_col], errors="coerce")
    out = pd.Series(r.values, index=d[gene_col].values, name="rank")
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out.sort_values(ascending=False)


def run_prerank_gsea(
    rank: pd.Series,
    *,
    table_dir: Path,
    fig_dir: Path,
    prefix: str,
    cfg: GseaConfig,
    overwrite: bool,
) -> list[Path]:
    """
    Runs gseapy.prerank for each gene set library and writes:
      - TSV result table per library
      - A simple barplot of top NES terms per library
    Returns list of written file paths.
    """
    logger = logging.getLogger(__name__)
    ensure_dir(table_dir)
    ensure_dir(fig_dir)

    written: list[Path] = []
    if rank.empty:
        return written

    try:
        import gseapy as gp
    except Exception as e:
        logger.warning("GSEA skipped (gseapy import failed): %s", e)
        return written

    for lib in cfg.gene_sets:
        safe = "".join([c if c.isalnum() or c in {"-", "_", "."} else "_" for c in lib])
        tsv_path = table_dir / f"{prefix}__gsea__{safe}.tsv"
        fig_path = fig_dir / f"{prefix}__gsea__{safe}.png"

        if tsv_path.exists() and not overwrite:
            written.append(tsv_path)
            if fig_path.exists():
                written.append(fig_path)
            continue

        try:
            pre = gp.prerank(
                rnk=rank,
                gene_sets=lib,
                processes=1,
                permutation_num=int(cfg.permutations),
                min_size=int(cfg.min_size),
                max_size=int(cfg.max_size),
                seed=int(cfg.seed),
                outdir=None,  # do not let gseapy spam files
                verbose=False,
            )
        except Exception as e:
            logger.warning("GSEA failed for gene_sets=%s: %s", lib, e)
            continue

        res = getattr(pre, "res2d", None)
        if res is None or not isinstance(res, pd.DataFrame) or res.empty:
            continue

        res = res.copy()
        # Normalize column names (gseapy versions vary)
        rename = {}
        for c in res.columns:
            lc = str(c).lower()
            if lc == "nes":
                rename[c] = "NES"
            elif lc in {"pval", "p-value", "p_value"}:
                rename[c] = "p"
            elif lc in {"fdr", "fdr q-val", "fdr_q-val", "fdr_qval", "fdr q-val"}:
                rename[c] = "fdr"
            elif lc in {"es"}:
                rename[c] = "ES"
            elif lc in {"lead_genes", "ledge_genes", "lead_genes"}:
                rename[c] = "lead_genes"
        if rename:
            res = res.rename(columns=rename)

        res = res.reset_index().rename(columns={"index": "term"})
        write_tsv(res, tsv_path)
        written.append(tsv_path)

        # Plot: top positive/negative NES (paper-friendly)
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            d = res.copy()
            if "NES" not in d.columns:
                continue
            d["NES"] = pd.to_numeric(d["NES"], errors="coerce")
            d = d.dropna(subset=["NES", "term"])
            if d.empty:
                continue

            d_pos = d.sort_values("NES", ascending=False).head(10)
            d_neg = d.sort_values("NES", ascending=True).head(10)
            d_plot = pd.concat([d_pos, d_neg], axis=0)
            d_plot = d_plot.drop_duplicates(subset=["term"]).sort_values("NES", ascending=True)

            sns.set_theme(style="whitegrid")
            fig, ax = plt.subplots(figsize=(7.5, max(4.0, 0.28 * len(d_plot) + 1.0)))
            colors = ["#2ca02c" if v > 0 else "#1f77b4" for v in d_plot["NES"].to_numpy(dtype=float)]
            ax.barh(d_plot["term"].astype(str), d_plot["NES"].to_numpy(dtype=float), color=colors)
            ax.axvline(0.0, color="black", linewidth=1)
            ax.set_title(f"GSEA prerank ({lib})")
            ax.set_xlabel("Normalized Enrichment Score (NES)")
            savefig(fig, fig_path, dpi=300)
            written.append(fig_path)
        except Exception:
            continue

    return written
