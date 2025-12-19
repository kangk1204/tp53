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
    """
    Build a prerank Series (index=gene, value=t-statistic), sorted descending.
    Ties are broken deterministically by gene name for reproducibility.
    """
    d = de[[gene_col, t_col]].copy().dropna()
    d[gene_col] = d[gene_col].astype(str)
    d = d[~d[gene_col].isin(["", "nan", "None"])]
    d[t_col] = pd.to_numeric(d[t_col], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[t_col])
    d = d.drop_duplicates(subset=[gene_col], keep="first")
    if d.empty:
        return pd.Series(dtype=float, name="rank")
    d = d.sort_values([t_col, gene_col], ascending=[False, True], kind="mergesort")
    return pd.Series(d[t_col].to_numpy(dtype=float), index=d[gene_col].to_numpy(dtype=str), name="rank")


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

    available_libs: set[str] = set()
    try:
        available_libs = set(gp.get_library_name())
    except Exception:
        # If the lookup fails (e.g., offline), we will try running anyway.
        available_libs = set()

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
            is_path = Path(str(lib)).exists()
        except Exception:
            is_path = False
        if (not is_path) and available_libs and lib not in available_libs:
            logger.warning("GSEA skipped (unsupported gene_sets library): %s", lib)
            continue

        try:
            pre = gp.prerank(
                rnk=rank,
                gene_sets=lib,
                threads=1,
                permutation_num=int(cfg.permutations),
                min_size=int(cfg.min_size),
                max_size=int(cfg.max_size),
                seed=int(cfg.seed),
                outdir=None,  # do not let gseapy spam files
                no_plot=True,
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
            lc = str(c).strip().lower()
            if lc == "term":
                rename[c] = "term"
            elif lc == "nes":
                rename[c] = "NES"
            elif lc in {"pval", "p-value", "p_value", "nom p-val", "nom pval", "p"}:
                rename[c] = "p"
            elif lc in {"fdr", "fdr q-val", "fdr_q-val", "fdr_qval", "fdr q-val"}:
                rename[c] = "fdr"
            elif lc in {"es"}:
                rename[c] = "ES"
            elif lc in {"lead_genes", "ledge_genes", "lead genes"}:
                rename[c] = "lead_genes"
            elif lc == "name":
                rename[c] = "name"
            elif lc in {"fwer p-val", "fwer pval"}:
                rename[c] = "fwer_p"
            elif lc in {"tag %", "tag%"}:
                rename[c] = "tag_pct"
            elif lc in {"gene %", "gene%"}:
                rename[c] = "gene_pct"
        if rename:
            res = res.rename(columns=rename)

        if "term" not in res.columns:
            # Some gseapy versions place terms in the index.
            res = res.reset_index().rename(columns={"index": "term"})
        write_tsv(res, tsv_path)
        written.append(tsv_path)

        # Plot: top positive/negative NES (paper-friendly)
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import textwrap

            d = res.copy()
            if "NES" not in d.columns:
                continue
            term_col = None
            if "term" in d.columns:
                # Back-compat: older outputs may have numeric 'term' (row index) and a separate 'Term' column.
                try:
                    numeric_ratio = float(pd.to_numeric(d["term"], errors="coerce").notna().mean())
                except Exception:
                    numeric_ratio = 0.0
                if numeric_ratio > 0.9 and "Term" in d.columns:
                    term_col = "Term"
                else:
                    term_col = "term"
            elif "Term" in d.columns:
                term_col = "Term"
            if term_col is None:
                continue
            d["NES"] = pd.to_numeric(d["NES"], errors="coerce")
            d = d.dropna(subset=["NES", term_col])
            if d.empty:
                continue

            d_pos = d.sort_values("NES", ascending=False).head(10)
            d_neg = d.sort_values("NES", ascending=True).head(10)
            d_plot = pd.concat([d_pos, d_neg], axis=0)
            d_plot = d_plot.drop_duplicates(subset=[term_col]).sort_values("NES", ascending=True)

            sns.set_theme(style="whitegrid")
            fig, ax = plt.subplots(figsize=(7.5, max(4.0, 0.28 * len(d_plot) + 1.0)))
            colors = ["#2ca02c" if v > 0 else "#1f77b4" for v in d_plot["NES"].to_numpy(dtype=float)]
            labels = [textwrap.fill(str(x), width=45) for x in d_plot[term_col].astype(str).tolist()]
            ax.barh(labels, d_plot["NES"].to_numpy(dtype=float), color=colors)
            ax.axvline(0.0, color="black", linewidth=1)
            ax.set_title(f"GSEA prerank ({lib})")
            ax.set_xlabel("Normalized Enrichment Score (NES)")
            ax.tick_params(axis="y", labelsize=8)
            savefig(fig, fig_path, dpi=300)
            written.append(fig_path)
        except Exception:
            continue

    return written
