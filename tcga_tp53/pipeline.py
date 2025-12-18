from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tcga_tp53.analyses import (
    comutation_against_tp53,
    differential_cnv,
    differential_expression,
    differential_immune,
    differential_methylation,
    immune_subtype_crosstab,
    methylation_gene_averages,
    summarize_tp53_sparse,
    tmb_from_mut_gene,
    top_tp53_hotspots,
    tp53_binary_from_mut_gene,
)
from tcga_tp53.cache import stable_hash
from tcga_tp53.config import XenaDatasets
from tcga_tp53.io import ensure_dir, write_tsv
from tcga_tp53.plots import km_plot, savefig, top_barplot, volcano_plot
from tcga_tp53.select_cohorts import cohort_samples, select_worst_os_cancers
from tcga_tp53.survival import fit_cox_tp53, parse_stage_to_int, summarize_endpoint_by_group
from tcga_tp53.xena_client import XenaClient
from tcga_tp53.gene_survival import cox_gene_tp53_interaction_screen


def _safe_float(x: object) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def _readable_stage(df: pd.DataFrame) -> pd.Series:
    for col in ["ajcc_pathologic_tumor_stage", "clinical_stage"]:
        if col in df.columns:
            return df[col].apply(parse_stage_to_int)
    return pd.Series(index=df.index, dtype="Int64")


def _extract_tp53_term(cox: pd.DataFrame) -> pd.Series | None:
    if cox.empty:
        return None
    row = cox[cox["term"] == "TP53_mut"]
    if row.empty:
        return None
    return row.iloc[0]


@dataclass
class PipelineParams:
    out_dir: Path
    cache_dir: Path
    top_n: int
    min_samples: int
    min_events: int
    methylation_top_genes: int
    max_survival_genes: int
    threads: int
    seed: int
    overwrite: bool


def run_pipeline(
    *,
    out_dir: Path,
    cache_dir: Path,
    top_n: int = 10,
    min_samples: int = 80,
    min_events: int = 30,
    methylation_top_genes: int = 200,
    max_survival_genes: int = 0,
    threads: int = 4,
    seed: int = 0,
    overwrite: bool = False,
    show_progress: bool = True,
) -> None:
    logger = logging.getLogger(__name__)
    params = PipelineParams(
        out_dir=out_dir,
        cache_dir=cache_dir,
        top_n=top_n,
        min_samples=min_samples,
        min_events=min_events,
        methylation_top_genes=methylation_top_genes,
        max_survival_genes=max_survival_genes,
        threads=threads,
        seed=seed,
        overwrite=overwrite,
    )
    ensure_dir(out_dir)

    logger.info(
        "starting pipeline: out=%s cache=%s top_n=%d min_samples=%d min_events=%d methylation_top_genes=%d max_survival_genes=%d",
        out_dir,
        cache_dir,
        top_n,
        min_samples,
        min_events,
        methylation_top_genes,
        max_survival_genes,
    )

    client = XenaClient(cache_dir=cache_dir, overwrite_cache=False, show_progress=show_progress)

    t0 = time.perf_counter()
    surv = client.fetch_clinical(XenaDatasets.SURVIVAL, cache_key="all")
    surv = client.decode_field_codes(
        XenaDatasets.SURVIVAL,
        surv,
        fields=[
            "cancer type abbreviation",
            "gender",
            "ajcc_pathologic_tumor_stage",
            "clinical_stage",
        ],
    )
    logger.info(
        "loaded survival table: %d samples x %d cols (%.1fs)",
        surv.shape[0],
        surv.shape[1],
        time.perf_counter() - t0,
    )
    worst = select_worst_os_cancers(
        surv,
        top_n=top_n,
        min_samples=min_samples,
        min_events=min_events,
    )
    write_tsv(worst, out_dir / "selected_cohorts.tsv")
    logger.info("selected worst cancers (OS): %s", ", ".join(worst["cancer"].tolist()))

    # Cache global small clinical matrices (immune subtype, stemness methylation) once.
    t1 = time.perf_counter()
    immune_subtype_all = client.fetch_clinical(XenaDatasets.IMMUNE_SUBTYPE, cache_key="all")
    immune_subtype_all = client.decode_field_codes(
        XenaDatasets.IMMUNE_SUBTYPE,
        immune_subtype_all,
        fields=["Subtype_Immune_Model_Based"],
    )
    stemness_meth_all = client.fetch_clinical(XenaDatasets.STEMNESS_METH, cache_key="all")
    logger.info("loaded global clinical matrices (%.1fs)", time.perf_counter() - t1)

    # Pre-fetch sample sets once (avoid repeated API calls inside per-cancer loop).
    t2 = time.perf_counter()
    rna_samples_all = set(client.dataset_samples(XenaDatasets.RNA))
    cnv_samples_all = set(client.dataset_samples(XenaDatasets.CNV_GENE))
    mut_samples_all = set(client.dataset_samples(XenaDatasets.MUT_GENE))
    immune_samples_all = set(client.dataset_samples(XenaDatasets.IMMUNE_SIGS))
    meth_samples_all = set(client.dataset_samples(XenaDatasets.METH450))
    logger.info(
        "sample coverage: RNA=%d CNV=%d MUT=%d IMMUNE=%d METH=%d (%.1fs)",
        len(rna_samples_all),
        len(cnv_samples_all),
        len(mut_samples_all),
        len(immune_samples_all),
        len(meth_samples_all),
        time.perf_counter() - t2,
    )

    # Pre-fetch probe lists (genes/features) once to avoid repeated API calls.
    t3 = time.perf_counter()
    expr_genes = client.dataset_fields(XenaDatasets.RNA)
    cnv_genes = client.dataset_fields(XenaDatasets.CNV_GENE)
    mut_genes = client.dataset_fields(XenaDatasets.MUT_GENE)
    immune_sigs = client.dataset_fields(XenaDatasets.IMMUNE_SIGS)
    immune_sigs = [x for x in immune_sigs if x != "sampleID"]
    logger.info(
        "feature counts: expr=%d cnv=%d mut=%d immune=%d (%.1fs)",
        len(expr_genes),
        len(cnv_genes),
        len(mut_genes),
        len(immune_sigs),
        time.perf_counter() - t3,
    )

    pancancer_rows: list[dict] = []
    cancers = worst["cancer"].tolist()
    try:
        from tqdm import tqdm

        iterator = tqdm(cancers, desc="Per-cancer analyses", disable=not show_progress)
    except Exception:
        iterator = cancers

    for cancer in iterator:
        logger.info("[%s] start", cancer)
        t_cancer = time.perf_counter()
        _run_one_cancer(
            client=client,
            surv=surv,
            immune_subtype_all=immune_subtype_all,
            stemness_meth_all=stemness_meth_all,
            cancer=cancer,
            out_base=out_dir / cancer,
            expr_genes=expr_genes,
            cnv_genes=cnv_genes,
            mut_genes=mut_genes,
            immune_sigs=immune_sigs,
            rna_samples_all=rna_samples_all,
            cnv_samples_all=cnv_samples_all,
            mut_samples_all=mut_samples_all,
            immune_samples_all=immune_samples_all,
            meth_samples_all=meth_samples_all,
            params=params,
            pancancer_rows=pancancer_rows,
        )
        logger.info("[%s] done (%.1fs)", cancer, time.perf_counter() - t_cancer)

    # Pan-cancer summary
    pancancer = pd.DataFrame(pancancer_rows)
    if not pancancer.empty:
        write_tsv(pancancer, out_dir / "pancancer" / "tables" / "tp53_cox_summary.tsv")
        _forest_plot_tp53(pancancer, out_dir / "pancancer" / "figures" / "tp53_forest.png")
        logger.info(
            "wrote pan-cancer summary: rows=%d (%s)",
            pancancer.shape[0],
            out_dir / "pancancer",
        )
    else:
        logger.warning("no pan-cancer summary rows produced")


def _forest_plot_tp53(pancancer: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pancancer.copy()
    df = df.dropna(subset=["endpoint", "cancer", "hr", "hr_ci_lower", "hr_ci_upper"])
    if df.empty:
        return

    sns.set_theme(style="whitegrid")
    endpoints = ["OS", "PFI", "DSS"]
    df["endpoint"] = pd.Categorical(df["endpoint"], categories=endpoints, ordered=True)
    df = df.sort_values(["endpoint", "hr"], ascending=[True, False])

    fig, axes = plt.subplots(1, 3, figsize=(12, max(4, 0.35 * df["cancer"].nunique() + 1)), sharey=True)
    for ax, ep in zip(axes, endpoints):
        sub = df[df["endpoint"] == ep].copy()
        if sub.empty:
            ax.axis("off")
            continue
        sub = sub.sort_values("hr", ascending=True)
        hr = sub["hr"].to_numpy(dtype=float)
        hr_lo = sub["hr_ci_lower"].to_numpy(dtype=float)
        hr_hi = sub["hr_ci_upper"].to_numpy(dtype=float)
        y = np.arange(len(sub))
        ax.errorbar(
            hr,
            y,
            xerr=[hr - hr_lo, hr_hi - hr],
            fmt="o",
            color="black",
            ecolor="gray",
            capsize=2,
        )
        ax.axvline(1.0, color="red", linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["cancer"].tolist())
        ax.set_xlabel("Hazard ratio (TP53 mut vs wt)")
        ax.set_title(ep)
    savefig(fig, out_path)


def _run_one_cancer(
    *,
    client: XenaClient,
    surv: pd.DataFrame,
    immune_subtype_all: pd.DataFrame,
    stemness_meth_all: pd.DataFrame,
    cancer: str,
    out_base: Path,
    expr_genes: list[str],
    cnv_genes: list[str],
    mut_genes: list[str],
    immune_sigs: list[str],
    rna_samples_all: set[str],
    cnv_samples_all: set[str],
    mut_samples_all: set[str],
    immune_samples_all: set[str],
    meth_samples_all: set[str],
    params: PipelineParams,
    pancancer_rows: list[dict],
) -> None:
    logger = logging.getLogger(__name__)
    tables_dir = out_base / "tables"
    figs_dir = out_base / "figures"
    ensure_dir(tables_dir)
    ensure_dir(figs_dir)

    samples = cohort_samples(surv, cancer=cancer)
    if not samples:
        logger.warning("[%s] no samples from survival table; skipping", cancer)
        return
    logger.info("[%s] cohort samples (deduped primary): %d", cancer, len(samples))

    # Restrict to samples that have mutation-gene data (needed for TP53 status).
    samples = [s for s in samples if s in mut_samples_all]
    if not samples:
        logger.warning("[%s] no samples with mutation matrix coverage; skipping", cancer)
        return
    logger.info("[%s] samples with mutation coverage: %d", cancer, len(samples))

    # TP53 status from nonsilentGene matrix
    tp53_row = client.fetch_dense_matrix(
        XenaDatasets.MUT_GENE,
        samples=samples,
        probes=["TP53"],
        cache_key=f"{cancer}__TP53_status",
        probe_chunk_size=1,
        dtype="int8",
    )
    tp53_mut = tp53_binary_from_mut_gene(tp53_row)
    tp53_mut.name = "TP53_mut"
    counts = tp53_mut.value_counts(dropna=False).to_dict()
    logger.info(
        "[%s] TP53 status: WT=%d MUT=%d (rate=%.3f)",
        cancer,
        int(counts.get(0, 0)),
        int(counts.get(1, 0)),
        float(counts.get(1, 0)) / max(int(tp53_mut.shape[0]), 1),
    )

    clinical = surv.reindex(samples).copy()
    clinical = clinical.drop(columns=["sampleID"], errors="ignore")
    if "age_at_initial_pathologic_diagnosis" in clinical.columns:
        clinical["age_at_initial_pathologic_diagnosis"] = pd.to_numeric(
            clinical["age_at_initial_pathologic_diagnosis"], errors="coerce"
        )
    clinical["TP53_mut"] = tp53_mut.reindex(clinical.index).astype("Int64")
    clinical["stage_int"] = _readable_stage(clinical).astype("Int64")

    write_tsv(
        clinical.reset_index(names="sampleID"),
        tables_dir / "clinical_with_tp53.tsv",
    )
    logger.info("[%s] wrote clinical table: %s", cancer, tables_dir / "clinical_with_tp53.tsv")

    # Survival analyses for OS/PFI/DSS
    covariates: list[str] = []
    if "age_at_initial_pathologic_diagnosis" in clinical.columns:
        age_non_missing = int(clinical["age_at_initial_pathologic_diagnosis"].notna().sum())
        if age_non_missing >= max(20, int(0.5 * clinical.shape[0])):
            covariates.append("age_at_initial_pathologic_diagnosis")
    if "gender" in clinical.columns:
        covariates.append("gender")
    if "stage_int" in clinical.columns:
        stage_non_missing = int(clinical["stage_int"].notna().sum())
        stage_unique = int(clinical["stage_int"].dropna().nunique())
        if stage_unique >= 2 and stage_non_missing >= max(20, int(0.5 * clinical.shape[0])):
            covariates.append("stage_int")

    for endpoint in ["OS", "PFI", "DSS"]:
        time_col = f"{endpoint}.time"
        event_col = endpoint
        if time_col not in clinical.columns or event_col not in clinical.columns:
            logger.warning("[%s] missing endpoint columns for %s; skipping", cancer, endpoint)
            continue

        df_ep = clinical[[time_col, event_col, "TP53_mut", *covariates]].copy()
        df_ep[time_col] = pd.to_numeric(df_ep[time_col], errors="coerce")
        df_ep[event_col] = pd.to_numeric(df_ep[event_col], errors="coerce")
        df_ep["TP53_mut"] = pd.to_numeric(df_ep["TP53_mut"], errors="coerce")
        for c in covariates:
            if c in df_ep.columns and df_ep[c].dtype != "object":
                df_ep[c] = pd.to_numeric(df_ep[c], errors="coerce")

        df_ep = df_ep.dropna(subset=[time_col, event_col, "TP53_mut"])
        if df_ep.empty:
            logger.warning("[%s] no usable rows for %s; skipping", cancer, endpoint)
            continue
        n_ep = int(df_ep.shape[0])
        events_ep = int(df_ep[event_col].fillna(0).astype(int).sum())
        logger.info("[%s] %s: n=%d events=%d", cancer, endpoint, n_ep, events_ep)

        km_plot(
            df_ep,
            time_col=time_col,
            event_col=event_col,
            group_col="TP53_mut",
            title=f"{cancer} TP53 (0=WT,1=MUT) - {endpoint}",
            out_path=figs_dir / f"km_{endpoint}.png",
        )
        logger.info("[%s] wrote KM plot: %s", cancer, figs_dir / f"km_{endpoint}.png")

        summary = summarize_endpoint_by_group(df_ep, time_col=time_col, event_col=event_col, group_col="TP53_mut")
        summary["endpoint"] = endpoint
        write_tsv(summary, tables_dir / f"survival_summary_{endpoint}.tsv")

        cox = fit_cox_tp53(df_ep, time_col=time_col, event_col=event_col, covariates=covariates)
        if not cox.empty:
            write_tsv(cox, tables_dir / f"cox_{endpoint}.tsv")
            tp53_row = _extract_tp53_term(cox)
            if tp53_row is not None:
                hr = _safe_float(tp53_row.get("exp(coef)"))
                p = _safe_float(tp53_row.get("p"))
                lo = _safe_float(tp53_row.get("exp(coef) lower 95%"))
                hi = _safe_float(tp53_row.get("exp(coef) upper 95%"))
                logger.info(
                    "[%s] Cox %s (TP53 mut vs wt): HR=%s p=%s CI=[%s,%s]",
                    cancer,
                    endpoint,
                    f"{hr:.3f}" if hr is not None else "NA",
                    f"{p:.2e}" if p is not None else "NA",
                    f"{lo:.3f}" if lo is not None else "NA",
                    f"{hi:.3f}" if hi is not None else "NA",
                )
                try:
                    n_cox = int(tp53_row.get("n")) if pd.notna(tp53_row.get("n")) else int(df_ep.shape[0])
                except Exception:
                    n_cox = int(df_ep.shape[0])
                try:
                    events_cox = (
                        int(tp53_row.get("events"))
                        if pd.notna(tp53_row.get("events"))
                        else int(pd.to_numeric(df_ep[event_col], errors="coerce").fillna(0).astype(int).sum())
                    )
                except Exception:
                    events_cox = int(pd.to_numeric(df_ep[event_col], errors="coerce").fillna(0).astype(int).sum())
                pancancer_rows.append(
                    {
                        "cancer": cancer,
                        "endpoint": endpoint,
                        "coef": _safe_float(tp53_row.get("coef")),
                        "hr": _safe_float(tp53_row.get("exp(coef)")),
                        "p": _safe_float(tp53_row.get("p")),
                        "hr_ci_lower": _safe_float(tp53_row.get("exp(coef) lower 95%")),
                        "hr_ci_upper": _safe_float(tp53_row.get("exp(coef) upper 95%")),
                        "n": n_cox,
                        "events": events_cox,
                    }
                )
        else:
            logger.warning("[%s] Cox %s failed/empty (insufficient data?)", cancer, endpoint)

    # TP53 mutation types / hotspots
    tp53_sparse = client.fetch_tp53_sparse_mutations(samples=samples)
    tp53_sparse = summarize_tp53_sparse(tp53_sparse)
    if not tp53_sparse.empty:
        write_tsv(tp53_sparse, tables_dir / "tp53_mutations_sparse.tsv")
        logger.info("[%s] TP53 sparse mutations: %d rows", cancer, int(tp53_sparse.shape[0]))
        hotspots = top_tp53_hotspots(tp53_sparse, n=15)
        if not hotspots.empty:
            write_tsv(hotspots, tables_dir / "tp53_hotspots.tsv")
            top_barplot(
                hotspots,
                x_col="count",
                y_col="amino_acid",
                title=f"{cancer} TP53 hotspots (MC3)",
                out_path=figs_dir / "tp53_hotspots.png",
                n=15,
            )
            logger.info("[%s] wrote TP53 hotspot plot: %s", cancer, figs_dir / "tp53_hotspots.png")
    else:
        logger.info("[%s] TP53 sparse mutations: none", cancer)

    # Expression DE
    expr_samples = [s for s in samples if s in rna_samples_all]
    if len(expr_samples) >= 20 and tp53_mut.reindex(expr_samples).nunique() == 2:
        logger.info("[%s] RNA: samples=%d genes=%d", cancer, len(expr_samples), len(expr_genes))
        expr = client.fetch_dense_matrix(
            XenaDatasets.RNA,
            samples=expr_samples,
            probes=expr_genes,
            cache_key=f"{cancer}__expr",
            probe_chunk_size=500,
            dtype="float32",
        )
        de = differential_expression(expr, tp53_mut.reindex(expr_samples))
        if not de.empty:
            write_tsv(de, tables_dir / "de_expression_tp53.tsv")
            logger.info(
                "[%s] DE (RNA): tested=%d sig_fdr<0.05=%d",
                cancer,
                int(de.shape[0]),
                int((de["fdr"] < 0.05).sum()),
            )
            volcano_plot(
                de,
                effect_col="delta_mean",
                p_col="p",
                label_col="gene",
                title=f"{cancer} DE (TP53 mut vs wt)",
                out_path=figs_dir / "volcano_expression.png",
            )
            logger.info("[%s] wrote DE volcano: %s", cancer, figs_dir / "volcano_expression.png")
        # Optional: gene×TP53 interaction survival screen on expression
        if params.max_survival_genes and params.max_survival_genes > 0:
            var = expr.var(axis=1).sort_values(ascending=False)
            genes_screen = var.head(params.max_survival_genes).index.tolist()
            for endpoint in ["OS", "PFI", "DSS"]:
                time_col = f"{endpoint}.time"
                event_col = endpoint
                if time_col not in clinical.columns or event_col not in clinical.columns:
                    continue
                screen = cox_gene_tp53_interaction_screen(
                    expr=expr,
                    clinical=clinical,
                    time_col=time_col,
                    event_col=event_col,
                    tp53_col="TP53_mut",
                    genes=genes_screen,
                    penalizer=0.1,
                    min_events=20,
                )
                if not screen.empty:
                    write_tsv(screen, tables_dir / f"expr_tp53_interaction_{endpoint}.tsv")
    else:
        logger.info("[%s] RNA: insufficient coverage or only one TP53 group; skipping", cancer)

    # CNV DE + CNV burden
    cnv_samples = [s for s in samples if s in cnv_samples_all]
    if len(cnv_samples) >= 20 and tp53_mut.reindex(cnv_samples).nunique() == 2:
        logger.info("[%s] CNV: samples=%d genes=%d", cancer, len(cnv_samples), len(cnv_genes))
        cnv = client.fetch_dense_matrix(
            XenaDatasets.CNV_GENE,
            samples=cnv_samples,
            probes=cnv_genes,
            cache_key=f"{cancer}__cnv",
            probe_chunk_size=500,
            dtype="float32",
        )
        de_cnv = differential_cnv(cnv, tp53_mut.reindex(cnv_samples))
        if not de_cnv.empty:
            write_tsv(de_cnv, tables_dir / "de_cnv_tp53.tsv")
            logger.info(
                "[%s] DE (CNV): tested=%d sig_fdr<0.05=%d",
                cancer,
                int(de_cnv.shape[0]),
                int((de_cnv["fdr"] < 0.05).sum()),
            )
            volcano_plot(
                de_cnv,
                effect_col="delta_mean",
                p_col="p",
                label_col="gene",
                title=f"{cancer} CNV difference (TP53 mut vs wt)",
                out_path=figs_dir / "volcano_cnv.png",
            )
            logger.info("[%s] wrote CNV volcano: %s", cancer, figs_dir / "volcano_cnv.png")

        cnv_burden = cnv.abs().mean(axis=0)
        cnv_burden.name = "cnv_burden_mean_abs"
        write_tsv(
            pd.DataFrame(
                {
                    "sampleID": cnv_burden.index,
                    "cnv_burden_mean_abs": cnv_burden.values,
                    "TP53_mut": tp53_mut.reindex(cnv_burden.index).values,
                }
            ),
            tables_dir / "cnv_burden.tsv",
        )
    else:
        logger.info("[%s] CNV: insufficient coverage or only one TP53 group; skipping", cancer)

    # Mutation co-occurrence
    mut_samples = [s for s in samples if s in mut_samples_all]
    if len(mut_samples) >= 30 and tp53_mut.reindex(mut_samples).nunique() == 2:
        logger.info("[%s] MUT: samples=%d genes=%d", cancer, len(mut_samples), len(mut_genes))
        mut_gene = client.fetch_dense_matrix(
            XenaDatasets.MUT_GENE,
            samples=mut_samples,
            probes=mut_genes,
            cache_key=f"{cancer}__mut_gene",
            probe_chunk_size=600,
            dtype="int8",
        )
        co = comutation_against_tp53(mut_gene, tp53_mut.reindex(mut_samples))
        if not co.empty:
            write_tsv(co, tables_dir / "comutation_vs_tp53.tsv")
            logger.info(
                "[%s] co-mutation: tested=%d sig_fdr<0.05=%d",
                cancer,
                int(co.shape[0]),
                int((co["fdr"] < 0.05).sum()),
            )

        tmb = tmb_from_mut_gene(mut_gene)
        write_tsv(
            pd.DataFrame(
                {
                    "sampleID": tmb.index,
                    "mutated_genes_count": tmb.values,
                    "TP53_mut": tp53_mut.reindex(tmb.index).values,
                }
            ),
            tables_dir / "tmb_mutated_genes_count.tsv",
        )
    else:
        logger.info("[%s] MUT: insufficient samples or only one TP53 group; skipping", cancer)

    # Immune signatures + immune subtype
    immune_samples = [s for s in samples if s in immune_samples_all]
    if len(immune_samples) >= 20 and tp53_mut.reindex(immune_samples).nunique() == 2:
        logger.info("[%s] IMMUNE: samples=%d sigs=%d", cancer, len(immune_samples), len(immune_sigs))
        immune = client.fetch_dense_matrix(
            XenaDatasets.IMMUNE_SIGS,
            samples=immune_samples,
            probes=immune_sigs,
            cache_key=f"{cancer}__immune_sigs",
            probe_chunk_size=200,
            dtype="float32",
        )
        de_imm = differential_immune(immune, tp53_mut.reindex(immune_samples))
        if not de_imm.empty:
            write_tsv(de_imm, tables_dir / "de_immune_sigs_tp53.tsv")
            logger.info(
                "[%s] immune sig diff: tested=%d sig_fdr<0.05=%d",
                cancer,
                int(de_imm.shape[0]),
                int((de_imm["fdr"] < 0.05).sum()),
            )
            # Show top by absolute delta (among significant-ish)
            tmp = de_imm.copy()
            tmp["abs_delta"] = tmp["delta_mean"].abs()
            tmp = tmp.sort_values(["fdr", "abs_delta"], ascending=[True, False])
            top_barplot(
                tmp.head(20),
                x_col="abs_delta",
                y_col="signature",
                title=f"{cancer} immune signatures |Δ| (TP53 mut vs wt)",
                out_path=figs_dir / "immune_signatures_top.png",
                n=20,
            )
            logger.info("[%s] wrote immune top plot: %s", cancer, figs_dir / "immune_signatures_top.png")
    else:
        logger.info("[%s] IMMUNE: insufficient coverage or only one TP53 group; skipping", cancer)

    if not immune_subtype_all.empty and "Subtype_Immune_Model_Based" in immune_subtype_all.columns:
        subtype = immune_subtype_all["Subtype_Immune_Model_Based"].reindex(samples)
        ct = immune_subtype_crosstab(subtype, tp53_mut.reindex(samples))
        if not ct.empty:
            write_tsv(ct, tables_dir / "immune_subtype_crosstab.tsv")
            logger.info("[%s] wrote immune subtype crosstab: %s", cancer, tables_dir / "immune_subtype_crosstab.tsv")

    # Methylation-derived stemness
    if not stemness_meth_all.empty:
        cols = [c for c in stemness_meth_all.columns if c != "sampleID"]
        stem = stemness_meth_all[cols].reindex(samples)
        stem["TP53_mut"] = tp53_mut.reindex(stem.index).astype("Int64")
        stem = stem.dropna(subset=["TP53_mut"])
        if stem["TP53_mut"].nunique() == 2:
            rows: list[dict] = []
            for c in cols:
                a = pd.to_numeric(stem.loc[stem["TP53_mut"] == 1, c], errors="coerce")
                b = pd.to_numeric(stem.loc[stem["TP53_mut"] == 0, c], errors="coerce")
                if a.dropna().shape[0] < 5 or b.dropna().shape[0] < 5:
                    continue
                t, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
                rows.append({"score": c, "mean_mut": float(a.mean()), "mean_wt": float(b.mean()), "p": float(p)})
            if rows:
                out = pd.DataFrame(rows)
                # Proper FDR
                from tcga_tp53.stats import fdr_bh

                out["fdr"] = fdr_bh(out["p"].to_numpy())
                write_tsv(out.sort_values(["fdr", "p"]), tables_dir / "stemness_methylation_scores.tsv")
                logger.info(
                    "[%s] stemness(meth) scores: tested=%d sig_fdr<0.05=%d",
                    cancer,
                    int(out.shape[0]),
                    int((out["fdr"] < 0.05).sum()),
                )

    # Targeted methylation (TP53 + top DE genes)
    if params.methylation_top_genes > 0:
        try:
            de_path = tables_dir / "de_expression_tp53.tsv"
            if de_path.exists():
                de = pd.read_csv(de_path, sep="\t")
                top_genes = de.dropna(subset=["gene"]).head(params.methylation_top_genes)["gene"].astype(str).tolist()
            else:
                top_genes = []
        except Exception:
            top_genes = []

        genes_for_meth = ["TP53", *top_genes]
        genes_for_meth = sorted(set([g for g in genes_for_meth if g and g != "nan"]))

        if genes_for_meth:
            meth_samples = [s for s in samples if s in meth_samples_all]
            if len(meth_samples) >= 20 and tp53_mut.reindex(meth_samples).nunique() == 2:
                logger.info(
                    "[%s] METH450: samples=%d genes(target)=%d",
                    cancer,
                    len(meth_samples),
                    len(genes_for_meth),
                )
                key = f"{cancer}__meth450__{stable_hash(genes_for_meth)}"
                probe_by_sample, probe_meta = client.fetch_methylation_probes_for_genes(
                    samples=meth_samples, genes=genes_for_meth, cache_key=key
                )
                gene_meth = methylation_gene_averages(probe_by_sample, probe_meta, genes=genes_for_meth)
                if not gene_meth.empty:
                    dm = differential_methylation(gene_meth, tp53_mut.reindex(gene_meth.columns))
                    if not dm.empty:
                        write_tsv(dm, tables_dir / "dm_methylation_tp53.tsv")
                        logger.info(
                            "[%s] DM (methylation): tested=%d sig_fdr<0.05=%d",
                            cancer,
                            int(dm.shape[0]),
                            int((dm["fdr"] < 0.05).sum()),
                        )
                        volcano_plot(
                            dm,
                            effect_col="delta_beta",
                            p_col="p",
                            label_col="gene",
                            title=f"{cancer} methylation (beta) difference (TP53 mut vs wt)",
                            out_path=figs_dir / "volcano_methylation.png",
                        )
                        logger.info("[%s] wrote methylation volcano: %s", cancer, figs_dir / "volcano_methylation.png")
            else:
                logger.info("[%s] METH450: insufficient coverage or only one TP53 group; skipping", cancer)
    else:
        warnings.warn("methylation_top_genes=0: skipping methylation probe fetch", stacklevel=2)
