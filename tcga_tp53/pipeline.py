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
from tcga_tp53.effect_sizes import chi2_independence, cliffs_delta, mannwhitney_u_p
from tcga_tp53.enrichment import GseaConfig, rank_from_de_tstat, run_prerank_gsea
from tcga_tp53.io import ensure_dir, write_tsv
from tcga_tp53.meta_analysis import log_hr_se_from_ci, random_effects_meta
from tcga_tp53.plots import km_plot, km_plot_multi, savefig, top_barplot, violin_box_plot, volcano_plot
from tcga_tp53.select_cohorts import cohort_samples, select_worst_os_cancers
from tcga_tp53.stats import fdr_bh
from tcga_tp53.survival import (
    fit_cox_categorical_group,
    fit_cox_stratified,
    fit_cox_tp53,
    km_logrank_p,
    parse_stage_to_int,
    summarize_endpoint_by_group,
)
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


def _tp53_palette() -> dict[int, str]:
    # Colorblind-friendly: WT=blue, MUT=red
    return {0: "#1f77b4", 1: "#d62728"}


def _tp53_group_labels() -> dict[int, str]:
    return {0: "TP53 WT", 1: "TP53 mutated"}


def _tp53_class_palette() -> dict[str, str]:
    return {
        "WT": "#1f77b4",
        "truncating": "#d62728",
        "splice": "#9467bd",
        "missense": "#ff7f0e",
        "inframe": "#2ca02c",
        "other": "#7f7f7f",
    }


def _pc1_scores(features_by_sample: pd.DataFrame) -> pd.Series | None:
    """
    features_by_sample: rows=features, cols=samples
    Returns PC1 score per sample (Series indexed by sample).
    """
    if features_by_sample.empty:
        return None
    x = features_by_sample.to_numpy(dtype=float).T  # samples x features
    if x.shape[0] < 5 or x.shape[1] < 2:
        return None

    # Fill NaNs with feature means
    col_means = np.nanmean(x, axis=0)
    inds = np.where(~np.isfinite(x))
    if inds[0].size:
        x[inds] = np.take(col_means, inds[1])

    # Z-score features
    x = x - np.mean(x, axis=0)
    x = x / (np.std(x, axis=0, ddof=0) + 1e-8)

    try:
        u, s, _vt = np.linalg.svd(x, full_matrices=False)
    except Exception:
        return None
    pc1 = u[:, 0] * s[0]
    pc1 = pd.Series(pc1, index=features_by_sample.columns, name="pc1").astype("float32")

    # Fix sign for reproducibility (align with mean signal).
    mean_signal = pd.Series(np.mean(x, axis=1), index=features_by_sample.columns)
    if pc1.corr(mean_signal) < 0:
        pc1 = -pc1
    return pc1


def _tp53_class_by_sample(tp53_sparse: pd.DataFrame, tp53_mut: pd.Series) -> pd.Series:
    """
    Returns TP53 class per sample (WT or mutation class from MC3 sparse records).
    Priority for multi-hit samples: truncating > splice > missense > inframe > other.
    """
    out = pd.Series(index=tp53_mut.index, dtype="object")
    out.loc[tp53_mut.astype("Int64") == 0] = "WT"
    out.loc[tp53_mut.astype("Int64") == 1] = "other"

    if tp53_sparse.empty or "sampleID" not in tp53_sparse.columns:
        return out
    if "class" not in tp53_sparse.columns:
        return out

    priority = {"truncating": 0, "splice": 1, "missense": 2, "inframe": 3, "other": 4}
    df = tp53_sparse[["sampleID", "class"]].copy().dropna()
    df["class"] = df["class"].astype(str)
    df = df[df["class"].isin(priority)]
    if df.empty:
        return out

    def pick_class(classes: pd.Series) -> str:
        vals = [c for c in classes.astype(str).tolist() if c in priority]
        if not vals:
            return "other"
        return sorted(vals, key=lambda c: priority[c])[0]

    picked = df.groupby("sampleID")["class"].apply(pick_class)
    mutated_samples = out[out != "WT"].index
    out.loc[mutated_samples] = picked.reindex(mutated_samples).fillna("other")
    return out


def _compare_binary_groups(values: pd.Series, group: pd.Series) -> dict[str, object] | None:
    """
    Compare values between group==1 and group==0 using Mann–Whitney U + Cliff's delta.
    """
    g = pd.to_numeric(group, errors="coerce").reindex(values.index)
    x = pd.to_numeric(values, errors="coerce")
    df = pd.DataFrame({"value": x, "group": g}).dropna()
    if df.empty or df["group"].nunique() != 2:
        return None
    a = df.loc[df["group"] == 1, "value"].to_numpy(dtype=float)
    b = df.loc[df["group"] == 0, "value"].to_numpy(dtype=float)
    if a.size < 5 or b.size < 5:
        return None
    p = mannwhitney_u_p(a, b)
    d = cliffs_delta(a, b)
    return {
        "n_mut": int(a.size),
        "n_wt": int(b.size),
        "mean_mut": float(np.mean(a)),
        "mean_wt": float(np.mean(b)),
        "median_mut": float(np.median(a)),
        "median_wt": float(np.median(b)),
        "diff_median": float(np.median(a) - np.median(b)),
        "p_mwu": p,
        "cliffs_delta": d,
    }


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
    run_tp53_class: bool
    run_extended_cox: bool
    run_burden_plots: bool
    run_gsea: bool
    gsea_gene_sets: list[str]
    gsea_permutations: int
    gsea_min_size: int
    gsea_max_size: int
    run_pancancer_meta: bool
    run_stratified_cox: bool


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
    run_all: bool = False,
    run_tp53_class: bool = True,
    run_extended_cox: bool = True,
    run_burden_plots: bool = True,
    run_gsea: bool = False,
    gsea_gene_sets: list[str] | None = None,
    gsea_permutations: int = 200,
    gsea_min_size: int = 10,
    gsea_max_size: int = 500,
    run_pancancer_meta: bool = True,
    run_stratified_cox: bool = True,
) -> None:
    logger = logging.getLogger(__name__)
    if out_dir.exists():
        if out_dir.is_file():
            raise FileExistsError(f"--out must be a directory, but got an existing file: {out_dir}")
        if any(out_dir.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {out_dir} (use --overwrite or choose a new --out)"
            )
    np.random.seed(seed)
    if run_all:
        run_tp53_class = True
        run_extended_cox = True
        run_burden_plots = True
        run_gsea = True
        run_pancancer_meta = True
        run_stratified_cox = True
    if gsea_gene_sets is None:
        gsea_gene_sets = ["MSigDB_Hallmark_2020", "Reactome_2022"]
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
        run_tp53_class=run_tp53_class,
        run_extended_cox=run_extended_cox,
        run_burden_plots=run_burden_plots,
        run_gsea=run_gsea,
        gsea_gene_sets=gsea_gene_sets,
        gsea_permutations=gsea_permutations,
        gsea_min_size=gsea_min_size,
        gsea_max_size=gsea_max_size,
        run_pancancer_meta=run_pancancer_meta,
        run_stratified_cox=run_stratified_cox,
    )
    ensure_dir(out_dir)

    logger.info(
        "starting pipeline: out=%s cache=%s top_n=%d min_samples=%d min_events=%d methylation_top_genes=%d max_survival_genes=%d threads=%d seed=%d overwrite=%s tp53_class=%s extended_cox=%s burden_plots=%s gsea=%s stratified_cox=%s",
        out_dir,
        cache_dir,
        top_n,
        min_samples,
        min_events,
        methylation_top_genes,
        max_survival_genes,
        threads,
        seed,
        overwrite,
        run_tp53_class,
        run_extended_cox,
        run_burden_plots,
        run_gsea,
        run_stratified_cox,
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
    pancancer_endpoint_dfs: dict[str, list[pd.DataFrame]] = {ep: [] for ep in ["OS", "PFI", "DSS"]}
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
            pancancer_endpoint_dfs=pancancer_endpoint_dfs,
        )
        logger.info("[%s] done (%.1fs)", cancer, time.perf_counter() - t_cancer)

    # Pan-cancer summary
    pancancer = pd.DataFrame(pancancer_rows)
    if not pancancer.empty:
        write_tsv(pancancer, out_dir / "pancancer" / "tables" / "tp53_cox_summary.tsv")
        meta = pd.DataFrame()
        if params.run_pancancer_meta:
            meta = _tp53_meta_analysis(pancancer)
            if not meta.empty:
                write_tsv(meta, out_dir / "pancancer" / "tables" / "tp53_cox_meta_random_effects.tsv")

        _forest_plot_tp53(
            pancancer,
            meta=meta if not meta.empty else None,
            out_path=out_dir / "pancancer" / "figures" / "tp53_forest.png",
        )

        if params.run_stratified_cox:
            for ep, parts in pancancer_endpoint_dfs.items():
                if not parts:
                    continue
                df_all = pd.concat(parts, axis=0, ignore_index=True)
                strat = fit_cox_stratified(
                    df_all,
                    time_col="time",
                    event_col="event",
                    covariates=["TP53_mut"],
                    strata_col="cancer",
                    penalizer=0.1,
                )
                if not strat.empty:
                    write_tsv(strat, out_dir / "pancancer" / "tables" / f"tp53_stratified_cox_{ep}.tsv")
        logger.info(
            "wrote pan-cancer summary: rows=%d (%s)",
            pancancer.shape[0],
            out_dir / "pancancer",
        )
    else:
        logger.warning("no pan-cancer summary rows produced")


def _tp53_meta_analysis(pancancer: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for ep, sub in pancancer.groupby("endpoint"):
        sub = sub.dropna(subset=["hr", "hr_ci_lower", "hr_ci_upper"])
        if sub.empty:
            continue
        vals: list[tuple[float, float]] = []
        for _, r in sub.iterrows():
            se = log_hr_se_from_ci(float(r["hr"]), float(r["hr_ci_lower"]), float(r["hr_ci_upper"]))
            if se is None:
                continue
            vals.append(se)
        if len(vals) < 2:
            continue
        log_hrs = np.array([v[0] for v in vals], dtype=float)
        ses = np.array([v[1] for v in vals], dtype=float)
        meta = random_effects_meta(log_hrs, ses, endpoint=str(ep))
        if meta is None:
            continue
        rows.append(
            {
                "endpoint": str(ep),
                "k": meta.k,
                "hr": meta.effect,
                "hr_ci_lower": meta.effect_ci_lower,
                "hr_ci_upper": meta.effect_ci_upper,
                "p": meta.p,
                "tau2": meta.tau2,
                "I2": meta.i2,
                "Q": meta.q,
                "Q_p": meta.q_p,
            }
        )
    return pd.DataFrame(rows)


def _forest_plot_tp53(pancancer: pd.DataFrame, *, meta: pd.DataFrame | None, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pancancer.copy()
    df = df.dropna(subset=["endpoint", "cancer", "hr", "hr_ci_lower", "hr_ci_upper"])
    if df.empty:
        return

    sns.set_theme(style="whitegrid", context="paper")
    endpoints = ["OS", "PFI", "DSS"]
    df["endpoint"] = pd.Categorical(df["endpoint"], categories=endpoints, ordered=True)
    df = df.sort_values(["endpoint", "hr"], ascending=[True, True])

    fig, axes = plt.subplots(1, 3, figsize=(12.5, max(4, 0.35 * df["cancer"].nunique() + 2)), sharey=True)
    for ax, ep in zip(axes, endpoints):
        sub = df[df["endpoint"] == ep].copy()
        if meta is not None and not meta.empty:
            m = meta[meta["endpoint"] == ep]
            if not m.empty:
                r = m.iloc[0].to_dict()
                sub = pd.concat(
                    [
                        sub,
                        pd.DataFrame(
                            [
                                {
                                    "cancer": f"Meta (RE)  I2={float(r.get('I2', float('nan'))):.1f}%",
                                    "endpoint": ep,
                                    "hr": float(r["hr"]),
                                    "hr_ci_lower": float(r["hr_ci_lower"]),
                                    "hr_ci_upper": float(r["hr_ci_upper"]),
                                }
                            ]
                        ),
                    ],
                    axis=0,
                    ignore_index=True,
                )
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
        ax.set_xscale("log")
        ax.set_xlabel("Hazard ratio (TP53 mutated vs WT)")
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
    pancancer_endpoint_dfs: dict[str, list[pd.DataFrame]],
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

    tp53_sparse = client.fetch_tp53_sparse_mutations(samples=samples)
    tp53_sparse = summarize_tp53_sparse(tp53_sparse)
    if params.run_tp53_class:
        clinical["TP53_class"] = _tp53_class_by_sample(tp53_sparse, tp53_mut).reindex(clinical.index).astype(str)

    write_tsv(
        clinical.reset_index(names="sampleID"),
        tables_dir / "clinical_with_tp53.tsv",
    )
    logger.info("[%s] wrote clinical table: %s", cancer, tables_dir / "clinical_with_tp53.tsv")

    metric_comparisons: list[dict] = []

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

    survival_overview_rows: list[dict] = []
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

        # Store for pan-cancer stratified Cox.
        df_pan = df_ep[[time_col, event_col, "TP53_mut"]].copy()
        df_pan = df_pan.rename(columns={time_col: "time", event_col: "event"})
        df_pan["cancer"] = cancer
        df_pan["sampleID"] = df_pan.index.astype(str)
        pancancer_endpoint_dfs.setdefault(endpoint, []).append(df_pan.reset_index(drop=True))

        logrank_p = km_logrank_p(df_ep[time_col], df_ep[event_col], df_ep["TP53_mut"])
        km_plot(
            df_ep,
            time_col=time_col,
            event_col=event_col,
            group_col="TP53_mut",
            title=f"{cancer} | TP53 WT vs mutated | {endpoint}",
            out_path=figs_dir / f"km_{endpoint}.png",
            group_labels=_tp53_group_labels(),
            palette=_tp53_palette(),
        )
        logger.info("[%s] wrote KM plot: %s", cancer, figs_dir / f"km_{endpoint}.png")

        summary = summarize_endpoint_by_group(df_ep, time_col=time_col, event_col=event_col, group_col="TP53_mut")
        summary["endpoint"] = endpoint
        write_tsv(summary, tables_dir / f"survival_summary_{endpoint}.tsv")

        cox = fit_cox_tp53(df_ep, time_col=time_col, event_col=event_col, covariates=covariates)
        row = {
            "endpoint": endpoint,
            "n": n_ep,
            "events": events_ep,
            "logrank_p": logrank_p,
            "cox_hr": None,
            "cox_p": None,
            "cox_hr_ci_lower": None,
            "cox_hr_ci_upper": None,
            "cox_n": None,
            "cox_events": None,
            "tp53_ph_p": None,
        }
        if not cox.empty:
            write_tsv(cox, tables_dir / f"cox_{endpoint}.tsv")
            tp53_row = _extract_tp53_term(cox)
            if tp53_row is not None:
                hr = _safe_float(tp53_row.get("exp(coef)"))
                p = _safe_float(tp53_row.get("p"))
                lo = _safe_float(tp53_row.get("exp(coef) lower 95%"))
                hi = _safe_float(tp53_row.get("exp(coef) upper 95%"))
                ph_p = _safe_float(tp53_row.get("ph_p"))
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
                row.update(
                    {
                        "cox_hr": hr,
                        "cox_p": p,
                        "cox_hr_ci_lower": lo,
                        "cox_hr_ci_upper": hi,
                        "cox_n": n_cox,
                        "cox_events": events_cox,
                        "tp53_ph_p": ph_p,
                    }
                )
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
        survival_overview_rows.append(row)

    if survival_overview_rows:
        surv_overview = pd.DataFrame(survival_overview_rows)
        surv_overview["logrank_fdr"] = fdr_bh(surv_overview["logrank_p"].fillna(1.0).to_numpy(dtype=float))
        surv_overview["cox_fdr"] = fdr_bh(surv_overview["cox_p"].fillna(1.0).to_numpy(dtype=float))
        write_tsv(surv_overview, tables_dir / "survival_overview_tp53.tsv")

    # TP53 class stratification (WT + mutation types)
    if params.run_tp53_class and "TP53_class" in clinical.columns:
        class_order = ["WT", "truncating", "splice", "missense", "inframe", "other"]
        present = [c for c in class_order if c in set(clinical["TP53_class"].astype(str).unique())]
        if len(present) >= 2:
            counts = (
                clinical["TP53_class"]
                .astype(str)
                .value_counts(dropna=False)
                .rename_axis("TP53_class")
                .reset_index(name="n")
            )
            write_tsv(counts, tables_dir / "tp53_class_counts.tsv")

            for endpoint in ["OS", "PFI", "DSS"]:
                time_col = f"{endpoint}.time"
                event_col = endpoint
                if time_col not in clinical.columns or event_col not in clinical.columns:
                    continue
                df_ep = clinical[[time_col, event_col, "TP53_class", *covariates]].copy()
                df_ep[time_col] = pd.to_numeric(df_ep[time_col], errors="coerce")
                df_ep[event_col] = pd.to_numeric(df_ep[event_col], errors="coerce")
                df_ep = df_ep.dropna(subset=[time_col, event_col, "TP53_class"])
                if df_ep.empty or df_ep["TP53_class"].nunique() < 2:
                    continue

                # Plot KM by class
                km_plot_multi(
                    df_ep,
                    time_col=time_col,
                    event_col=event_col,
                    group_col="TP53_class",
                    title=f"{cancer} | TP53 classes | {endpoint}",
                    out_path=figs_dir / f"km_tp53_class_{endpoint}.png",
                    order=present,
                    palette=_tp53_class_palette(),
                )

                # Multivariate log-rank p-value
                try:
                    from lifelines.statistics import multivariate_logrank_test

                    p_lr = float(multivariate_logrank_test(df_ep[time_col], df_ep["TP53_class"], df_ep[event_col]).p_value)
                except Exception:
                    p_lr = float("nan")

                summ = summarize_endpoint_by_group(
                    df_ep, time_col=time_col, event_col=event_col, group_col="TP53_class"
                )
                summ["endpoint"] = endpoint
                summ["logrank_p"] = p_lr
                write_tsv(summ, tables_dir / f"survival_summary_tp53_class_{endpoint}.tsv")

                cox_cls = fit_cox_categorical_group(
                    df_ep,
                    time_col=time_col,
                    event_col=event_col,
                    group_col="TP53_class",
                    covariates=covariates,
                    categories=present,
                    penalizer=0.1,
                )
                if not cox_cls.empty:
                    write_tsv(cox_cls, tables_dir / f"cox_tp53_class_{endpoint}.tsv")

    # TP53 mutation types / hotspots
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

            if params.run_gsea:
                cfg = GseaConfig(
                    gene_sets=list(params.gsea_gene_sets),
                    permutations=int(params.gsea_permutations),
                    min_size=int(params.gsea_min_size),
                    max_size=int(params.gsea_max_size),
                    seed=int(params.seed),
                )
                rank = rank_from_de_tstat(de)
                written = run_prerank_gsea(
                    rank,
                    table_dir=tables_dir / "gsea",
                    fig_dir=figs_dir / "gsea",
                    prefix="expression_tp53",
                    cfg=cfg,
                    overwrite=params.overwrite,
                )
                if written:
                    logger.info("[%s] GSEA results: %d files", cancer, len(written))
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
                    n_jobs=max(1, int(params.threads)),
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
        clinical["cnv_burden_mean_abs"] = cnv_burden.reindex(clinical.index).astype("float32")
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

        comp = _compare_binary_groups(cnv_burden, tp53_mut.reindex(cnv_burden.index))
        if comp is not None:
            metric_comparisons.append({"metric": "cnv_burden_mean_abs", **comp})
        if params.run_burden_plots:
            violin_box_plot(
                pd.DataFrame(
                    {
                        "TP53": tp53_mut.reindex(cnv_burden.index).map(_tp53_group_labels()),
                        "cnv_burden_mean_abs": cnv_burden.values,
                    }
                ),
                x_col="TP53",
                y_col="cnv_burden_mean_abs",
                title=f"{cancer} | CNV burden (mean |CNV|) by TP53",
                out_path=figs_dir / "cnv_burden_tp53.png",
                order=["TP53 WT", "TP53 mutated"],
                ylabel="Mean |CNV|",
            )
            if params.run_tp53_class and "TP53_class" in clinical.columns:
                cls = clinical["TP53_class"].reindex(cnv_burden.index).astype(str)
                df_cls = pd.DataFrame({"TP53_class": cls.values, "cnv_burden_mean_abs": cnv_burden.values})
                order_cls = [c for c in ["WT", "truncating", "splice", "missense", "inframe", "other"] if c in set(cls)]
                if len(order_cls) >= 2:
                    violin_box_plot(
                        df_cls,
                        x_col="TP53_class",
                        y_col="cnv_burden_mean_abs",
                        title=f"{cancer} | CNV burden by TP53 class",
                        out_path=figs_dir / "cnv_burden_tp53_class.png",
                        order=order_cls,
                        palette=_tp53_class_palette(),
                        ylabel="Mean |CNV|",
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
        clinical["mutated_genes_count"] = tmb.reindex(clinical.index).astype("float32")
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

        comp = _compare_binary_groups(tmb, tp53_mut.reindex(tmb.index))
        if comp is not None:
            metric_comparisons.append({"metric": "mutated_genes_count", **comp})
        if params.run_burden_plots:
            violin_box_plot(
                pd.DataFrame(
                    {
                        "TP53": tp53_mut.reindex(tmb.index).map(_tp53_group_labels()),
                        "mutated_genes_count": tmb.values,
                    }
                ),
                x_col="TP53",
                y_col="mutated_genes_count",
                title=f"{cancer} | Mutated genes count by TP53",
                out_path=figs_dir / "tmb_mutated_genes_count_tp53.png",
                order=["TP53 WT", "TP53 mutated"],
                ylabel="Mutated genes count (proxy)",
            )
            if params.run_tp53_class and "TP53_class" in clinical.columns:
                cls = clinical["TP53_class"].reindex(tmb.index).astype(str)
                df_cls = pd.DataFrame({"TP53_class": cls.values, "mutated_genes_count": tmb.values})
                order_cls = [c for c in ["WT", "truncating", "splice", "missense", "inframe", "other"] if c in set(cls)]
                if len(order_cls) >= 2:
                    violin_box_plot(
                        df_cls,
                        x_col="TP53_class",
                        y_col="mutated_genes_count",
                        title=f"{cancer} | Mutated genes count by TP53 class",
                        out_path=figs_dir / "tmb_mutated_genes_count_tp53_class.png",
                        order=order_cls,
                        palette=_tp53_class_palette(),
                        ylabel="Mutated genes count (proxy)",
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

        immune_pc1 = _pc1_scores(immune)
        if immune_pc1 is not None:
            clinical["immune_pc1"] = immune_pc1.reindex(clinical.index).astype("float32")
            comp = _compare_binary_groups(immune_pc1, tp53_mut.reindex(immune_pc1.index))
            if comp is not None:
                metric_comparisons.append({"metric": "immune_pc1", **comp})
            if params.run_burden_plots:
                violin_box_plot(
                    pd.DataFrame(
                        {
                            "TP53": tp53_mut.reindex(immune_pc1.index).map(_tp53_group_labels()),
                            "immune_pc1": immune_pc1.values,
                        }
                    ),
                    x_col="TP53",
                    y_col="immune_pc1",
                    title=f"{cancer} | Immune signatures PC1 by TP53",
                    out_path=figs_dir / "immune_pc1_tp53.png",
                    order=["TP53 WT", "TP53 mutated"],
                    ylabel="PC1 score",
                )
    else:
        logger.info("[%s] IMMUNE: insufficient coverage or only one TP53 group; skipping", cancer)

    if not immune_subtype_all.empty and "Subtype_Immune_Model_Based" in immune_subtype_all.columns:
        subtype = immune_subtype_all["Subtype_Immune_Model_Based"].reindex(samples)
        ct = immune_subtype_crosstab(subtype, tp53_mut.reindex(samples))
        if not ct.empty:
            write_tsv(ct, tables_dir / "immune_subtype_crosstab.tsv")
            logger.info("[%s] wrote immune subtype crosstab: %s", cancer, tables_dir / "immune_subtype_crosstab.tsv")

        try:
            tab = pd.crosstab(subtype, tp53_mut.reindex(samples)).astype(int)
            chi = chi2_independence(tab.to_numpy())
            if chi is not None:
                out = pd.DataFrame(
                    [
                        {
                            "chi2": chi.chi2,
                            "p": chi.p,
                            "dof": chi.dof,
                            "n": chi.n,
                            "cramers_v": chi.cramers_v,
                        }
                    ]
                )
                write_tsv(out, tables_dir / "immune_subtype_association.tsv")

            if params.run_burden_plots and not tab.empty and tab.shape[1] == 2:
                # Proportions stacked bar (TP53 WT vs mutated)
                try:
                    import matplotlib.pyplot as plt

                    prop = tab.div(tab.sum(axis=0), axis=1)
                    fig, ax = plt.subplots(figsize=(7.0, 4.8))
                    bottom = np.zeros(prop.shape[1], dtype=float)
                    cols = list(prop.columns)
                    labels: list[str] = []
                    for c in cols:
                        try:
                            labels.append(_tp53_group_labels().get(int(c), str(c)))
                        except Exception:
                            labels.append(str(c))
                    for idx in prop.index.astype(str):
                        vals = prop.loc[idx].to_numpy(dtype=float)
                        ax.bar(labels, vals, bottom=bottom, label=idx)
                        bottom = bottom + vals
                    ax.set_ylabel("Proportion")
                    ax.set_title(f"{cancer} | Immune subtype composition by TP53")
                    ax.legend(title="Immune subtype", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
                    savefig(fig, figs_dir / "immune_subtype_stacked.png", dpi=300)
                except Exception:
                    pass
        except Exception:
            pass

    # Methylation-derived stemness
    if not stemness_meth_all.empty:
        cols = [c for c in stemness_meth_all.columns if c != "sampleID"]
        stem = stemness_meth_all[cols].reindex(samples)
        stem_pc1 = _pc1_scores(stem.T)
        if stem_pc1 is not None:
            clinical["stemness_meth_pc1"] = stem_pc1.reindex(clinical.index).astype("float32")
            comp = _compare_binary_groups(stem_pc1, tp53_mut.reindex(stem_pc1.index))
            if comp is not None:
                metric_comparisons.append({"metric": "stemness_meth_pc1", **comp})
            if params.run_burden_plots:
                violin_box_plot(
                    pd.DataFrame(
                        {
                            "TP53": tp53_mut.reindex(stem_pc1.index).map(_tp53_group_labels()),
                            "stemness_meth_pc1": stem_pc1.values,
                        }
                    ),
                    x_col="TP53",
                    y_col="stemness_meth_pc1",
                    title=f"{cancer} | Stemness (DNAmeth) PC1 by TP53",
                    out_path=figs_dir / "stemness_meth_pc1_tp53.png",
                    order=["TP53 WT", "TP53 mutated"],
                    ylabel="PC1 score",
                )
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

    # Per-cancer metric comparisons (TP53 mutated vs WT)
    if metric_comparisons:
        mc = pd.DataFrame(metric_comparisons)
        mc["p_mwu"] = pd.to_numeric(mc["p_mwu"], errors="coerce")
        mc["fdr_mwu"] = fdr_bh(mc["p_mwu"].fillna(1.0).to_numpy(dtype=float))
        mc = mc.sort_values(["fdr_mwu", "p_mwu", "metric"], ascending=[True, True, True])
        write_tsv(mc, tables_dir / "tp53_mut_metric_comparisons.tsv")

    # Final per-cancer clinical table (includes derived omics covariates, if available).
    write_tsv(clinical.reset_index(names="sampleID"), tables_dir / "clinical_with_tp53_omics.tsv")

    # Extended Cox models (TP53 + additional covariates derived from other modalities)
    if params.run_extended_cox:
        extra_covs: list[str] = []
        for c in ["mutated_genes_count", "cnv_burden_mean_abs", "immune_pc1", "stemness_meth_pc1"]:
            if c not in clinical.columns:
                continue
            non_missing = int(pd.to_numeric(clinical[c], errors="coerce").notna().sum())
            if non_missing >= max(20, int(0.5 * clinical.shape[0])) and clinical[c].dropna().nunique() >= 2:
                extra_covs.append(c)

        if extra_covs:
            ext_rows: list[dict] = []
            for endpoint in ["OS", "PFI", "DSS"]:
                time_col = f"{endpoint}.time"
                event_col = endpoint
                if time_col not in clinical.columns or event_col not in clinical.columns:
                    continue
                df_ep = clinical[[time_col, event_col, "TP53_mut", *covariates, *extra_covs]].copy()
                df_ep[time_col] = pd.to_numeric(df_ep[time_col], errors="coerce")
                df_ep[event_col] = pd.to_numeric(df_ep[event_col], errors="coerce")
                df_ep["TP53_mut"] = pd.to_numeric(df_ep["TP53_mut"], errors="coerce")
                for c in [*covariates, *extra_covs]:
                    if c in df_ep.columns and df_ep[c].dtype != "object":
                        df_ep[c] = pd.to_numeric(df_ep[c], errors="coerce")
                df_ep = df_ep.dropna(subset=[time_col, event_col, "TP53_mut"])
                if df_ep.empty:
                    continue

                cox_ext = fit_cox_tp53(
                    df_ep,
                    time_col=time_col,
                    event_col=event_col,
                    covariates=[*covariates, *extra_covs],
                    penalizer=0.1,
                )
                if cox_ext.empty:
                    continue
                write_tsv(cox_ext, tables_dir / f"cox_extended_{endpoint}.tsv")
                tp53_row = _extract_tp53_term(cox_ext)
                if tp53_row is None:
                    continue
                ext_rows.append(
                    {
                        "endpoint": endpoint,
                        "covariates": ",".join(extra_covs),
                        "hr": _safe_float(tp53_row.get("exp(coef)")),
                        "p": _safe_float(tp53_row.get("p")),
                        "hr_ci_lower": _safe_float(tp53_row.get("exp(coef) lower 95%")),
                        "hr_ci_upper": _safe_float(tp53_row.get("exp(coef) upper 95%")),
                        "n": _safe_float(tp53_row.get("n")),
                        "events": _safe_float(tp53_row.get("events")),
                        "tp53_ph_p": _safe_float(tp53_row.get("ph_p")),
                    }
                )

            if ext_rows:
                ext = pd.DataFrame(ext_rows)
                ext["p"] = pd.to_numeric(ext["p"], errors="coerce")
                ext["fdr"] = fdr_bh(ext["p"].fillna(1.0).to_numpy(dtype=float))
                write_tsv(ext.sort_values(["fdr", "p"]), tables_dir / "survival_overview_tp53_extended.tsv")
