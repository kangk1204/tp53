from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

from tcga_tp53.stats import differential_ttest, fdr_bh


def tp53_binary_from_mut_gene(mut_gene: pd.DataFrame) -> pd.Series:
    """
    mut_gene: rows=genes, cols=samples (0/1)
    Returns: Series indexed by sample, values {0,1}
    """
    if "TP53" not in mut_gene.index:
        raise ValueError("TP53 not found in mutation gene matrix")
    s = mut_gene.loc["TP53"]
    return s.astype(int)


def classify_tp53_effect(effect: str | float | None) -> str | None:
    if effect is None or (isinstance(effect, float) and np.isnan(effect)):
        return None
    e = str(effect)
    # Common MC3 effects: Missense_Mutation, Nonsense_Mutation, Frame_Shift_Ins, Frame_Shift_Del,
    # Splice_Site, In_Frame_Ins, In_Frame_Del, Nonstop_Mutation, Translation_Start_Site, etc.
    trunc = {
        "Nonsense_Mutation",
        "Frame_Shift_Ins",
        "Frame_Shift_Del",
        "Nonstop_Mutation",
        "Translation_Start_Site",
    }
    if e in trunc:
        return "truncating"
    if e == "Splice_Site":
        return "splice"
    if e == "Missense_Mutation":
        return "missense"
    if e in {"In_Frame_Ins", "In_Frame_Del"}:
        return "inframe"
    return "other"


def summarize_tp53_sparse(tp53_sparse: pd.DataFrame) -> pd.DataFrame:
    if tp53_sparse.empty:
        return pd.DataFrame()
    df = tp53_sparse.copy()
    if "effect" in df.columns:
        df["class"] = df["effect"].apply(classify_tp53_effect)
    if "amino-acid" in df.columns:
        df["aa"] = df["amino-acid"].astype(str)
    return df


def differential_expression(expr: pd.DataFrame, tp53_mut: pd.Series) -> pd.DataFrame:
    out = differential_ttest(expr, group=tp53_mut.astype(bool), group1_label="mut", group0_label="wt")
    if out.empty:
        return out
    # Expr is already in log2-ish scale (EB++ adjusted). Use diff as "logFC-like".
    out = out.rename(columns={"feature": "gene", "diff_mean": "delta_mean"})
    return out


def differential_cnv(cnv: pd.DataFrame, tp53_mut: pd.Series) -> pd.DataFrame:
    out = differential_ttest(cnv, group=tp53_mut.astype(bool), group1_label="mut", group0_label="wt")
    if out.empty:
        return out
    out = out.rename(columns={"feature": "gene", "diff_mean": "delta_mean"})
    return out


def differential_immune(immune: pd.DataFrame, tp53_mut: pd.Series) -> pd.DataFrame:
    out = differential_ttest(immune, group=tp53_mut.astype(bool), group1_label="mut", group0_label="wt")
    if out.empty:
        return out
    out = out.rename(columns={"feature": "signature", "diff_mean": "delta_mean"})
    return out


def comutation_against_tp53(mut_gene: pd.DataFrame, tp53_mut: pd.Series) -> pd.DataFrame:
    """
    Fisher exact test per gene for association with TP53 mutation status.
    mut_gene: rows=genes, cols=samples (0/1)
    tp53_mut: Series indexed by sample with {0,1}
    """
    s = tp53_mut.reindex(mut_gene.columns).dropna().astype(bool)
    mut_gene = mut_gene.loc[:, s.index]

    tp = s.values
    not_tp = ~tp
    tp_n = int(tp.sum())
    not_tp_n = int(not_tp.sum())

    m = mut_gene.to_numpy(dtype=bool)
    a = m[:, tp].sum(axis=1)  # gene mutated & TP53-mut
    c = m[:, not_tp].sum(axis=1)  # gene mutated & TP53-wt
    b = tp_n - a
    d = not_tp_n - c

    genes = mut_gene.index.to_list()
    pvals = np.ones(len(genes), dtype=float)
    ors = np.full(len(genes), np.nan, dtype=float)
    for i in range(len(genes)):
        table = [[int(a[i]), int(b[i])], [int(c[i]), int(d[i])]]
        try:
            or_, p = stats.fisher_exact(table, alternative="two-sided")
        except Exception:
            or_, p = np.nan, 1.0
        ors[i] = or_
        pvals[i] = p

    out = pd.DataFrame(
        {
            "gene": genes,
            "a_tp53mut_gene_mut": a,
            "b_tp53mut_gene_wt": b,
            "c_tp53wt_gene_mut": c,
            "d_tp53wt_gene_wt": d,
            "odds_ratio": ors,
            "p": pvals,
        }
    )
    out["fdr"] = fdr_bh(out["p"].to_numpy())
    out["mut_freq_tp53mut"] = out["a_tp53mut_gene_mut"] / max(tp_n, 1)
    out["mut_freq_tp53wt"] = out["c_tp53wt_gene_mut"] / max(not_tp_n, 1)
    out["delta_freq"] = out["mut_freq_tp53mut"] - out["mut_freq_tp53wt"]
    out = out[out["gene"] != "TP53"].sort_values(["fdr", "p"]).reset_index(drop=True)
    return out


def tmb_from_mut_gene(mut_gene: pd.DataFrame) -> pd.Series:
    # Approx: number of mutated genes (not mutation count). Works without MAF parsing.
    return mut_gene.sum(axis=0).astype(int)


def immune_subtype_crosstab(immune_subtype: pd.Series, tp53_mut: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"immune_subtype": immune_subtype, "TP53_mut": tp53_mut}).dropna()
    if df.empty:
        return pd.DataFrame()
    ct = pd.crosstab(df["immune_subtype"], df["TP53_mut"])
    ct = ct.reset_index().rename(columns={0: "TP53_wt", 1: "TP53_mut"})
    return ct


def top_tp53_hotspots(tp53_sparse: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if tp53_sparse.empty or "amino-acid" not in tp53_sparse.columns:
        return pd.DataFrame()
    aas = tp53_sparse["amino-acid"].dropna().astype(str)
    # Normalize e.g. "p.R175H" -> "R175H"
    aas = aas.str.replace("^p\\.", "", regex=True)
    counts = Counter(aas)
    rows = [{"amino_acid": k, "count": v} for k, v in counts.most_common(n)]
    return pd.DataFrame(rows)


def methylation_gene_averages(
    probe_by_sample: pd.DataFrame,
    probe_meta: pd.DataFrame,
    *,
    genes: list[str],
) -> pd.DataFrame:
    """
    Convert probe-level methylation (beta) to gene-level by averaging probes mapped to each gene.

    probe_by_sample: rows=probe name, cols=samples
    probe_meta: columns include ["name","genes"] where genes is list[str]
    """
    if probe_by_sample.empty or probe_meta.empty:
        return pd.DataFrame()

    meta = probe_meta[["name", "genes"]].copy()
    meta = meta[meta["name"].isin(probe_by_sample.index)]

    def _as_list(x: object) -> list[str]:
        if isinstance(x, list):
            return [str(i) for i in x]
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return []
        return [str(x)]

    meta["genes"] = meta["genes"].apply(_as_list)
    meta = meta.explode("genes")
    meta = meta[meta["genes"].isin(set(genes))]
    if meta.empty:
        return pd.DataFrame()

    # group probes by gene
    gene_to_probes = meta.groupby("genes")["name"].apply(list)
    out: dict[str, np.ndarray] = {}
    for gene, probes in gene_to_probes.items():
        vals = probe_by_sample.loc[probes].to_numpy(dtype=float)
        out[gene] = np.nanmean(vals, axis=0)
    gene_by_sample = pd.DataFrame(out, index=probe_by_sample.columns).T
    return gene_by_sample.astype("float32")


def differential_methylation(gene_meth: pd.DataFrame, tp53_mut: pd.Series) -> pd.DataFrame:
    out = differential_ttest(gene_meth, group=tp53_mut.astype(bool), group1_label="mut", group0_label="wt")
    if out.empty:
        return out
    out = out.rename(columns={"feature": "gene", "diff_mean": "delta_beta"})
    return out
