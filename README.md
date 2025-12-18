# TCGA TP53 Pan-Cancer Pipeline (Python)

Download **public, processed TCGA PanCanAtlas** data from **UCSC Xena**, automatically pick the **top N cancers with the worst overall survival (OS)**, and run a **TP53-centered multi-omics analysis** (survival, mutation, RNA, CNV, immune, methylation). The pipeline writes **publication-ready tables and figures** to `results/`.

## What this pipeline does

For each selected cancer type:
- Builds a cohort (primary tumor only, one sample per patient)
- Derives **TP53 mutation status** from the MC3 nonsilent gene matrix
- Runs **KM + log-rank** and **Cox proportional hazards** (OS/PFI/DSS)
- Runs group comparisons (TP53-mut vs TP53-wt):
  - Differential expression (RNA)
  - Differential CNV (gene-level)
  - Immune signature differences + immune subtype cross-tab
  - Co-mutation enrichment vs TP53
  - Targeted methylation (TP53 + top DE genes; probe → gene averages)
- Produces a **pan-cancer forest plot** summary for TP53 across endpoints

## Methods (high level)

- Cohort: TCGA primary tumor samples only (sample type `01`), one sample per patient
- Survival:
  - Kaplan–Meier curves + log-rank test
  - Cox PH model for `TP53_mut` (optionally adjusted for age, gender, stage when available)
- Group comparisons (TP53-mut vs TP53-wt):
  - Welch’s t-test (unequal variance) for RNA/CNV/immune/methylation
  - Benjamini–Hochberg FDR correction (`fdr`)
- Co-mutation enrichment vs TP53:
  - Fisher’s exact test (two-sided) + BH FDR (`fdr`)
- Targeted methylation:
  - Probe-level beta values are averaged to gene-level using the probe map

## Requirements

- Python **3.10+**
- Internet access (downloads via UCSC Xena API)

## Getting the code

```bash
git clone https://github.com/kangk1204/tp53.git
cd tp53
```

## Installation (beginner-friendly)

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Quick import check (optional)
python -c "import tcga_tp53; print('ok')"
```

If you see errors like `numpy.dtype size changed`, you are likely mixing a system Python with incompatible binary packages. Activate `.venv` and re-run using the venv Python.

## Quickstart: run the full pipeline

```bash
python scripts/run_tcga_tp53_pipeline.py \
  --out results \
  --cache-dir cache \
  --top-n 10 \
  --min-samples 80 \
  --min-events 30
```

If `results/` already exists from a previous run, use `--overwrite` or choose a new `--out`.

## Command-line parameters

See the full built-in help:

```bash
python scripts/run_tcga_tp53_pipeline.py --help
```

### Core

- `--out` (default: `results`): output directory (tables/figures/logs are written here)
- `--cache-dir` (default: `cache`): cache directory for downloaded/processed Xena matrices
- `--top-n` (default: `10`): number of cancer types selected for analysis (selected by worst OS median)
- `--min-samples` (default: `80`): minimum cohort size per cancer (after primary-tumor filtering + 1 sample/patient)
- `--min-events` (default: `30`): minimum number of **OS events** per cancer for inclusion

### Optional analyses

- `--methylation-top-genes` (default: `200`): targeted methylation uses `TP53 + top N DE genes` per cancer
  - Set `0` to skip targeted methylation.
- `--max-survival-genes` (default: `0`): enable a (slow) gene×TP53 survival interaction screen on expression
  - `0` disables (recommended for a first run).

### Performance / reproducibility

- `--threads` (default: `4`): parallel workers for the interaction screen (only used if `--max-survival-genes > 0`)
- `--seed` (default: `0`): random seed for reproducibility (the main pipeline is mostly deterministic)
- `--overwrite`: allow writing into a non-empty `--out` directory

### Logging / UI

- `--log-level` (default: `INFO`): `DEBUG|INFO|WARNING|ERROR`
- `--log-file` (default: `<out>/run.log`): write logs to a file
- `--no-progress`: disable tqdm progress bars

Tip: to make sure you are using the venv Python, run `./.venv/bin/python scripts/run_tcga_tp53_pipeline.py ...`.

## Examples

Small/fast smoke run (less output, faster):

```bash
python scripts/run_tcga_tp53_pipeline.py \
  --out results_smoke \
  --cache-dir cache \
  --top-n 3 \
  --min-samples 50 \
  --min-events 15 \
  --methylation-top-genes 0
```

Enable the expression interaction screen (can be slow):

```bash
python scripts/run_tcga_tp53_pipeline.py \
  --out results \
  --cache-dir cache \
  --max-survival-genes 2000 \
  --threads 8
```

## Caching (important)

- Downloads and processed matrices are cached under `--cache-dir` (default: `cache/`).
- The first run can be slow; later runs are much faster.
- To force a fresh download, delete the cache directory (e.g., `rm -rf cache/`).

## Outputs (where to look first)

### Cohort selection
- `results/selected_cohorts.tsv`
  - The “worst” cancers are selected by **smallest OS median (days)**, after applying `--min-samples` and `--min-events`.

### Per-cancer outputs
For each cancer `CANCER`:
- `results/<CANCER>/tables/clinical_with_tp53.tsv`: clinical table + `TP53_mut` (0/1) + parsed stage
- `results/<CANCER>/tables/survival_summary_OS.tsv` (and PFI/DSS): group sizes, events, KM median
- `results/<CANCER>/tables/cox_OS.tsv` (and PFI/DSS): Cox model summary (includes TP53 term)
- `results/<CANCER>/figures/km_OS.png` (and PFI/DSS): Kaplan–Meier plot with log-rank p-value

Multi-omics tables/figures (if data coverage is sufficient):
- RNA: `results/<CANCER>/tables/de_expression_tp53.tsv`, `results/<CANCER>/figures/volcano_expression.png`
- CNV: `results/<CANCER>/tables/de_cnv_tp53.tsv`, `results/<CANCER>/tables/cnv_burden.tsv`, `results/<CANCER>/figures/volcano_cnv.png`
- Mutation: `results/<CANCER>/tables/comutation_vs_tp53.tsv`, `results/<CANCER>/tables/tmb_mutated_genes_count.tsv`
- TP53 hotspots: `results/<CANCER>/tables/tp53_hotspots.tsv`, `results/<CANCER>/figures/tp53_hotspots.png`
- Immune: `results/<CANCER>/tables/de_immune_sigs_tp53.tsv`, `results/<CANCER>/figures/immune_signatures_top.png`, `results/<CANCER>/tables/immune_subtype_crosstab.tsv`
- Methylation: `results/<CANCER>/tables/dm_methylation_tp53.tsv`, `results/<CANCER>/figures/volcano_methylation.png`

Some analyses are automatically skipped when:
- the cohort is too small, or
- only one TP53 group exists (all TP53-wt or all TP53-mut), or
- the selected dataset has limited sample coverage.

### Pan-cancer summary
- `results/pancancer/tables/tp53_cox_summary.tsv`: TP53 HR/p-value by cancer and endpoint
- `results/pancancer/figures/tp53_forest.png`: forest plot by endpoint

## How to interpret the results (simple guide)

### TP53 status
`TP53_mut` is derived from the MC3 **nonsilent gene mutation** matrix:
- `0` = TP53 wild-type (no nonsilent TP53 mutation in MC3)
- `1` = TP53 mutated

### Kaplan–Meier (KM) plots
`results/<CANCER>/figures/km_<ENDPOINT>.png`
- The title shows the **log-rank p-value**
- If the TP53-mut curve drops faster, it suggests **worse survival** in TP53-mut samples

### Cox proportional hazards
`results/<CANCER>/tables/cox_<ENDPOINT>.tsv`
- Focus on the row where `term == TP53_mut`
- Interpret `exp(coef)` as hazard ratio (HR):
  - **HR > 1**: TP53-mut has higher hazard (worse prognosis)
  - **HR < 1**: TP53-mut has lower hazard (better prognosis)
- The model may include covariates (when available): age, gender, stage

### Differential expression / CNV / immune signatures
`de_*_tp53.tsv`
- `delta_mean` (or `delta_beta` for methylation) is **mean(mut) − mean(wt)**
- Use `fdr` for multiple-testing–adjusted significance
- Volcano plots show effect size (x) vs `-log10(p)` (y)

### Co-mutation vs TP53
`results/<CANCER>/tables/comutation_vs_tp53.tsv`
- `odds_ratio > 1` means the gene is more often mutated in TP53-mut samples
- `delta_freq` is the frequency difference between TP53-mut and TP53-wt

### “TMB” output
`results/<CANCER>/tables/tmb_mutated_genes_count.tsv`
- This is **mutated genes count** (a simple proxy), not mutations-per-megabase.

## Data sources and references

- UCSC Xena PanCanAtlas hub: `https://pancanatlas.xenahubs.net`
- TCGA PanCanAtlas / MC3 are produced by the TCGA Research Network and related consortia.

## License

MIT License. See `LICENSE`.
