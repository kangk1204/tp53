#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TCGA TP53 PanCanAtlas pipeline (Xena)")
    p.add_argument("--out", type=Path, default=Path("results"), help="Output directory")
    p.add_argument("--cache-dir", type=Path, default=Path("cache"), help="Cache directory")
    p.add_argument("--top-n", type=int, default=10, help="Number of worst-prognosis cancers")
    p.add_argument("--min-samples", type=int, default=80, help="Min samples per cancer")
    p.add_argument("--min-events", type=int, default=30, help="Min events (OS) per cancer")
    p.add_argument(
        "--max-survival-genes",
        type=int,
        default=0,
        help="Max genes for gene×TP53 survival interaction screens (0=skip; can be slow)",
    )
    p.add_argument(
        "--methylation-top-genes",
        type=int,
        default=200,
        help="Fetch methylation probes for TP53 + top DE genes (per cancer)",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Parallel jobs for per-cancer computations (not Xena calls)",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed (tie-breaks only)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console/file log level",
    )
    p.add_argument("--log-file", type=Path, default=None, help="Log file path (default: <out>/run.log)")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    return p


def main() -> None:
    args = build_parser().parse_args()
    from tcga_tp53.logging_utils import configure_logging

    configure_logging(out_dir=args.out, level=args.log_level, log_file=args.log_file)
    from tcga_tp53.pipeline import run_pipeline

    run_pipeline(
        out_dir=args.out,
        cache_dir=args.cache_dir,
        top_n=args.top_n,
        min_samples=args.min_samples,
        min_events=args.min_events,
        methylation_top_genes=args.methylation_top_genes,
        max_survival_genes=args.max_survival_genes,
        threads=args.threads,
        seed=args.seed,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
