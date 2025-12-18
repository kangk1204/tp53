from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(*, out_dir: Path, level: str = "INFO", log_file: Path | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if log_file is None:
        log_file = out_dir / "run.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers on re-entry.
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    logging.captureWarnings(True)

