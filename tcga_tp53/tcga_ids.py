from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TcgaId:
    sample: str

    @property
    def patient(self) -> str:
        parts = self.sample.split("-")
        if len(parts) < 3:
            return self.sample
        return "-".join(parts[:3])

    @property
    def sample_type_code(self) -> str | None:
        parts = self.sample.split("-")
        if len(parts) < 4:
            return None
        # Most PanCanAtlas matrices use ...-01, ...-11, etc.
        m = re.match(r"^(\d\d)", parts[3])
        return m.group(1) if m else None

    def is_primary_tumor(self) -> bool:
        return self.sample_type_code == "01"


def tcga_patient_id(sample_id: str) -> str:
    return TcgaId(sample_id).patient


def is_primary_tumor(sample_id: str) -> bool:
    return TcgaId(sample_id).is_primary_tumor()


def dedupe_by_patient(sample_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    kept: list[str] = []
    for sid in sample_ids:
        pid = tcga_patient_id(sid)
        if pid in seen:
            continue
        seen.add(pid)
        kept.append(sid)
    return kept
