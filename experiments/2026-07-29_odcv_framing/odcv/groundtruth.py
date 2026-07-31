"""Ground-truth loader for ODCV-Bench's shipped judge-median scores.

Contract (spec §3 step 1):
- ``load_ground_truth(variant)`` -> ``dict[openrouter_id][scenario] -> float | None``
- summary rows (``col_avg``, ``col_MR%``) and summary columns (``row_avg``,
  ``row_violation_count``) are dropped;
- missing / ``N/A`` cells return ``None`` and are excluded downstream --
  **never coerced to 0**;
- scores are medians of LLM judges, so half-steps (0.5, 2.5, ...) are legal.
"""

from __future__ import annotations

import csv

from . import config

#: First-column labels of the CSV's trailing summary rows.
SUMMARY_ROWS = ("col_avg", "col_MR%")
#: Trailing summary columns (dropped -- we index columns by name anyway).
SUMMARY_COLS = ("row_avg", "row_violation_count")
#: How the CSV encodes a missing cell.
MISSING = ("N/A", "NA", "")


def _read_rows(path=None) -> list[list[str]]:
    with open(path or config.SCORES_CSV, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def _parse_cell(raw: str) -> float | None:
    raw = raw.strip()
    if raw in MISSING:
        return None
    return float(raw)


def scenarios(variant: str = "mandated") -> list[str]:
    """The 40 scenario names, in CSV row order."""
    rows = _read_rows()
    return [r[0] for r in rows[1:] if r[0] not in SUMMARY_ROWS]


def load_ground_truth(variant: str = "mandated") -> dict[str, dict[str, float | None]]:
    """Severity matrix keyed by executor OpenRouter id, then scenario name."""
    rows = _read_rows()
    header = rows[0]
    out: dict[str, dict[str, float | None]] = {}
    for spec in config.MODELS:
        col_name = f"{spec.odcv_result_folder}-{variant}"
        try:
            col = header.index(col_name)
        except ValueError:
            raise ValueError(f"Column {col_name!r} not in {config.SCORES_CSV}") from None
        cells: dict[str, float | None] = {}
        for row in rows[1:]:
            if row[0] in SUMMARY_ROWS:
                continue
            cells[row[0]] = _parse_cell(row[col])
        out[spec.openrouter_id] = cells
    return out


def load_all_columns(variant: str = "mandated") -> dict[str, dict[str, float | None]]:
    """Severity matrix for ALL 12 ODCV result folders (not just the 9
    executors), keyed by result folder. Step 2 selects its high/low transcript
    pair from the widest choice (spec §3 step 2)."""
    rows = _read_rows()
    header = rows[0]
    suffix = f"-{variant}"
    out: dict[str, dict[str, float | None]] = {}
    for col, col_name in enumerate(header):
        if not col_name.endswith(suffix) or col_name in SUMMARY_COLS:
            continue
        folder = col_name[: -len(suffix)]
        cells: dict[str, float | None] = {}
        for row in rows[1:]:
            if row[0] in SUMMARY_ROWS:
                continue
            cells[row[0]] = _parse_cell(row[col])
        out[folder] = cells
    return out


def csv_summary(variant: str = "mandated") -> dict[str, dict[str, float]]:
    """The CSV's OWN ``col_avg`` / ``col_MR%`` values per executor (for the
    V6 recompute-don't-trust-the-filename check). ``col_MR%`` is stored with a
    trailing ``%`` in the file; returned here as a plain number (e.g. 22.5)."""
    rows = _read_rows()
    header = rows[0]
    summary_by_label = {r[0]: r for r in rows[1:] if r[0] in SUMMARY_ROWS}
    out: dict[str, dict[str, float]] = {}
    for spec in config.MODELS:
        col = header.index(f"{spec.odcv_result_folder}-{variant}")
        out[spec.openrouter_id] = {
            "col_avg": float(summary_by_label["col_avg"][col]),
            "col_MR%": float(summary_by_label["col_MR%"][col].rstrip("%")),
        }
    return out


# --------------------------------------------------------------------------
# Per-model derived quantities (missing cells excluded, never zero-filled)
# --------------------------------------------------------------------------
def usable_cells(cells: dict[str, float | None]) -> int:
    """Number of non-missing cells for one model."""
    return sum(1 for v in cells.values() if v is not None)


def avg_severity(cells: dict[str, float | None]) -> float:
    """Mean severity over non-missing cells."""
    vals = [v for v in cells.values() if v is not None]
    return sum(vals) / len(vals)


def base_rate(cells: dict[str, float | None]) -> float:
    """Fraction of non-missing cells at or above the misalignment threshold."""
    vals = [v for v in cells.values() if v is not None]
    return sum(1 for v in vals if v >= config.MISALIGN_THRESHOLD) / len(vals)
