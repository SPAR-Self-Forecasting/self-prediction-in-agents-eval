"""Judge-noise ceiling re-derivation (spec §3 step 6a, acceptance 12).

Method: for every (executor, scenario) cell of the chosen variant, hold out
each of the 4 shipped judges in turn and compare its score against the median
of the other judges' scores for the same cell. Two statistics:

- binary agreement: held-out judge and others' median fall on the same side of
  the misalignment threshold (>= 3);
- mean absolute difference on the 0-5 scale.

This bounds *scoring* noise only -- how reproducible a score is given the
transcript -- not the N=1 behavioral noise (spec is explicit about this).
The value computed here on the 9-executor subset replaces the spec's 92.8%
(measured on the original 10-executor set) everywhere downstream.
"""

from __future__ import annotations

import csv
import json
import statistics

from . import config, groundtruth

MISSING = ("N/A", "NA", "")


def _load_judge_matrix(path, variant: str) -> dict[str, dict[str, float]]:
    """One judge's scores: dict[result_folder][scenario] -> float (missing skipped).

    Rows are POSITIVELY filtered to the final CSV's 40 scenario names -- the
    per-judge files carry different summary rows (``col_avg``, ``col_na_count``)
    than the final median file, so a blacklist would let fake cells leak in.
    """
    scenario_names = set(groundtruth.scenarios(variant))
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    out: dict[str, dict[str, float]] = {}
    for spec in config.MODELS:
        col_name = f"{spec.odcv_result_folder}-{variant}"
        if col_name not in header:
            continue
        col = header.index(col_name)
        cells: dict[str, float] = {}
        for row in rows[1:]:
            if row[0] not in scenario_names:
                continue
            raw = row[col].strip()
            if raw in MISSING:
                continue
            cells[row[0]] = float(raw)
        out[spec.odcv_result_folder] = cells
    return out


def compute_ceiling(variant: str = "mandated") -> dict:
    """Hold-one-judge-out ceiling over the 9-executor subset."""
    judges = {
        name: _load_judge_matrix(path, variant)
        for name, path in config.PER_JUDGE_CSVS.items()
    }
    thr = config.MISALIGN_THRESHOLD

    per_judge: dict[str, dict] = {}
    all_agree: list[int] = []
    all_absdiff: list[float] = []

    for held_out in judges:
        agree: list[int] = []
        absdiff: list[float] = []
        for spec in config.MODELS:
            folder = spec.odcv_result_folder
            held_cells = judges[held_out].get(folder, {})
            for scenario, own in held_cells.items():
                others = [
                    judges[j][folder][scenario]
                    for j in judges
                    if j != held_out and scenario in judges[j].get(folder, {})
                ]
                if len(others) < 2:
                    continue  # a median of <2 other judges is not a panel
                med = statistics.median(others)
                agree.append(int((own >= thr) == (med >= thr)))
                absdiff.append(abs(own - med))
        per_judge[held_out] = {
            "n": len(agree),
            "binary_agreement": sum(agree) / len(agree),
            "mean_abs_diff": sum(absdiff) / len(absdiff),
        }
        all_agree.extend(agree)
        all_absdiff.extend(absdiff)

    return {
        "variant": variant,
        "n_executors": len(config.MODELS),
        "n_comparisons": len(all_agree),
        "binary_agreement": sum(all_agree) / len(all_agree),
        "mean_abs_diff": sum(all_absdiff) / len(all_absdiff),
        "per_judge": per_judge,
    }


def main() -> dict:
    result = compute_ceiling(config.VARIANT)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "judge_ceiling.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(
        f"judge-noise ceiling ({result['variant']}, {result['n_executors']} executors): "
        f"{result['binary_agreement']:.1%} binary agreement, "
        f"{result['mean_abs_diff']:.3f} mean |diff|, "
        f"n={result['n_comparisons']} -> {out_path}"
    )
    return result


if __name__ == "__main__":
    main()
