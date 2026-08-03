"""Two self-forecasts per model: unprimed vs identity-primed.

Elicits both variants of the self-forecast on the same tasks, conditions and
sample slots, so the only difference between them is whether the model was first
asked -- and answered -- "do you know what model you are?" in the same context:

* ``self``        -- ``prompts/self_forecast.txt``, exactly as the main run sends it.
* ``self_primed`` -- the identity question, the model's own answer, then that
  identical prompt as a third turn.

What this script does *not* do is say which variant is more accurate. Accuracy
needs the rollout outcomes, which live in the run's ``data_*`` directory and are
not committed; without them the only honest comparison is between the two sets of
predictions -- how far priming moves a model's beliefs about itself, and in which
direction. Point ``--outcomes-from`` at a populated run directory to also get the
realised losses.

    python compare_self_forecasts.py --dry-run --tasks 2       # no API calls
    python compare_self_forecasts.py --tasks 5 --samples 3     # real, ~$
    python compare_self_forecasts.py --models z-ai/glm-5 --tasks 30
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_EXP_DIR = Path(__file__).parent.resolve()
_ROOT = _EXP_DIR.parents[1]
for _p in (str(_ROOT), str(_EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402
import dataset  # noqa: E402
import forecasting  # noqa: E402
from src.types import Forecast  # noqa: E402

#: The two variants this script contrasts, in report order.
VARIANTS = ("self", "self_primed")


# --------------------------------------------------------------------------
# Task selection
# --------------------------------------------------------------------------
def load_tasks(n_tasks: int, *, dry_run: bool, data_dir: Path | None = None) -> list:
    """The tasks to forecast, preferring a run's cached ``tasks.json``.

    Reusing the cache matters: comparing variants across *different* task samples
    would confound the priming effect with task difficulty. Falls back to the
    seeded selection, which reproduces the same set given the same seed.
    """
    if data_dir is not None:
        cached = data_dir / "tasks.json"
        if cached.exists():
            rows = json.loads(cached.read_text(encoding="utf-8"))
            return [dataset.Task.from_dict(d) for d in rows][:n_tasks]
    selected = dataset.mock_tasks(config.SEED) if dry_run else dataset.select_tasks(config.SEED)
    return selected[:n_tasks]


# --------------------------------------------------------------------------
# Elicitation
# --------------------------------------------------------------------------
def sample_path(out_dir: Path, variant: str, slug: str, task_id: str, sample_idx: int) -> Path:
    """``<out_dir>/<variant>/<slug>/<task_id>/s<i>.json`` -- one file per sample.

    Mirrors the main run's sub-cell layout so a killed sweep resumes by re-running
    only the samples missing from disk.
    """
    return out_dir / variant / slug / task_id / f"s{sample_idx}.json"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)                       # atomic: a crash never leaves a half file


def elicit_cell(llm, spec: config.ModelSpec, task, variant: str, sample_idx: int,
                conditions, *, fallback_template=None) -> dict:
    """One sample of one variant for one (model, task) cell -- all conditions."""
    out = {}
    for cond in conditions:
        fc = forecasting.elicit_forecast(
            llm, spec, task, cond, variant,
            fallback_template=fallback_template, sample_idx=sample_idx,
        )
        out[cond] = fc.to_dict()
    return out


def plan(models, tasks, conditions, samples: int) -> list[tuple]:
    """Every (variant, model, task, sample) unit this run would cover."""
    return [
        (variant, spec, task, idx)
        for variant in VARIANTS
        for spec in models
        for task in tasks
        for idx in range(samples)
    ]


def call_budget(n_units: int, n_conditions: int) -> dict:
    """Forecast calls implied by a plan -- primed units cost an extra turn each.

    Half the units are primed, and each of those spends one call on the identity
    question per condition on top of the forecast itself.
    """
    per_variant = n_units // 2
    plain = per_variant * n_conditions
    primed = per_variant * n_conditions * 2
    return {"units": n_units, "plain_calls": plain, "primed_calls": primed,
            "total_calls": plain + primed}


def run_sweep(llm, models, tasks, conditions, samples: int, out_dir: Path, *,
              max_workers: int = 8, fallback_template=None) -> int:
    """Elicit every missing unit; returns how many were newly elicited."""
    units = plan(models, tasks, conditions, samples)
    todo = [
        (v, s, t, i, sample_path(out_dir, v, s.slug, t.instance_id, i))
        for (v, s, t, i) in units
        if not sample_path(out_dir, v, s.slug, t.instance_id, i).exists()
    ]

    def _one(unit):
        variant, spec, task, idx, path = unit
        _write(path, elicit_cell(llm, spec, task, variant, idx, conditions,
                                 fallback_template=fallback_template))

    if todo:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_one, todo))
    return len(todo)


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
def load_cells(out_dir: Path, models, tasks, conditions, samples: int) -> dict:
    """Read every written sample back as ``[variant][model][task][condition] -> [Forecast]``."""
    cells: dict = {v: {} for v in VARIANTS}
    for variant in VARIANTS:
        for spec in models:
            per_task: dict = {}
            for task in tasks:
                per_cond: dict = {c: [] for c in conditions}
                for idx in range(samples):
                    path = sample_path(out_dir, variant, spec.slug, task.instance_id, idx)
                    if not path.exists():
                        continue
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    for cond in conditions:
                        if cond in raw:
                            per_cond[cond].append(Forecast.from_dict(raw[cond]))
                per_task[task.instance_id] = per_cond
            cells[variant][spec.openrouter_id] = per_task
    return cells


def _mean_predictions(samples: list[Forecast]) -> dict:
    """Mean prediction per dimension across non-broken samples (run.py Option A)."""
    usable = [f for f in samples if not f.broken and f.predictions]
    if not usable:
        return {}
    return {
        dim: statistics.fmean(vals)
        for dim in config.DIMENSION_NAMES
        if (vals := [f.predictions[dim] for f in usable if dim in f.predictions])
    }


def compare(cells: dict, models, conditions) -> dict:
    """Per-model, per-dimension shift from priming.

    Reports the signed mean shift, not its magnitude: whether priming makes a
    model *more* or *less* optimistic about itself is the interesting part, and an
    absolute value would hide it. Dimensions are reported in their own units, so
    the numbers are comparable within a dimension and never across.
    """
    report: dict = {}
    for spec in models:
        mid = spec.openrouter_id
        per_dim: dict = {d: [] for d in config.DIMENSION_NAMES}
        n_pairs = 0
        for task_id, per_cond in cells["self"].get(mid, {}).items():
            primed_task = cells["self_primed"].get(mid, {}).get(task_id, {})
            for cond in conditions:
                plain = _mean_predictions(per_cond.get(cond, []))
                primed = _mean_predictions(primed_task.get(cond, []))
                if not plain or not primed:
                    continue
                n_pairs += 1
                for dim in config.DIMENSION_NAMES:
                    if dim in plain and dim in primed:
                        per_dim[dim].append(primed[dim] - plain[dim])
        report[mid] = {
            "name": spec.name,
            "n_pairs": n_pairs,
            "shift": {d: round(statistics.fmean(v), 4) for d, v in per_dim.items() if v},
            "identity_replies": _identity_summary(cells["self_primed"].get(mid, {}), conditions),
        }
    return report


def _identity_summary(per_task: dict, conditions) -> dict:
    """Whether the priming turns actually elicited a self-identification."""
    replies = [
        f.identity_reply
        for per_cond in per_task.values()
        for cond in conditions
        for f in per_cond.get(cond, [])
        if f.identity_reply is not None
    ]
    return {
        "n": len(replies),
        "n_empty": sum(1 for r in replies if not r.strip()),
        "example": next((" ".join(r.split())[:120] for r in replies if r.strip()), None),
    }


def format_report(report: dict, *, top_k: int = 3) -> str:
    """One block per model: the dimensions priming moved most."""
    lines = ["Self-forecast: unprimed vs identity-primed", ""]
    for mid, row in report.items():
        lines.append(f"{row['name']}  ({mid})   pairs={row['n_pairs']}")
        ident = row["identity_replies"]
        if ident["n"]:
            lines.append(f"    primed with: {ident['example']!r}"
                         + (f"  [{ident['n_empty']} empty]" if ident["n_empty"] else ""))
        if not row["shift"]:
            lines.append("    no comparable pairs")
        else:
            ranked = sorted(row["shift"].items(), key=lambda kv: -abs(kv[1]))[:top_k]
            for dim, delta in ranked:
                lines.append(f"    {dim:<26} {delta:+.4f}   (primed − unprimed)")
        lines.append("")
    lines.append("Shifts are in each dimension's own units; sign is the direction "
                 "priming moved the forecast.")
    lines.append("These are prediction shifts, not accuracy changes — scoring needs "
                 "the rollout outcomes.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Elicit unprimed and identity-primed self-forecasts and diff them.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="OpenRouter ids (default: all of config.MODELS).")
    parser.add_argument("--tasks", type=int, default=5,
                        help="How many tasks to forecast (default 5; the full run uses 30).")
    parser.add_argument("--samples", type=int, default=config.N_FORECAST_SAMPLES,
                        help=f"Samples per cell (default {config.N_FORECAST_SAMPLES}).")
    parser.add_argument("--conditions", nargs="+", default=list(config.CONDITIONS),
                        choices=list(config.CONDITIONS))
    parser.add_argument("--out-dir", default=str(_EXP_DIR / "data_self_primed"),
                        help="Where samples are cached; re-running resumes from here.")
    parser.add_argument("--tasks-from", default=None,
                        help="Run data dir to reuse tasks.json from, so the task set matches.")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Mocked client, no API calls.")
    args = parser.parse_args(argv)

    unknown = [m for m in (args.models or []) if m not in config.MODEL_BY_ID]
    if unknown:
        parser.error(f"not in config.MODELS: {', '.join(unknown)}")

    models = ([config.MODEL_BY_ID[m] for m in args.models] if args.models
              else list(config.MODELS))
    tasks = load_tasks(args.tasks, dry_run=args.dry_run,
                       data_dir=Path(args.tasks_from) if args.tasks_from else None)
    if not tasks:
        parser.error("no tasks selected")

    out_dir = Path(args.out_dir)
    budget = call_budget(len(plan(models, tasks, args.conditions, args.samples)),
                         len(args.conditions))
    print(f"{len(models)} models x {len(tasks)} tasks x {len(args.conditions)} conditions "
          f"x {args.samples} samples x 2 variants")
    print(f"budget: {budget['total_calls']} forecast calls "
          f"({budget['primed_calls']} of them primed, which is 2 calls per sample)")

    if args.dry_run:
        from run import MockLLMClient
        llm = MockLLMClient()
    else:
        from src.llm_client import LLMClient
        llm = LLMClient()

    elicited = run_sweep(llm, models, tasks, args.conditions, args.samples, out_dir,
                         max_workers=args.max_workers)
    print(f"elicited {elicited} new samples (rest cached in {out_dir})\n")

    cells = load_cells(out_dir, models, tasks, args.conditions, args.samples)
    report = compare(cells, models, args.conditions)
    (out_dir / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(format_report(report))
    print(f"\nwrote {out_dir / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
