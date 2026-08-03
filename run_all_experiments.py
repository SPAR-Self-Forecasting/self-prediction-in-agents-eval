"""Root orchestrator for the self-prediction capabilities project.

Sequences the experiments with a tunable configuration. Experiment 1
(SWE-bench introspection) is driven end-to-end from here. Experiment 2
(ODCV framing gap) has shipped but is *not* orchestrated: it is a ladder of
individually-approved paid steps against a read-only external benchmark
checkout, so it prints its reproduction recipe rather than launching spend
from a batch entry point. Experiment 3 is not started.

Usage examples
--------------
    python run_all_experiments.py --experiments 1 --quick      # fast mocked smoke run
    python run_all_experiments.py --experiments 1              # full mocked dry-run
    python run_all_experiments.py --experiments 1 --real       # the real run (spends money)
    python run_all_experiments.py --experiments 2              # print exp2's recipe
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.resolve()
_EXP1_DIR = _ROOT / "experiments" / "2026-05-16_self_prediction"
_EXP2_DIR = _ROOT / "experiments" / "2026-07-29_odcv_framing"

#: Ordered stages of Experiment 2, as run on 2026-07-30 (see its REPORT.md §6).
#: Each is a separate module invocation with its own checkpoint and spend rung.
EXP2_STEPS: tuple[tuple[str, str], ...] = (
    ("odcv.judge_ceiling", "free"),
    ("odcv.extract_conflicts", "paid ~$0.8"),
    ("odcv.discovery_check", "paid ~$4"),
    ("odcv.fidelity_gate", "free gate"),
    ("odcv.elicit smoke", "paid ~$0.12"),
    ("odcv.elicit full", "paid ~$23"),
    ("odcv.analysis", "free"),
    ("odcv.qual_coding", "paid ~$0.03"),
)

#: Registry of the project's experiments.
EXPERIMENT_REGISTRY: dict[int, dict] = {
    1: {"name": "SWE-bench introspection battery", "status": "ready",
        "orchestrated": True,
        "report": "experiments/2026-05-16_self_prediction/REPORT.md"},
    2: {"name": "ODCV-Bench self/other framing gap", "status": "complete",
        "orchestrated": False,
        "report": "experiments/2026-07-29_odcv_framing/REPORT.md"},
    3: {"name": "Self-prediction integrating Petri", "status": "not_started",
        "orchestrated": False, "report": None},
}


def describe_experiment_2() -> dict:
    """Print Experiment 2's reproduction recipe; return its registry summary.

    Deliberately does not launch anything: the run is a sequence of paid steps
    gated on a read-only ``external/ODCV-Bench`` checkout and an approved spend
    rung per step.
    """
    meta = EXPERIMENT_REGISTRY[2]
    print(f"  Status: {meta['status']} — results are committed under {_EXP2_DIR.name}/.")
    print(f"  Report: {meta['report']}")
    print("  Not orchestrated from here (paid steps are approved individually).")
    print("  To reproduce, from the repo root with external/ODCV-Bench present:")
    for module, cost in EXP2_STEPS:
        print(f"    PYTHONPATH={_EXP2_DIR.relative_to(_ROOT)} python -m {module:<24} # {cost}")
    return {"status": meta["status"], "name": meta["name"],
            "report": meta["report"], "orchestrated": False}


def _load_exp1():
    """Load Experiment 1's ``run.py`` as a module (its directory name is not a
    valid Python identifier, so it cannot be imported normally)."""
    path = _EXP1_DIR / "run.py"
    spec = importlib.util.spec_from_file_location("exp1_run", path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: dataclass annotation resolution looks the module up
    # in sys.modules by name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_experiment_1(
    *,
    data_dir: Path | None = None,
    results_dir: Path | None = None,
    dry_run: bool = True,
    quick: bool = False,
    n_tasks: int | None = None,
    model_ids: tuple[str, ...] | None = None,
) -> dict:
    """Invoke Experiment 1 with the given tunable settings; return its run summary."""
    exp1 = _load_exp1()
    cfg = exp1.RunConfig(
        data_dir=data_dir or (_EXP1_DIR / "data"),
        results_dir=results_dir or (_EXP1_DIR / "results"),
        dry_run=dry_run,
    )
    if quick:
        cfg = dataclasses.replace(cfg, model_ids=exp1.QUICK_MODELS, n_tasks=4)
    if n_tasks is not None:
        cfg = dataclasses.replace(cfg, n_tasks=n_tasks)
    if model_ids is not None:
        cfg = dataclasses.replace(cfg, model_ids=tuple(model_ids))
    return exp1.run(cfg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the self-prediction capabilities experiments.")
    parser.add_argument("--experiments", nargs="+", type=int, default=[1],
                        choices=[1, 2, 3], help="Which experiments to run, in order.")
    parser.add_argument("--real", action="store_true",
                        help="REAL run (spends money, needs Docker + API key). Default: dry-run.")
    parser.add_argument("--quick", action="store_true", help="Small smoke run (Experiment 1).")
    parser.add_argument("--n-tasks", type=int, default=None)
    parser.add_argument("--data-dir", default=None, help="Override Experiment 1's data/ cache directory.")
    parser.add_argument("--results-dir", default=None, help="Override Experiment 1's results/ directory.")
    args = parser.parse_args(argv)

    if args.real:
        try:
            from dotenv import load_dotenv

            load_dotenv(_ROOT / ".env")
        except ImportError:
            pass

    summaries: dict[int, dict] = {}
    for exp_id in args.experiments:
        meta = EXPERIMENT_REGISTRY[exp_id]
        print(f"\n=== Experiment {exp_id}: {meta['name']} ===")
        if exp_id == 1:
            summaries[exp_id] = run_experiment_1(
                data_dir=Path(args.data_dir) if args.data_dir else None,
                results_dir=Path(args.results_dir) if args.results_dir else None,
                dry_run=not args.real, quick=args.quick,
                n_tasks=args.n_tasks,
            )
        elif exp_id == 2:
            summaries[exp_id] = describe_experiment_2()
        else:
            print(f"  Experiment {exp_id} is not yet implemented "
                  f"(status: {meta['status']}). Skipping.")
            summaries[exp_id] = {"status": meta["status"], "name": meta["name"]}

    print("\n=== Orchestration summary ===")
    print(json.dumps(summaries, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
