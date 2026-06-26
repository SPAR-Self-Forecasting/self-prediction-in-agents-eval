"""Root orchestrator for the self-prediction capabilities project (EVAL_SPEC_V1.md sec 0).

Sequences the three experiments with a tunable configuration. Experiment 1
(SWE-bench introspection) is implemented; Experiments 2 and 3 are not yet
started and are reported as such rather than silently skipped.

Usage examples
--------------
    python run_all_experiments.py --experiments 1 --quick      # fast mocked smoke run
    python run_all_experiments.py --experiments 1              # full mocked dry-run
    python run_all_experiments.py --experiments 1 --real       # the real run (spends money)
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

#: Registry of the three experiments (sec 0).
EXPERIMENT_REGISTRY: dict[int, dict] = {
    1: {"name": "SWE-bench introspection battery", "status": "ready"},
    2: {"name": "Adversarial evaluation / scheming", "status": "not_started"},
    3: {"name": "Self-prediction integrating Petri", "status": "not_started"},
}


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
    bootstrap_b: int = 10_000,
    n_tasks: int | None = None,
    model_ids: tuple[str, ...] | None = None,
    n_alt_cross: int = 0,
) -> dict:
    """Invoke Experiment 1 with the given tunable settings; return its run summary."""
    exp1 = _load_exp1()
    cfg = exp1.RunConfig(
        data_dir=data_dir or (_EXP1_DIR / "data"),
        results_dir=results_dir or (_EXP1_DIR / "results"),
        dry_run=dry_run,
        bootstrap_b=bootstrap_b,
        n_alt_cross=n_alt_cross,
    )
    if quick:
        cfg = dataclasses.replace(cfg, model_ids=exp1.QUICK_MODELS, n_tasks=4,
                                  bootstrap_b=min(bootstrap_b, 200))
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
    parser.add_argument("--bootstrap-b", type=int, default=10_000)
    parser.add_argument("--n-tasks", type=int, default=None)
    parser.add_argument("--n-alt-cross", type=int, default=0,
                        help="Alternative cross-predictors for the sec 9.5 sensitivity battery.")
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
                bootstrap_b=args.bootstrap_b, n_tasks=args.n_tasks,
                n_alt_cross=args.n_alt_cross,
            )
        else:
            print(f"  Experiment {exp_id} is not yet implemented "
                  f"(status: {meta['status']}; see EVAL_SPEC_V1.md sec 0). Skipping.")
            summaries[exp_id] = {"status": "not_started"}

    print("\n=== Orchestration summary ===")
    print(json.dumps(summaries, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
