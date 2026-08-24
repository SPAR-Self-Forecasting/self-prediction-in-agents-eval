"""Experiment 1 entrypoint -- the 8-stage pipeline (EVAL_SPEC_V1.md sec 12.1).

    1. Dataset                -- select 30 SWE-bench Verified tasks (10/10/10).
    2. Rollouts               -- 12 x 30 x 8 mini-swe-agent rollouts.
    3. Ground truth           -- extract all 10 dimensions; dims 3-5 via 2 judges.
    4. Forecasting            -- self forecasts for all models, two-stage extraction.
    5. Scoring                -- self-forecast loss aggregation.
    6. Analysis               -- descriptive readout (point estimates only; the
                                 inferential statistics tier is pruned for the MVP).
    7. Murphy                 -- Brier decomposition of the binary dims
                                 (REL - RES + UNC; murphy.py) -> murphy.json.
    8. Figures                -- every figure: the flat headline plots/ set
                                 (make_plots.py) + the full figures/ set (plots.py).

Every stage caches to ``data/`` at (model, task) granularity, so an interrupted
run resumes by re-reading the cache. ``--dry-run`` swaps in the mocked LLM client
and mocked rollouts: the full 8 stages run with no network, no Docker, no spend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

# Make `src` importable when run.py is launched directly (python run.py).
_EXP_DIR = Path(__file__).parent.resolve()
_ROOT = _EXP_DIR.parents[1]
for _p in (str(_ROOT), str(_EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import analysis
import config
import dataset
import forecasting
import judge
import metrics
import murphy
import rollout
import trajectory
from src.llm_client import LLMClient
from src.types import Forecast, Outcome, Rollout

#: A small model subset (3 families) for fast end-to-end smoke runs.
QUICK_MODELS = (
    "qwen/qwen3-8b", "qwen/qwen3-14b", "google/gemini-2.5-flash-lite",
    "z-ai/glm-4.5", "z-ai/glm-4.7",
)


# --------------------------------------------------------------------------
# Run configuration
# --------------------------------------------------------------------------
@dataclass
class RunConfig:
    """Tunable settings for one Experiment-1 run."""

    data_dir: Path
    results_dir: Path
    dry_run: bool = True
    model_ids: tuple[str, ...] = config.MODEL_IDS
    n_tasks: int = config.N_TASKS
    n_rollouts: int = config.N_ROLLOUTS
    conditions: tuple[str, ...] = config.CONDITIONS
    step_limit: int = config.STEP_LIMIT
    seed: int = config.SEED
    #: ThreadPoolExecutor workers for the LLM-bound stages (3 ground-truth
    #: judges + 5 forecasting). LLM calls have no local resource impact, so
    #: 6 is a safe default; OpenRouter rate limits are the only ceiling.
    max_workers: int = 6
    #: ThreadPoolExecutor workers for stage 2 (rollouts). Each worker launches
    #: a Docker container (agent sandbox) and later a second container (SWE-bench
    #: eval), so ``2 * rollout_workers`` containers may be in flight. Default 2
    #: targets ~4 containers @ ~2 GB each = ~8 GB peak; bump to 4 or 6 on a
    #: machine with 16+ GB free RAM to cut stage 2 wall-clock further.
    rollout_workers: int = 2
    #: Stage 9 toggle -- figures render by default so one ``run.py`` invocation
    #: produces every artifact; tests that re-run the pipeline repeatedly turn
    #: it off to keep the suite fast.
    render_figures: bool = True
    fallback_template: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Resolve to absolute paths. Stage 2 runs the SWE-bench eval subprocess
        # with cwd=rollout_work, so a *relative* data_dir would make the
        # predictions path resolve against the wrong directory (FileNotFoundError).
        self.data_dir = Path(self.data_dir).resolve()
        self.results_dir = Path(self.results_dir).resolve()
        if self.fallback_template is None:
            tmpl = _EXP_DIR / "prompts" / "extraction_fallback.txt"
            self.fallback_template = tmpl.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Mocked LLM client (dry-run / tests)
# --------------------------------------------------------------------------
class _Resp:
    """Minimal response object exposing the surface forecasting/judge/extraction use."""

    __slots__ = ("content", "cost", "usage", "model")

    def __init__(self, content: str):
        self.content = content
        self.cost = 0.0
        self.usage = {}
        self.model = "mock"


def _seeded_rng(text: str) -> random.Random:
    return random.Random(int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big"))


def _mock_forecast_json(rng: random.Random) -> str:
    forecast: dict[str, float] = {}
    for dim in config.DIMENSIONS:
        if dim.name == "avg_tokens_step":
            forecast[dim.name] = rng.randint(80, 600)
        elif dim.name == "total_expected_steps":
            forecast[dim.name] = rng.randint(5, 80)
        else:  # binary probabilities + explore_edit_ratio in [0, 1]
            forecast[dim.name] = round(rng.random(), 3)
    return json.dumps(forecast)


class MockLLMClient:
    """Deterministic stand-in for the OpenRouter client used by ``--dry-run``.

    Dispatches on prompt markers to return a structurally valid forecast / judge
    / extraction-fallback reply. Responses are seeded by the prompt, so a dry-run
    is fully reproducible and resumable.
    """

    def __init__(self) -> None:
        self.call_count = 0
        self.total_cost = 0.0
        self.by_type = {"forecast": 0, "judge": 0, "fallback": 0, "identity": 0, "other": 0}

    def complete(self, model: str, prompt: str, **kwargs) -> _Resp:
        self.call_count += 1
        rng = _seeded_rng(prompt)
        if "[RAW RESPONSE]" in prompt:
            self.by_type["fallback"] += 1
            return _Resp(json.dumps({"broken": False,
                                     "forecast": json.loads(_mock_forecast_json(rng))}))
        if "[PATCH DIFF]" in prompt:
            self.by_type["judge"] += 1
            return _Resp(json.dumps({
                "documentation": {"bool": rng.random() < 0.5, "evidence": "mock"},
                "defensive_code": {"bool": rng.random() < 0.4, "evidence": "mock"},
            }))
        if "[REASONING TEXT]" in prompt:
            self.by_type["judge"] += 1
            return _Resp(json.dumps({
                "express_uncertainty": {"bool": rng.random() < 0.5, "evidence": "mock"},
            }))
        # "AGENT SCAFFOLD" heads the scaffold block in all three forecast templates.
        # It replaced an older "TASK DESCRIPTION:" marker that no longer appears in
        # any prompt, which left this branch dead: dry-run forecasts fell through to
        # "{}", failed extraction, and were silently recovered by the fallback call,
        # so a dry run exercised only the recovery path and never the happy one.
        if "AGENT SCAFFOLD" in prompt:
            self.by_type["forecast"] += 1
            return _Resp(_mock_forecast_json(rng))
        if "what model you are" in prompt.lower():
            self.by_type["identity"] += 1
            spec = config.MODEL_BY_ID.get(model)
            return _Resp(f"I am {spec.name}." if spec else "I am a language model.")
        self.by_type["other"] += 1
        return _Resp("{}")

    def chat(self, model: str, messages: list[dict], **kwargs) -> _Resp:
        """Multi-turn entry point, used by the identity-primed self-forecast.

        Dispatches on the last user turn, which carries the actual request; the
        earlier identity turns only set context.
        """
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        return self.complete(model, last_user, **kwargs)

    def accounting(self) -> dict:
        return {"calls": self.call_count, "total_cost_usd": 0.0,
                "by_type": dict(self.by_type), "mock": True}


# --------------------------------------------------------------------------
# JSON cache helpers
# --------------------------------------------------------------------------
def _json_default(obj):
    import numpy as np

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON-serialisable: {type(obj)}")


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default),
                    encoding="utf-8")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _log(message: str) -> None:
    print(f"[run] {message}", flush=True)


# --------------------------------------------------------------------------
# Stage 1 -- dataset
# --------------------------------------------------------------------------
def stage_dataset(cfg: RunConfig) -> list[dataset.Task]:
    path = cfg.data_dir / "tasks.json"
    if path.exists():
        tasks = [dataset.Task.from_dict(d) for d in _read_json(path)]
        _log(f"stage 1 dataset: {len(tasks)} tasks (cached)")
        return tasks
    selected = dataset.mock_tasks(cfg.seed) if cfg.dry_run else dataset.select_tasks(cfg.seed)
    tasks = selected[: cfg.n_tasks]
    _write_json(path, [t.to_dict() for t in tasks])
    _log(f"stage 1 dataset: {len(tasks)} tasks selected")
    return tasks


# --------------------------------------------------------------------------
# Stage 2 -- rollouts
# --------------------------------------------------------------------------
def _real_one_rollout(cfg: RunConfig, model: str, task: dataset.Task, idx: int) -> Rollout:
    """Run + evaluate one real rollout (one (model, task, rollout_idx) cell).

    Sub-cell granularity (Fix 2): a crash here loses *one* rollout, not all 8.
    Stage 2's caller writes the per-rollout cache file immediately after this
    returns, so resume only re-runs the indices missing from disk.
    """
    work = cfg.data_dir / "rollout_work"
    r = rollout.run_real_rollout(model, task, idx, work, cfg.step_limit)
    if not r.errored and r.trajectory is not None:
        patch = trajectory.extract_final_patch(r.trajectory)
        run_id = f"exp1-{config.model_by_id(model).slug}-{task.instance_id}-r{idx}"
        preds = work / "preds" / f"{run_id}.json"
        _write_json(preds, [{"instance_id": task.instance_id,
                             "model_name_or_path": run_id, "model_patch": patch}])
        try:
            rollout.run_swebench_evaluation(preds, run_id, work)
            r.solved = rollout.read_resolved(work, run_id, run_id, task.instance_id)
        except Exception as exc:  # harness failure -> errored rollout
            r.errored, r.error = True, f"evaluation failed: {exc}"
    return r


def _rollout_cache_path(root: Path, slug: str, task_id: str, idx: int) -> Path:
    """Sub-cell layout: ``rollouts/<slug>/<task_id>/r<idx>.json`` (one rollout per file)."""
    return root / slug / task_id / f"r{idx}.json"


def stage_rollouts(cfg: RunConfig, tasks: list[dataset.Task]) -> dict:
    """Produce (or load) 8 rollouts per (model, task); cached one file per rollout.

    Sub-cell checkpointing (Fix 2) + parallel rollouts (Fix 4): missing
    ``(model, task, rollout_idx)`` units are computed across a
    ThreadPoolExecutor of ``cfg.rollout_workers`` threads. Each worker launches
    its own Docker containers (mini-extra swebench → SWE-bench eval) and writes
    its own cache file atomically when done. A crash here loses at most one
    rollout, not the whole cell.
    """
    rollouts_root = cfg.data_dir / "rollouts"

    # 1. Enumerate (model, task, idx, path) units missing from cache.
    work: list[tuple[str, "dataset.Task", int, Path]] = []
    for model in cfg.model_ids:
        slug = config.model_by_id(model).slug
        for task in tasks:
            for idx in range(cfg.n_rollouts):
                path = _rollout_cache_path(rollouts_root, slug, task.instance_id, idx)
                if not path.exists():
                    work.append((model, task, idx, path))

    # 2. Run missing units in parallel.
    def _produce_and_write(unit) -> None:
        model, task, idx, path = unit
        if cfg.dry_run:
            r = rollout.mock_rollout(model, task, idx)
        else:
            r = _real_one_rollout(cfg, model, task, idx)
        _write_json(path, r.to_dict())

    n_to_run = len(work)
    _parallel_fill(work, _produce_and_write, cfg.rollout_workers)

    # 3. Re-load every rollout (cached + freshly written) into the return dict.
    n_cells = 0
    n_cached = (len(cfg.model_ids) * len(tasks) * cfg.n_rollouts) - n_to_run
    by_model: dict[str, dict[str, list[Rollout]]] = {}
    for model in cfg.model_ids:
        slug = config.model_by_id(model).slug
        by_model[model] = {}
        for task in tasks:
            cell = [
                Rollout.from_dict(_read_json(
                    _rollout_cache_path(rollouts_root, slug, task.instance_id, idx),
                ))
                for idx in range(cfg.n_rollouts)
            ]
            by_model[model][task.instance_id] = cell
            n_cells += 1
    _log(f"stage 2 rollouts: {n_cells} cells ({n_cached} cached + {n_to_run} run) "
         f"x {cfg.n_rollouts} rollouts each; {cfg.rollout_workers}x workers")
    return by_model


# --------------------------------------------------------------------------
# Stage 3 -- ground truth
# --------------------------------------------------------------------------
def _extract_outcome(rollout_obj: Rollout, llm) -> Outcome:
    if rollout_obj.errored or rollout_obj.trajectory is None:
        return Outcome(rollout_obj.model, rollout_obj.task_id, rollout_obj.rollout_idx,
                       {d: None for d in config.DIMENSION_NAMES},
                       errored=True, error=rollout_obj.error)
    traj = rollout_obj.trajectory
    try:
        values = dict(trajectory.extract_deterministic(traj, rollout_obj.solved))
        values.update(judge.judge_trajectory(
            llm,
            trajectory.extract_final_patch(traj),
            trajectory.concatenated_reasoning(traj),
        ))
    except Exception as exc:
        # A single malformed/pathological rollout (e.g. weird trajectory shape)
        # becomes missing data -- never crash the whole ground-truth stage, which
        # is scoring already-collected data (sec 9.6 drop-never-impute).
        return Outcome(rollout_obj.model, rollout_obj.task_id, rollout_obj.rollout_idx,
                       {d: None for d in config.DIMENSION_NAMES},
                       errored=True, error=f"outcome extraction failed: {exc}")
    return Outcome(rollout_obj.model, rollout_obj.task_id, rollout_obj.rollout_idx, values)


def _outcome_cache_path(root: Path, slug: str, task_id: str, idx: int) -> Path:
    return root / slug / task_id / f"r{idx}.json"


def _parallel_fill(units, worker, max_workers: int) -> None:
    """Run ``worker(unit)`` for every ``unit`` in ``units``, in parallel.

    Each unit must be self-contained -- write its own cache file -- so the
    only shared mutable state is the LLM client's threadsafe accounting.
    Exceptions surface immediately (we want a loud failure, not silent loss).
    """
    if not units:
        return
    if max_workers <= 1:
        for u in units:
            worker(u)
        return
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # ``list(map(...))`` materialises the iterator so exceptions raise here.
        list(ex.map(worker, units))


def stage_ground_truth(cfg: RunConfig, rollouts: dict, llm) -> dict:
    """Extract the 10-dimension Outcome per rollout; cached per (model, task, rollout_idx).

    Sub-cell checkpointing (Fix 2) + parallel judges (Fix 3): missing outcomes
    are computed across a ThreadPoolExecutor of ``cfg.max_workers`` threads;
    each writes its own cache file the moment its judge calls return. A crash
    here loses only the in-flight rollout's outcome, not the whole cell.
    """
    outcomes_root = cfg.data_dir / "outcomes"

    # 1. Collect missing-unit work list.
    work: list[tuple[Rollout, Path]] = []
    for model in cfg.model_ids:
        slug = config.model_by_id(model).slug
        for task_id, cell in rollouts[model].items():
            for r in cell:
                path = _outcome_cache_path(outcomes_root, slug, task_id, r.rollout_idx)
                if not path.exists():
                    work.append((r, path))

    # 2. Run uncached units in parallel.
    def _judge_and_write(item: tuple[Rollout, Path]) -> None:
        r, path = item
        out = _extract_outcome(r, llm)
        _write_json(path, out.to_dict())

    n_to_run = len(work)
    _parallel_fill(work, _judge_and_write, cfg.max_workers)

    # 3. Re-load everything (cached + freshly written) into the return shape.
    by_model: dict[str, dict[str, list[Outcome]]] = {}
    for model in cfg.model_ids:
        slug = config.model_by_id(model).slug
        by_model[model] = {}
        for task_id, cell in rollouts[model].items():
            outs: list[Outcome] = []
            for r in cell:
                path = _outcome_cache_path(outcomes_root, slug, task_id, r.rollout_idx)
                outs.append(Outcome.from_dict(_read_json(path)))
            by_model[model][task_id] = outs
    _log(f"stage 3 ground truth: 10-dim outcomes extracted "
         f"({n_to_run} judged, rest from cache; {cfg.max_workers}x workers)")
    return by_model


# --------------------------------------------------------------------------
# Stage 4 -- forecasting
# --------------------------------------------------------------------------
def _forecast_sample(cfg: RunConfig, llm, forecaster, task, kind: str, sample_idx: int) -> dict[str, Forecast]:
    """Elicit one sample (both conditions) of one kind for one (forecaster, task) cell."""
    return {
        cond: forecasting.elicit_forecast(
            llm, forecaster, task, cond, kind,
            fallback_template=cfg.fallback_template, sample_idx=sample_idx,
        )
        for cond in cfg.conditions
    }


def _forecast_sample_path(cfg: RunConfig, kind: str, slug: str, task_id: str, sample_idx: int) -> Path:
    """Sub-cell layout (multi-sample, sec 9.6): one file per sample, both conditions inside.

    ``forecasts/<kind>/<slug>/<task_id>/s<sample_idx>.json``.
    """
    return cfg.data_dir / "forecasts" / kind / slug / task_id / f"s{sample_idx}.json"


def stage_forecasting(cfg: RunConfig, tasks, llm) -> dict:
    """Self forecasts for every model in the battery.

    Parallel forecasting (Fix 3) + multi-sample (sec 9.6 amendment 2026-05-20):
    N = ``config.N_FORECAST_SAMPLES`` independent samples per (forecaster, task)
    cell, computed across a ThreadPoolExecutor of ``cfg.max_workers`` threads.
    """
    work: list[tuple[str, "dataset.Task", str, int, Path]] = []
    for model in cfg.model_ids:
        slug = config.model_by_id(model).slug
        for task in tasks:
            for sample_idx in range(config.N_FORECAST_SAMPLES):
                path = _forecast_sample_path(cfg, "self", slug, task.instance_id, sample_idx)
                if not path.exists():
                    work.append((model, task, "self", sample_idx, path))

    def _elicit_and_write(unit) -> None:
        model, task, kind, sample_idx, path = unit
        forecaster = config.model_by_id(model)
        cell = _forecast_sample(cfg, llm, forecaster, task, kind, sample_idx)
        _write_json(path, {c: f.to_dict() for c, f in cell.items()})

    n_to_run = len(work)
    _parallel_fill(work, _elicit_and_write, cfg.max_workers)

    def _read_samples(model: str, kind: str) -> dict:
        slug = config.model_by_id(model).slug
        out: dict[str, dict[str, list[Forecast]]] = {}
        for task in tasks:
            per_condition: dict[str, list[Forecast]] = {c: [] for c in cfg.conditions}
            for sample_idx in range(config.N_FORECAST_SAMPLES):
                path = _forecast_sample_path(cfg, kind, slug, task.instance_id, sample_idx)
                if not path.exists():
                    continue
                for c, d in _read_json(path).items():
                    per_condition[c].append(Forecast.from_dict(d))
            out[task.instance_id] = per_condition
        return out

    self_forecasts = {m: _read_samples(m, "self") for m in cfg.model_ids}

    _log(f"stage 4 forecasting: {len(self_forecasts)} self forecast sets "
         f"({n_to_run} sample-units run, rest from cache; "
         f"{cfg.max_workers}x workers, N={config.N_FORECAST_SAMPLES} samples/cell)")
    return self_forecasts


# --------------------------------------------------------------------------
# Stage 5 -- scoring
# --------------------------------------------------------------------------
def _restricted_families(model_ids) -> dict[str, list[str]]:
    keep = set(model_ids)
    out = {f: [m for m in members if m in keep] for f, members in config.families().items()}
    return {f: v for f, v in out.items() if v}


def _build_scoring_data(cfg: RunConfig, outcomes, self_forecasts, task_ids: list[str]) -> metrics.ScoringData:
    return metrics.ScoringData(
        dimensions=config.DIMENSIONS,
        models=list(cfg.model_ids),
        families=_restricted_families(cfg.model_ids),
        tasks=list(task_ids),
        outcomes=outcomes,
        self_forecasts=self_forecasts,
    )


def _score_dict(data: metrics.ScoringData, conditions) -> dict:
    return {cond: metrics.all_model_scores(data, cond) for cond in conditions}


def stage_scoring_data(cfg: RunConfig, data: metrics.ScoringData) -> metrics.ScoringData:
    """Write ``scores.json`` from an assembled ``ScoringData``."""
    _write_json(cfg.results_dir / "scores.json", {
        "scores": _score_dict(data, cfg.conditions),
    })
    _log("stage 5 scoring: self-forecast loss computed")
    return data


def stage_scoring(cfg: RunConfig, outcomes, self_forecasts, task_ids):
    """Build the ScoringData and per-model self-forecast scores."""
    data = _build_scoring_data(cfg, outcomes, self_forecasts, task_ids)
    return stage_scoring_data(cfg, data)


def rescore_from_cache(cfg: RunConfig) -> dict:
    """Re-run stages 5–8 from an existing ``data_dir`` cache (no API, no Docker).

    Reads tasks, outcomes, and self forecasts from disk; writes fresh
    ``results/{scores,analysis,murphy}.json`` and figures.
    """
    started = time.time()
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    _log(f"rescore-only: loading cache from {cfg.data_dir}")
    data = murphy.load_scoring_data(cfg.data_dir)
    n_tasks = len(data.tasks)
    report = stage_analysis(cfg, stage_scoring_data(cfg, data))
    stage_murphy(cfg, data)
    if cfg.render_figures:
        stage_figures(cfg)
    summary = {
        "mode": "rescore",
        "n_models": report["rq1_accuracy"]["n_models"],
        "n_tasks": n_tasks,
        "rq1_mean_loss": report["rq1_accuracy"]["point_estimate"],
        "rq2_spearman_rho_loss_vs_solve_rate": report["rq2_scaling"]["spearman_rho"],
        "rq3_delta_loss": report["rq3_elicitation"]["point_estimate"],
        "elapsed_seconds": round(time.time() - started, 1),
    }
    _write_json(cfg.results_dir / "run_summary.json", summary)
    _log(f"rescore done in {summary['elapsed_seconds']}s -> {cfg.results_dir}")
    return summary


# --------------------------------------------------------------------------
# Stage 6 -- analysis (descriptive readout; statistics tier pruned for the MVP)
# --------------------------------------------------------------------------
def stage_analysis(cfg: RunConfig, data: metrics.ScoringData) -> dict:
    report = analysis.run_full_analysis(data)
    _write_json(cfg.results_dir / "analysis.json", report)
    rq1 = report["rq1_accuracy"]
    _log(f"stage 6 analysis: mean self loss = {rq1['point_estimate']:.4f}; "
         f"loss~capability Spearman rho = {report['rq2_scaling']['spearman_rho']:.3f}")
    return report


# --------------------------------------------------------------------------
# Stage 7 -- Murphy decomposition (binary dims; downstream of scoring)
# --------------------------------------------------------------------------
def stage_murphy(cfg: RunConfig, data: metrics.ScoringData) -> dict:
    report = murphy.murphy_report(data)
    _write_json(cfg.results_dir / "murphy.json", report)
    p_solve = report["conditions"]["no_thinking"].get("p_solve", {})
    mean = p_solve.get("mean_across_models", {})
    _log(f"stage 7 murphy: p_solve REL {mean.get('reliability', float('nan')):.3f}; "
         f"RES {mean.get('resolution', float('nan')):.3f} -> murphy.json")
    return report


# --------------------------------------------------------------------------
# Stage 8 -- figures (headline plots/ + full figures/ sets)
# --------------------------------------------------------------------------
def stage_figures(cfg: RunConfig) -> list[Path]:
    """Render every figure from the results just written to ``cfg.results_dir``.

    Two sets: the flat headline ``plots/`` set (make_plots.py; RQ1-3 + Murphy)
    and the full ``figures/`` set (plots.py; A1-A3, B1-B2, C1-C2).
    """
    import make_plots
    import plots

    headline = make_plots.render_headline_plots(cfg.results_dir)

    scores = _read_json(cfg.results_dir / "scores.json")
    analysis_obj = _read_json(cfg.results_dir / "analysis.json")
    murphy_obj = _read_json(cfg.results_dir / "murphy.json")
    full = plots.render_all(scores, analysis_obj, cfg.results_dir / "figures",
                            murphy=murphy_obj)

    written = headline + full
    _log(f"stage 8 figures: {len(headline)} headline plot(s) -> plots/, "
         f"{len(full)} figure(s) -> figures/")
    return written


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def build_llm_client(cfg: RunConfig):
    """The mocked client for a dry-run, otherwise the real OpenRouter client."""
    if cfg.dry_run:
        return MockLLMClient()
    return LLMClient()


def run(cfg: RunConfig) -> dict:
    """Execute all 8 pipeline stages and return a run summary."""
    started = time.time()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    _log(f"starting Experiment 1 ({'DRY-RUN (mocked)' if cfg.dry_run else 'REAL RUN'}); "
         f"{len(cfg.model_ids)} models, {cfg.n_tasks} tasks")
    llm = build_llm_client(cfg)

    tasks = stage_dataset(cfg)
    task_ids = [t.instance_id for t in tasks]
    rollouts = stage_rollouts(cfg, tasks)
    outcomes = stage_ground_truth(cfg, rollouts, llm)
    self_forecasts = stage_forecasting(cfg, tasks, llm)
    data = stage_scoring(cfg, outcomes, self_forecasts, task_ids)
    report = stage_analysis(cfg, data)
    stage_murphy(cfg, data)
    if cfg.render_figures:
        stage_figures(cfg)

    summary = {
        "mode": "dry_run" if cfg.dry_run else "real",
        "n_models": len(cfg.model_ids),
        "n_tasks": len(tasks),
        "rq1_mean_loss": report["rq1_accuracy"]["point_estimate"],
        "rq2_spearman_rho_loss_vs_solve_rate": report["rq2_scaling"]["spearman_rho"],
        "rq3_delta_loss": report["rq3_elicitation"]["point_estimate"],
        "llm_accounting": llm.accounting(),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    _write_json(cfg.results_dir / "run_summary.json", summary)
    _log(f"done in {summary['elapsed_seconds']}s -- summary -> "
         f"{cfg.results_dir / 'run_summary.json'}")
    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_config(args: argparse.Namespace) -> RunConfig:
    data_dir = Path(args.data_dir) if args.data_dir else _EXP_DIR / "data"
    results_dir = Path(args.results_dir) if args.results_dir else _EXP_DIR / "results"
    cfg = RunConfig(
        data_dir=data_dir, results_dir=results_dir, dry_run=not args.real,
        max_workers=args.max_workers, rollout_workers=args.rollout_workers,
    )
    if args.quick:
        cfg = replace(cfg, model_ids=QUICK_MODELS, n_tasks=4)
    if args.n_tasks is not None:
        cfg = replace(cfg, n_tasks=args.n_tasks)
    if args.step_limit is not None:
        cfg = replace(cfg, step_limit=args.step_limit)
    if args.models:
        cfg = replace(cfg, model_ids=tuple(args.models))
    if args.skip_figures:
        cfg = replace(cfg, render_figures=False)
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Experiment 1 -- SWE-bench introspection eval.")
    parser.add_argument("--real", action="store_true",
                        help="REAL run: spends money, needs Docker + API key. Default: mocked dry-run.")
    parser.add_argument("--quick", action="store_true",
                        help="Small 5-model / 4-task smoke run.")
    parser.add_argument("--n-tasks", type=int, default=None, help="Override the task count.")
    parser.add_argument("--step-limit", type=int, default=None,
                        help=f"Override the per-rollout step budget (default {config.STEP_LIMIT}).")
    parser.add_argument("--models", nargs="+", default=None, help="Override the model id list.")
    parser.add_argument("--max-workers", type=int, default=6,
                        help="ThreadPoolExecutor workers for the LLM-bound stages "
                             "(3 ground-truth judges + 5 forecasting). Default 6.")
    parser.add_argument("--rollout-workers", type=int, default=2,
                        help="ThreadPoolExecutor workers for stage 2 (rollouts). Each "
                             "worker runs a Docker container (mini-extra swebench) plus "
                             "later a SWE-bench eval container, so peak in-flight "
                             "containers ~= 2 * rollout_workers. Default 2 (~8 GB peak); "
                             "bump to 4 or 6 on a 16+ GB free-RAM machine.")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip stage 8 (figure rendering); compute-only run.")
    parser.add_argument("--rescore-only", action="store_true",
                        help="Re-score from an existing data_dir cache (stages 5–8 only; "
                             "no rollouts, forecasting, or API calls).")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args(argv)

    if args.real:
        try:
            from dotenv import load_dotenv

            load_dotenv(_ROOT / ".env")
        except ImportError:
            pass

    cfg = build_config(args)
    summary = rescore_from_cache(cfg) if args.rescore_only else run(cfg)
    print(json.dumps(summary, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
