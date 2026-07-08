"""Integration test for run.py -- the mocked dry-run executes all 9 stages."""

import dataclasses
import json

import pytest

import run
from run import QUICK_MODELS, RunConfig


def _quick_config(tmp_path, render_figures: bool = False) -> RunConfig:
    """Figures off by default: most tests re-run the pipeline repeatedly and
    only the dedicated end-to-end test needs stage 9's matplotlib work."""
    return RunConfig(
        data_dir=tmp_path / "data",
        results_dir=tmp_path / "results",
        dry_run=True,
        model_ids=QUICK_MODELS,
        n_tasks=4,
        render_figures=render_figures,
    )


# --------------------------------------------------------------------------
# Mock LLM client
# --------------------------------------------------------------------------
def test_mock_client_dispatches_by_prompt():
    llm = run.MockLLMClient()
    assert "p_solve" in llm.complete("m", "...\nTASK DESCRIPTION:\nfix it").content
    assert "documentation" in llm.complete("m", "[PATCH DIFF]\ndiff").content
    assert "express_uncertainty" in llm.complete("m", "[REASONING TEXT]\nhmm").content
    assert "broken" in llm.complete("m", "[RAW RESPONSE]\ngarbage").content
    assert llm.call_count == 4


# --------------------------------------------------------------------------
# End-to-end dry-run
# --------------------------------------------------------------------------
def test_dry_run_executes_all_nine_stages(tmp_path):
    summary = run.run(_quick_config(tmp_path, render_figures=True))

    # all artifacts written
    for name in ("tasks.json", "cross_predictor.json"):
        assert (tmp_path / "data" / name).exists()
    for name in ("scores.json", "analysis.json", "murphy.json", "run_summary.json"):
        assert (tmp_path / "results" / name).exists()

    # stage 9: the flat headline plots/ set (RQ1-3 + the two Murphy figures)...
    plots_dir = tmp_path / "results" / "plots"
    for name in ("rq1_wins_by_model", "rq2_wins_vs_solverate", "rq3_thinking_delta",
                 "murphy_decomposition", "reliability_diagrams"):
        assert (plots_dir / f"{name}.png").exists(), f"missing headline plot {name}"
    # ...and the full figures/ set under primary__<cross_slug>/ (A1-A3, B1-B2, C1-C2)
    cross_slug = json.loads(
        (tmp_path / "data" / "cross_predictor.json").read_text())["cross_model"].replace("/", "__")
    figures_dir = tmp_path / "results" / "figures" / f"primary__{cross_slug}"
    assert len(list(figures_dir.glob("*.png"))) == 7

    # stage 8: murphy.json covers the binary dims with the exact CORP identity
    murphy = json.loads((tmp_path / "results" / "murphy.json").read_text())
    block = murphy["conditions"]["no_thinking"]["p_solve"]
    for entry in block["per_model"].values():
        for kind in ("self", "cross"):
            comp = entry[kind]
            assert comp["brier"] == pytest.approx(
                comp["reliability"] - comp["resolution"] + comp["uncertainty"], abs=1e-12)

    # summary carries each RQ's headline point estimate (wins-based, sec 15.6)
    assert summary["mode"] == "dry_run"
    assert summary["n_models"] == 5 and summary["n_tasks"] == 4
    assert summary["n_iai_models"] == 4              # 5 models - 1 cross-predictor
    for key in ("rq1_mean_wins", "rq2_spearman_rho_wins_vs_solve_rate", "rq3_delta_wins"):
        assert isinstance(summary[key], float)

    analysis = json.loads((tmp_path / "results" / "analysis.json").read_text())
    for section in ("rq1_existence", "rq2_scaling", "rq3_elicitation",
                    "exploratory", "missingness", "solve_rates"):
        assert section in analysis
    # rq2 is descriptive (Spearman rho only); the inferential battery stays pruned
    assert "spearman_rho" in analysis["rq2_scaling"]
    assert "robustness" not in analysis


def test_dry_run_designates_a_cross_predictor(tmp_path):
    summary = run.run(_quick_config(tmp_path))
    assert summary["cross_model"] in QUICK_MODELS
    # the cross-predictor receives no wins score
    scores = json.loads((tmp_path / "results" / "scores.json").read_text())
    assert summary["cross_model"] not in scores["wins"]["no_thinking"]


def test_dry_run_is_resumable(tmp_path):
    cfg = _quick_config(tmp_path)
    first = run.run(cfg)
    # second run re-reads every cache; designation and headline estimates are stable
    second = run.run(cfg)
    assert first["cross_model"] == second["cross_model"]
    assert first["rq1_mean_wins"] == pytest.approx(second["rq1_mean_wins"])
    assert first["llm_accounting"]["calls"] > 0
    assert second["llm_accounting"]["calls"] == 0  # everything served from cache


def test_rollouts_are_cached_per_rollout_index(tmp_path):
    """Fix 2: sub-cell checkpointing -- one cache file per (model, task, rollout_idx).

    The crucial robustness property: if a real run crashes mid-cell (say at
    rollout 5 of 8), the first 5 cached rollouts must survive the crash and
    only the remaining 3 get re-run on resume.
    """
    cfg = _quick_config(tmp_path)
    run.run(cfg)
    # Pick any (model, task) cell and confirm 8 per-rollout files exist.
    rollouts_root = tmp_path / "data" / "rollouts"
    cells = list(rollouts_root.rglob("r*.json"))
    # 5 models x 4 tasks x 8 rollouts = 160 files
    assert len(cells) == 5 * 4 * 8, f"expected 160 per-rollout files, got {len(cells)}"
    # And each (model, task) directory holds exactly 8 of them.
    for model_dir in rollouts_root.iterdir():
        for task_dir in model_dir.iterdir():
            assert task_dir.is_dir(), f"per-cell directory expected at {task_dir}"
            assert len(list(task_dir.glob("r*.json"))) == 8


def test_run_config_has_max_workers_default(tmp_path):
    """Fix 3: stages 3 + 5 parallelism is governed by ``max_workers`` (default 6)."""
    cfg = _quick_config(tmp_path)
    assert cfg.max_workers == 6


def test_run_config_has_rollout_workers_default(tmp_path):
    """Fix 4: stage 2 rollout parallelism is a separate knob.

    Conservative default 2 because each rollout launches a Docker container
    (agent sandbox) plus a second container for the SWE-bench eval; doubling
    that is enough RAM headroom for most dev machines.
    """
    cfg = _quick_config(tmp_path)
    assert cfg.rollout_workers == 2


def test_dry_run_with_high_parallelism_matches_serial(tmp_path, monkeypatch):
    """Output deterministically equals the serial reading regardless of worker count.

    Covers both knobs: LLM-bound (max_workers) and rollout (rollout_workers).
    """
    # Serial reference -- both knobs at 1.
    serial_dir = tmp_path / "serial"
    cfg_serial = dataclasses.replace(
        _quick_config(serial_dir), max_workers=1, rollout_workers=1,
    )
    s_summary = run.run(cfg_serial)
    s_scores = json.loads((serial_dir / "results" / "scores.json").read_text())

    # Parallel re-run on a separate data dir.
    parallel_dir = tmp_path / "parallel"
    cfg_parallel = dataclasses.replace(
        _quick_config(parallel_dir), max_workers=6, rollout_workers=4,
    )
    p_summary = run.run(cfg_parallel)
    p_scores = json.loads((parallel_dir / "results" / "scores.json").read_text())

    # Bit-for-bit identical outcomes -- mocked client is deterministic.
    assert s_summary["rq1_mean_wins"] == pytest.approx(p_summary["rq1_mean_wins"])
    assert s_summary["cross_model"] == p_summary["cross_model"]
    assert s_scores["wins"] == p_scores["wins"]


def test_partial_cell_resume_completes_missing_rollouts(tmp_path):
    """Delete 3 of 8 cached rollouts in a cell; the resume run must regenerate just those."""
    cfg = _quick_config(tmp_path)
    run.run(cfg)
    # Drop rollouts 1, 3, 5 of the first (model, task) cell.
    rollouts_root = tmp_path / "data" / "rollouts"
    model_dirs = sorted(rollouts_root.iterdir())
    task_dirs = sorted(model_dirs[0].iterdir())
    target = task_dirs[0]
    for idx in (1, 3, 5):
        (target / f"r{idx}.json").unlink()
    # Drop matching outcomes so stage 3 regenerates them too.
    outcomes_dir = tmp_path / "data" / "outcomes" / model_dirs[0].name / target.name
    for idx in (1, 3, 5):
        outcome_file = outcomes_dir / f"r{idx}.json"
        if outcome_file.exists():
            outcome_file.unlink()

    # Resume -- pipeline must rebuild only the deleted rollouts/outcomes.
    summary = run.run(cfg)
    assert summary["mode"] == "dry_run"
    # The cell is whole again.
    assert len(list(target.glob("r*.json"))) == 8
