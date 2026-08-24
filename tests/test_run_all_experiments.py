"""Tests for run_all_experiments.py -- the root orchestrator (sec 0)."""

from pathlib import Path

import run_all_experiments as orch


def test_experiment_registry_has_three_experiments():
    assert set(orch.EXPERIMENT_REGISTRY) == {1, 2, 3}
    assert orch.EXPERIMENT_REGISTRY[1]["status"] == "ready"
    assert orch.EXPERIMENT_REGISTRY[2]["status"] == "complete"
    assert orch.EXPERIMENT_REGISTRY[3]["status"] == "not_started"


def test_shipped_experiments_point_at_an_existing_report():
    root = Path(orch.__file__).parent
    for exp_id in (1, 2):
        report = orch.EXPERIMENT_REGISTRY[exp_id]["report"]
        assert (root / report).exists(), f"experiment {exp_id} report missing: {report}"


def test_experiment_2_describes_but_never_launches(capsys):
    summary = orch.describe_experiment_2()
    out = capsys.readouterr().out
    assert summary["status"] == "complete"
    assert summary["orchestrated"] is False
    # The recipe is printed in full so nobody has to guess the step order.
    for module, _cost in orch.EXP2_STEPS:
        assert module in out


def test_orchestrator_invokes_experiment_1(tmp_path):
    summary = orch.run_experiment_1(
        data_dir=tmp_path / "data",
        results_dir=tmp_path / "results",
        dry_run=True,
        quick=True,
    )
    assert summary["mode"] == "dry_run"
    assert summary["n_models"] == 5
    assert isinstance(summary["rq1_mean_loss"], float)
    assert (tmp_path / "results" / "analysis.json").exists()


def test_main_runs_experiment_1_quick(tmp_path, capsys):
    rc = orch.main([
        "--experiments", "1", "--quick",
        "--data-dir", str(tmp_path / "data"),
        "--results-dir", str(tmp_path / "results"),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Experiment 1" in out
    assert (tmp_path / "results" / "run_summary.json").exists()


def test_main_skips_unimplemented_experiments(capsys):
    rc = orch.main(["--experiments", "2", "3"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "not yet implemented" in out
    # ...but only about experiment 3 -- experiment 2 has shipped.
    unimplemented = out.split("=== Experiment 3")[1]
    assert "not yet implemented" in unimplemented
    assert "not yet implemented" not in out.split("=== Experiment 3")[0]
