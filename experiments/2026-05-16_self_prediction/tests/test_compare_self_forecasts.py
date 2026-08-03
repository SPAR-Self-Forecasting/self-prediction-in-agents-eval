"""Tests for compare_self_forecasts.py -- the unprimed vs identity-primed sweep.

The risks worth pinning: the two variants must be compared on the *same* tasks
and sample slots (or task difficulty confounds the priming effect), a killed sweep
must resume rather than re-bill, and the reported shift must keep its sign.
"""

from __future__ import annotations

import json

import pytest

import compare_self_forecasts as csf
import config
from dataset import mock_tasks

TASKS = mock_tasks()[:2]
MODELS = [config.model_by_id("qwen/qwen3-8b"), config.model_by_id("z-ai/glm-5")]
CONDITIONS = ["no_thinking"]


def _forecast_json(**ov):
    vals = {name: 0.5 for name in config.DIMENSION_NAMES}
    vals.update({"avg_tokens_step": 300, "total_expected_steps": 35})
    vals.update(ov)
    return json.dumps(vals)


class _Scripted:
    """Returns a fixed forecast, and a fixed identity reply for the priming turn."""

    def __init__(self, plain=0.4, primed=0.6, identity="I am a model."):
        self.plain, self.primed, self.identity = plain, primed, identity
        self.calls, self.chats = [], []

    def complete(self, model, prompt, **kwargs):
        self.calls.append(prompt)
        if "what model you are" in prompt.lower():
            return type("R", (), {"content": self.identity})()
        return type("R", (), {"content": _forecast_json(p_solve=self.plain)})()

    def chat(self, model, messages, **kwargs):
        self.chats.append(messages)
        return type("R", (), {"content": _forecast_json(p_solve=self.primed)})()


# --------------------------------------------------------------------------
# Planning and budget
# --------------------------------------------------------------------------
def test_plan_covers_both_variants_over_identical_cells():
    units = csf.plan(MODELS, TASKS, CONDITIONS, samples=2)
    assert len(units) == 2 * len(MODELS) * len(TASKS) * 2
    plain = {(m.openrouter_id, t.instance_id, i) for (v, m, t, i) in units if v == "self"}
    primed = {(m.openrouter_id, t.instance_id, i) for (v, m, t, i) in units if v == "self_primed"}
    assert plain == primed          # same cells, or the comparison is confounded


def test_budget_charges_the_primed_variant_two_calls_per_sample():
    units = csf.plan(MODELS, TASKS, CONDITIONS, samples=2)
    budget = csf.call_budget(len(units), len(CONDITIONS))
    assert budget["primed_calls"] == 2 * budget["plain_calls"]
    assert budget["total_calls"] == budget["plain_calls"] + budget["primed_calls"]


# --------------------------------------------------------------------------
# Sweep and resume
# --------------------------------------------------------------------------
def test_sweep_writes_one_file_per_sample_with_every_condition(tmp_path):
    llm = _Scripted()
    csf.run_sweep(llm, MODELS, TASKS, CONDITIONS, 1, tmp_path, max_workers=2)
    path = csf.sample_path(tmp_path, "self_primed", MODELS[0].slug, TASKS[0].instance_id, 0)
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == set(CONDITIONS)
    assert payload["no_thinking"]["kind"] == "self_primed"
    assert payload["no_thinking"]["identity_reply"] == "I am a model."


def test_second_run_re_elicits_nothing(tmp_path):
    llm = _Scripted()
    first = csf.run_sweep(llm, MODELS, TASKS, CONDITIONS, 1, tmp_path, max_workers=2)
    second = csf.run_sweep(llm, MODELS, TASKS, CONDITIONS, 1, tmp_path, max_workers=2)
    assert first > 0 and second == 0


def test_a_half_written_sweep_resumes_only_the_missing_samples(tmp_path):
    csf.run_sweep(_Scripted(), MODELS, TASKS, CONDITIONS, 1, tmp_path, max_workers=2)
    victim = csf.sample_path(tmp_path, "self", MODELS[0].slug, TASKS[0].instance_id, 0)
    victim.unlink()
    assert csf.run_sweep(_Scripted(), MODELS, TASKS, CONDITIONS, 1, tmp_path, max_workers=2) == 1


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
def test_shift_keeps_its_sign(tmp_path):
    """Priming that raises p_solve must report positive, not just 'different'."""
    csf.run_sweep(_Scripted(plain=0.4, primed=0.7), MODELS, TASKS, CONDITIONS, 1,
                  tmp_path, max_workers=2)
    cells = csf.load_cells(tmp_path, MODELS, TASKS, CONDITIONS, 1)
    report = csf.compare(cells, MODELS, CONDITIONS)
    shift = report[MODELS[0].openrouter_id]["shift"]["p_solve"]
    assert shift == pytest.approx(0.3, abs=1e-6)


def test_downward_shift_is_reported_negative(tmp_path):
    csf.run_sweep(_Scripted(plain=0.8, primed=0.5), MODELS, TASKS, CONDITIONS, 1,
                  tmp_path, max_workers=2)
    cells = csf.load_cells(tmp_path, MODELS, TASKS, CONDITIONS, 1)
    report = csf.compare(cells, MODELS, CONDITIONS)
    assert report[MODELS[0].openrouter_id]["shift"]["p_solve"] == pytest.approx(-0.3, abs=1e-6)


def test_comparison_counts_the_pairs_it_used(tmp_path):
    csf.run_sweep(_Scripted(), MODELS, TASKS, CONDITIONS, 1, tmp_path, max_workers=2)
    cells = csf.load_cells(tmp_path, MODELS, TASKS, CONDITIONS, 1)
    report = csf.compare(cells, MODELS, CONDITIONS)
    assert report[MODELS[0].openrouter_id]["n_pairs"] == len(TASKS) * len(CONDITIONS)


def test_a_model_with_no_primed_half_reports_no_pairs(tmp_path):
    """A one-sided cell must be dropped, never compared against itself."""
    csf.run_sweep(_Scripted(), MODELS, TASKS, CONDITIONS, 1, tmp_path, max_workers=2)
    for task in TASKS:
        csf.sample_path(tmp_path, "self_primed", MODELS[0].slug, task.instance_id, 0).unlink()
    cells = csf.load_cells(tmp_path, MODELS, TASKS, CONDITIONS, 1)
    report = csf.compare(cells, MODELS, CONDITIONS)
    assert report[MODELS[0].openrouter_id]["n_pairs"] == 0
    assert report[MODELS[0].openrouter_id]["shift"] == {}
    assert report[MODELS[1].openrouter_id]["n_pairs"] > 0     # the other model is unaffected


def test_identity_replies_are_surfaced_for_audit(tmp_path):
    csf.run_sweep(_Scripted(identity="I am GLM-5."), MODELS, TASKS, CONDITIONS, 1,
                  tmp_path, max_workers=2)
    cells = csf.load_cells(tmp_path, MODELS, TASKS, CONDITIONS, 1)
    report = csf.compare(cells, MODELS, CONDITIONS)
    ident = report[MODELS[0].openrouter_id]["identity_replies"]
    assert ident["n"] == len(TASKS) * len(CONDITIONS)
    assert ident["n_empty"] == 0
    assert ident["example"] == "I am GLM-5."


def test_empty_identity_replies_are_counted(tmp_path):
    csf.run_sweep(_Scripted(identity=""), MODELS, TASKS, CONDITIONS, 1,
                  tmp_path, max_workers=2)
    cells = csf.load_cells(tmp_path, MODELS, TASKS, CONDITIONS, 1)
    report = csf.compare(cells, MODELS, CONDITIONS)
    ident = report[MODELS[0].openrouter_id]["identity_replies"]
    assert ident["n_empty"] == ident["n"] > 0
    assert ident["example"] is None


def test_report_never_claims_to_measure_accuracy(tmp_path):
    """Without outcomes these are belief shifts; the text must not overstate."""
    csf.run_sweep(_Scripted(), MODELS, TASKS, CONDITIONS, 1, tmp_path, max_workers=2)
    cells = csf.load_cells(tmp_path, MODELS, TASKS, CONDITIONS, 1)
    text = csf.format_report(csf.compare(cells, MODELS, CONDITIONS))
    assert "not accuracy changes" in text


# --------------------------------------------------------------------------
# Task selection
# --------------------------------------------------------------------------
def test_tasks_are_reused_from_a_run_cache_when_present(tmp_path):
    """Comparing across different task samples would confound difficulty."""
    (tmp_path / "tasks.json").write_text(
        json.dumps([t.to_dict() for t in TASKS]), encoding="utf-8")
    loaded = csf.load_tasks(2, dry_run=True, data_dir=tmp_path)
    assert [t.instance_id for t in loaded] == [t.instance_id for t in TASKS]


def test_task_selection_is_seeded_and_repeatable():
    a = csf.load_tasks(3, dry_run=True)
    b = csf.load_tasks(3, dry_run=True)
    assert [t.instance_id for t in a] == [t.instance_id for t in b]
    assert len(a) == 3


def test_main_dry_run_writes_a_comparison(tmp_path):
    out = tmp_path / "sweep"
    assert csf.main(["--dry-run", "--tasks", "1", "--samples", "1",
                     "--models", "z-ai/glm-5", "--out-dir", str(out)]) == 0
    report = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert "z-ai/glm-5" in report


def test_main_rejects_an_unknown_model():
    with pytest.raises(SystemExit):
        csf.main(["--dry-run", "--models", "not/a-model"])
