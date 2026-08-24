"""Tests for src/types.py -- shared dataclasses."""

import pytest

from src.types import Cell, Dimension, Forecast, Outcome, Rollout


# --------------------------------------------------------------------------
# Dimension
# --------------------------------------------------------------------------
def test_dimension_valid():
    d = Dimension(name="p_solve", kind="binary")
    assert d.name == "p_solve"
    assert d.is_binary
    assert not d.is_continuous


def test_dimension_continuous_flags():
    d = Dimension(name="avg_tokens_step", kind="continuous")
    assert d.is_continuous
    assert not d.is_binary


def test_dimension_rejects_bad_kind():
    with pytest.raises(ValueError):
        Dimension(name="x", kind="ternary")


# --------------------------------------------------------------------------
# Rollout
# --------------------------------------------------------------------------
def test_rollout_roundtrip():
    r = Rollout(
        model="qwen/qwen3-8b",
        task_id="django__django-1",
        rollout_idx=3,
        trajectory={"messages": [], "info": {"submission": "diff"}},
        solved=True,
        errored=False,
        error=None,
    )
    d = r.to_dict()
    r2 = Rollout.from_dict(d)
    assert r2 == r
    assert r2.trajectory["info"]["submission"] == "diff"


def test_rollout_errored_defaults():
    r = Rollout(model="m", task_id="t", rollout_idx=0)
    assert r.errored is False
    assert r.solved is None
    assert r.trajectory is None


# --------------------------------------------------------------------------
# Outcome
# --------------------------------------------------------------------------
def test_outcome_valid_for():
    o = Outcome(
        model="m",
        task_id="t",
        rollout_idx=0,
        values={"p_solve": 1.0, "avg_tokens_step": None},
    )
    assert o.valid_for("p_solve")
    assert not o.valid_for("avg_tokens_step")  # value is None
    assert not o.valid_for("missing_dim")      # absent key


def test_outcome_errored_is_never_valid():
    o = Outcome(
        model="m", task_id="t", rollout_idx=0,
        values={"p_solve": 1.0}, errored=True, error="boom",
    )
    assert not o.valid_for("p_solve")


def test_outcome_roundtrip():
    o = Outcome(model="m", task_id="t", rollout_idx=1, values={"p_solve": 0.0})
    assert Outcome.from_dict(o.to_dict()) == o


# --------------------------------------------------------------------------
# Forecast
# --------------------------------------------------------------------------
def test_forecast_roundtrip():
    f = Forecast(
        model="z-ai/glm-5",
        task_id="t",
        condition="thinking",
        kind="self",
        predictions={"p_solve": 0.7},
        broken=False,
        raw_response='{"p_solve": 0.7}',
    )
    assert Forecast.from_dict(f.to_dict()) == f


def test_forecast_rejects_bad_enum():
    with pytest.raises(ValueError):
        Forecast(model="m", task_id="t", condition="thinking", kind="other")
    with pytest.raises(ValueError):
        Forecast(model="m", task_id="t", condition="sometimes", kind="self")


def test_primed_self_forecast_roundtrips_its_priming_turn():
    f = Forecast(model="qwen/qwen3-8b", task_id="t", condition="thinking",
                 kind="self_primed", identity_reply="I am Qwen3-8B.")
    assert Forecast.from_dict(f.to_dict()) == f


def test_pre_existing_records_still_load():
    """Old cache files have no identity_reply field; they must keep deserialising."""
    f = Forecast.from_dict({"model": "m", "task_id": "t", "condition": "thinking",
                            "kind": "self", "predictions": {"p_solve": 0.5}})
    assert f.identity_reply is None


def test_primed_forecast_demands_its_priming_turn():
    with pytest.raises(ValueError, match="identity_reply"):
        Forecast(model="m", task_id="t", condition="thinking", kind="self_primed")


def test_unprimed_forecast_refuses_a_priming_turn():
    with pytest.raises(ValueError, match="identity_reply"):
        Forecast(model="m", task_id="t", condition="thinking", kind="self",
                 identity_reply="I am m.")


def test_an_empty_priming_turn_is_still_a_priming_turn():
    """A model that answered with nothing was primed; "" must not read as absent."""
    f = Forecast(model="m", task_id="t", condition="thinking", kind="self_primed",
                 identity_reply="")
    assert f.identity_reply == ""


# --------------------------------------------------------------------------
# Cell
# --------------------------------------------------------------------------
def _cell_with_n_valid(dim: str, n_valid: int, n_total: int = 8) -> Cell:
    outcomes = []
    for i in range(n_total):
        val = 1.0 if i < n_valid else None
        outcomes.append(Outcome(model="m", task_id="t", rollout_idx=i, values={dim: val}))
    return Cell(model="m", task_id="t", outcomes=outcomes)


def test_cell_valid_values():
    c = _cell_with_n_valid("p_solve", n_valid=6)
    assert c.valid_values("p_solve") == [1.0] * 6


def test_cell_retained_threshold():
    # >=5 of 8 valid -> retained
    assert _cell_with_n_valid("p_solve", 5).is_retained("p_solve")
    assert _cell_with_n_valid("p_solve", 8).is_retained("p_solve")
    # <5 valid -> dropped
    assert not _cell_with_n_valid("p_solve", 4).is_retained("p_solve")
    assert not _cell_with_n_valid("p_solve", 0).is_retained("p_solve")


def test_cell_roundtrip():
    c = _cell_with_n_valid("p_solve", 5)
    assert Cell.from_dict(c.to_dict()) == c
