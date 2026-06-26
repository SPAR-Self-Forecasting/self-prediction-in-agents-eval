"""Tests for src/extraction.py -- two-stage forecast JSON extractor (sec 10.9)."""

import json

import pytest

from src.extraction import extract_forecast, find_json_objects

DIMS = [
    "p_solve", "p_first_attempt_correct",
    "documentation", "defensive_code", "express_uncertainty",
    "avg_tokens_step", "total_expected_steps", "explore_edit_ratio",
]


def _good_forecast(**overrides):
    f = {
        "p_solve": 0.7, "p_first_attempt_correct": 0.5,
        "documentation": 0.4, "defensive_code": 0.3, "express_uncertainty": 0.2,
        "avg_tokens_step": 350, "total_expected_steps": 40, "explore_edit_ratio": 0.6,
    }
    f.update(overrides)
    return f


class FakeLLM:
    """Minimal client stub exposing the ``.complete`` surface extraction.py uses."""

    def __init__(self, content):
        self.content = content
        self.calls = 0

    def complete(self, model, prompt, **kwargs):
        self.calls += 1
        return type("R", (), {"content": self.content})()


# --------------------------------------------------------------------------
# find_json_objects
# --------------------------------------------------------------------------
def test_find_json_objects_balanced():
    text = 'noise {"a": 1} more {"b": {"c": 2}} tail'
    objs = find_json_objects(text)
    assert objs == ['{"a": 1}', '{"b": {"c": 2}}']


def test_find_json_objects_none():
    assert find_json_objects("no braces here") == []


# --------------------------------------------------------------------------
# Stage 1: programmatic parse
# --------------------------------------------------------------------------
def test_clean_json_parsed_programmatically():
    res = extract_forecast(json.dumps(_good_forecast()), DIMS)
    assert not res.broken
    assert res.stage == "programmatic"
    assert not res.used_fallback
    assert res.predictions["p_solve"] == pytest.approx(0.7)


def test_json_with_surrounding_prose():
    raw = "Here is my forecast:\n" + json.dumps(_good_forecast()) + "\nThanks!"
    res = extract_forecast(raw, DIMS)
    assert not res.broken and res.stage == "programmatic"


def test_markdown_fenced_json():
    raw = "Reasoning...\n```json\n" + json.dumps(_good_forecast()) + "\n```\n"
    res = extract_forecast(raw, DIMS)
    assert not res.broken and res.stage == "programmatic"


def test_reasoning_leakage_picks_trailing_forecast():
    # A stray brace-object from leaked reasoning precedes the real forecast.
    raw = (
        'Let me think. {"note": "p_solve is hard to call"} '
        "Final answer:\n" + json.dumps(_good_forecast(p_solve=0.9))
    )
    res = extract_forecast(raw, DIMS)
    assert not res.broken
    assert res.predictions["p_solve"] == pytest.approx(0.9)


# --------------------------------------------------------------------------
# Stage 1 failure -> Stage 2 LLM fallback
# --------------------------------------------------------------------------
def test_missing_dimension_triggers_fallback():
    bad = _good_forecast()
    del bad["explore_edit_ratio"]
    recovered = json.dumps({"broken": False, "forecast": _good_forecast()})
    llm = FakeLLM(recovered)
    res = extract_forecast(json.dumps(bad), DIMS, llm_client=llm, fallback_model="openai/gpt-4o-mini")
    assert res.used_fallback and not res.broken
    assert res.stage == "llm_fallback"
    assert llm.calls == 1
    assert res.predictions["explore_edit_ratio"] == pytest.approx(0.6)


def test_string_value_triggers_fallback():
    bad = _good_forecast(p_solve="seventy percent")
    llm = FakeLLM(json.dumps({"broken": False, "forecast": _good_forecast()}))
    res = extract_forecast(json.dumps(bad), DIMS, llm_client=llm)
    assert res.used_fallback and not res.broken


def test_bool_value_rejected_by_stage1():
    # booleans are JSON-valid but not well-typed numeric forecasts.
    bad = _good_forecast(documentation=True)
    llm = FakeLLM(json.dumps({"broken": False, "forecast": _good_forecast()}))
    res = extract_forecast(json.dumps(bad), DIMS, llm_client=llm)
    assert res.used_fallback


def test_fallback_declares_broken():
    llm = FakeLLM(json.dumps({"broken": True}))
    res = extract_forecast("total garbage, no json", DIMS, llm_client=llm)
    assert res.broken and res.used_fallback
    assert res.predictions is None


def test_no_llm_client_and_stage1_fails_is_broken():
    res = extract_forecast("garbage", DIMS)
    assert res.broken
    assert not res.used_fallback
    assert res.stage == "failed"


# --------------------------------------------------------------------------
# value_validator
# --------------------------------------------------------------------------
def test_value_validator_clamps():
    def clamp01(d):
        return {k: (min(1.0, max(0.0, v)) if k.startswith("p_") else v) for k, v in d.items()}

    raw = json.dumps(_good_forecast(p_solve=1.4, p_first_attempt_correct=-0.3))
    res = extract_forecast(raw, DIMS, value_validator=clamp01)
    assert res.predictions["p_solve"] == pytest.approx(1.0)
    assert res.predictions["p_first_attempt_correct"] == pytest.approx(0.0)


def test_fallback_output_also_validated():
    def clamp01(d):
        return {k: (min(1.0, max(0.0, v)) if k.startswith("p_") else v) for k, v in d.items()}

    llm = FakeLLM(json.dumps({"broken": False, "forecast": _good_forecast(p_solve=2.0)}))
    res = extract_forecast("garbage", DIMS, llm_client=llm, value_validator=clamp01)
    assert res.predictions["p_solve"] == pytest.approx(1.0)
