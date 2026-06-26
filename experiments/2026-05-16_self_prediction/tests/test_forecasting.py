"""Tests for forecasting.py -- self/cross forecast prompts and elicitation (sec 4, 5)."""

import json

import pytest

import config
import forecasting
from dataset import mock_tasks

TASK = mock_tasks()[0]
QWEN8B = config.model_by_id("qwen/qwen3-8b")          # hybrid
GPT41 = config.model_by_id("openai/gpt-4.1-mini")      # non-reasoning


def _good_forecast_json(**ov):
    # Build from the live dimension battery so the fixture never drifts from config.
    defaults = {
        "p_solve": 0.6, "p_first_attempt_correct": 0.4,
        "documentation": 0.3, "defensive_code": 0.2, "express_uncertainty": 0.5,
        "avg_tokens_step": 300, "total_expected_steps": 35, "explore_edit_ratio": 0.5,
        "mean_action_block_size": 1.5, "n_constraint_violations": 0,
    }
    f = {name: defaults.get(name, 0.5) for name in config.DIMENSION_NAMES}
    f.update(ov)
    return json.dumps(f)


class FakeLLM:
    """Records calls; replays scripted response contents."""

    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = []

    def complete(self, model, prompt, **kwargs):
        self.calls.append({"model": model, "prompt": prompt, "kwargs": kwargs})
        return type("R", (), {"content": self.contents.pop(0)})()


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------
def test_self_prompt_contains_task_and_all_dimensions():
    prompt = forecasting.build_self_prompt(TASK, QWEN8B, "no_thinking")
    assert TASK.problem_statement in prompt
    for dim in config.DIMENSION_NAMES:
        assert dim in prompt


def test_self_prompt_no_thinking_omits_instruction():
    prompt = forecasting.build_self_prompt(TASK, GPT41, "no_thinking")
    assert config.THINKING_INSTRUCTION not in prompt


def test_self_prompt_thinking_adds_instruction_for_non_reasoning_model():
    prompt = forecasting.build_self_prompt(TASK, GPT41, "thinking")
    assert config.THINKING_INSTRUCTION in prompt


def test_self_prompt_hybrid_identical_across_conditions():
    # Hybrids vary only the API reasoning parameter -- the prompt is unchanged.
    a = forecasting.build_self_prompt(TASK, QWEN8B, "no_thinking")
    b = forecasting.build_self_prompt(TASK, QWEN8B, "thinking")
    assert a == b


def test_cross_prompt_is_anonymous():
    prompt = forecasting.build_cross_prompt(TASK, QWEN8B, "no_thinking")
    assert "An AI coding agent" in prompt
    # the target model is never named in the cross prompt (sec 5)
    assert QWEN8B.name not in prompt
    assert QWEN8B.openrouter_id not in prompt


# --------------------------------------------------------------------------
# value validator
# --------------------------------------------------------------------------
def test_value_validator_clamps_probabilities_and_counts():
    cleaned = forecasting.forecast_value_validator({
        "p_solve": 1.7, "explore_edit_ratio": -0.2,
        "total_expected_steps": -5.0, "avg_tokens_step": 250.0,
    })
    assert cleaned["p_solve"] == 1.0
    assert cleaned["explore_edit_ratio"] == 0.0
    assert cleaned["total_expected_steps"] == 0.0
    assert cleaned["avg_tokens_step"] == 250.0


# --------------------------------------------------------------------------
# elicit_forecast
# --------------------------------------------------------------------------
def test_elicit_self_forecast_happy_path():
    llm = FakeLLM([_good_forecast_json()])
    fc = forecasting.elicit_forecast(llm, QWEN8B, TASK, "no_thinking", "self")
    assert fc.kind == "self" and fc.condition == "no_thinking"
    assert fc.model == "qwen/qwen3-8b"
    assert fc.task_id == TASK.instance_id
    assert not fc.broken
    assert set(fc.predictions) == set(config.DIMENSION_NAMES)
    assert llm.calls[0]["kwargs"]["reasoning"] == {"enabled": False}  # hybrid no-thinking


def test_elicit_thinking_condition_threads_reasoning_on():
    llm = FakeLLM([_good_forecast_json()])
    forecasting.elicit_forecast(llm, QWEN8B, TASK, "thinking", "self")
    assert llm.calls[0]["kwargs"]["reasoning"] == {"enabled": True}


def test_elicit_non_reasoning_model_sends_no_reasoning_param():
    llm = FakeLLM([_good_forecast_json()])
    forecasting.elicit_forecast(llm, GPT41, TASK, "thinking", "self")
    assert llm.calls[0]["kwargs"].get("reasoning") is None


def test_elicit_cross_forecast():
    llm = FakeLLM([_good_forecast_json()])
    fc = forecasting.elicit_forecast(llm, QWEN8B, TASK, "no_thinking", "cross")
    assert fc.kind == "cross"
    assert "An AI coding agent" in llm.calls[0]["prompt"]


def test_elicit_broken_forecast_flagged():
    # Multi-attempt extraction (Layer B, sec 10.9) + multi-elicit (Layer C):
    # garbage every time -> ELICIT_ATTEMPTS rounds of (1 elicit + EXTRACTOR_FALLBACK_ATTEMPTS fallback) calls.
    n_per_round = 1 + config.EXTRACTOR_FALLBACK_ATTEMPTS
    expected_calls = config.ELICIT_ATTEMPTS * n_per_round
    contents: list[str] = []
    for _ in range(config.ELICIT_ATTEMPTS):
        contents.append("total garbage no json")  # elicit call
        contents.extend([json.dumps({"broken": True})] * config.EXTRACTOR_FALLBACK_ATTEMPTS)
    llm = FakeLLM(contents)
    fc = forecasting.elicit_forecast(llm, QWEN8B, TASK, "no_thinking", "self")
    assert fc.broken
    assert fc.predictions == {}
    assert len(llm.calls) == expected_calls


def test_elicit_recovers_on_second_attempt():
    # First elicit returns garbage AND its fallbacks all declare broken; the
    # second elicit (Layer C) returns valid JSON and parses programmatically.
    contents: list[str] = [
        "garbage",
        *([json.dumps({"broken": True})] * config.EXTRACTOR_FALLBACK_ATTEMPTS),
        _good_forecast_json(),
    ]
    llm = FakeLLM(contents)
    fc = forecasting.elicit_forecast(llm, QWEN8B, TASK, "no_thinking", "self")
    assert not fc.broken
    assert set(fc.predictions) == set(config.DIMENSION_NAMES)


def test_elicit_passes_temperature_for_diversity():
    # Multi-sample diversity (sec 9.6): forecast calls must request a non-zero
    # temperature so deterministic-mode forecasters produce distinct samples.
    llm = FakeLLM([_good_forecast_json()])
    forecasting.elicit_forecast(llm, QWEN8B, TASK, "no_thinking", "self", sample_idx=2)
    assert llm.calls[0]["kwargs"].get("temperature") == pytest.approx(config.FORECAST_TEMPERATURE)


def test_elicit_records_sample_idx():
    llm = FakeLLM([_good_forecast_json()])
    fc = forecasting.elicit_forecast(llm, QWEN8B, TASK, "no_thinking", "self", sample_idx=3)
    assert fc.sample_idx == 3


def test_elicit_rejects_bad_kind():
    llm = FakeLLM([_good_forecast_json()])
    with pytest.raises(ValueError):
        forecasting.elicit_forecast(llm, QWEN8B, TASK, "no_thinking", "sideways")
