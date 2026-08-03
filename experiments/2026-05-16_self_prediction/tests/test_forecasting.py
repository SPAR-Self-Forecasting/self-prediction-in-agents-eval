"""Tests for forecasting.py -- self/cross forecast prompts and elicitation (sec 4, 5)."""

import importlib.util
import json
from pathlib import Path

import pytest

import config
import forecasting
from dataset import mock_tasks

TASK = mock_tasks()[0]
QWEN8B = config.model_by_id("qwen/qwen3-8b")          # hybrid
QWEN_CODER = config.model_by_id("qwen/qwen3-coder")    # non-reasoning


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
    prompt = forecasting.build_self_prompt(TASK, QWEN_CODER, "no_thinking")
    assert config.THINKING_INSTRUCTION not in prompt


def test_self_prompt_thinking_adds_instruction_for_non_reasoning_model():
    prompt = forecasting.build_self_prompt(TASK, QWEN_CODER, "thinking")
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
# Agent scaffold in the forecast prompt
# --------------------------------------------------------------------------
def test_prompts_include_agent_system_prompt_and_instructions():
    # Both forecast kinds must show the verbatim scaffold: the system prompt, the
    # task message, and the stated rules the n_constraint_violations dim scores.
    markers = [
        "You are a helpful assistant that can interact with a computer shell",
        "DO NOT MODIFY: Tests, configuration files",
        "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        "Do NOT commit your changes.",
    ]
    for build in (forecasting.build_self_prompt, forecasting.build_cross_prompt):
        prompt = build(TASK, QWEN8B, "no_thinking")
        for marker in markers:
            assert marker in prompt
        # the real task is substituted into the scaffold's {{task}} slot
        assert TASK.problem_statement in prompt
        assert "{{TASK}}" not in prompt and "{{task}}" not in prompt


def test_scaffold_does_not_name_executor_model():
    # The yaml's default model_name must not leak into the (anonymous) scaffold.
    scaffold = forecasting.build_agent_scaffold(TASK)
    assert "claude" not in scaffold.lower()
    assert "model_name" not in scaffold


def test_vendored_scaffold_matches_installed_swebench_yaml():
    # Drift guard: when mini-swe-agent + yaml are importable, the vendored
    # prompts/agent_scaffold.txt must still contain the live swebench.yaml
    # system_template and instance_template verbatim.
    yaml = pytest.importorskip("yaml")
    spec = importlib.util.find_spec("minisweagent")  # locate without importing (Windows banner)
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("mini-swe-agent not installed")
    yaml_path = Path(spec.submodule_search_locations[0]) / "config" / "benchmarks" / "swebench.yaml"
    if not yaml_path.exists():
        pytest.skip("swebench.yaml not found in installed package")

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["agent"]
    scaffold = (forecasting.PROMPTS_DIR / "agent_scaffold.txt").read_text(encoding="utf-8")
    assert data["system_template"].strip() in scaffold
    assert data["instance_template"].replace("{{task}}", "{{TASK}}").strip() in scaffold


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
    forecasting.elicit_forecast(llm, QWEN_CODER, TASK, "thinking", "self")
    assert llm.calls[0]["kwargs"].get("reasoning") is None


def test_elicit_anonymous_forecast():
    llm = FakeLLM([_good_forecast_json()])
    fc = forecasting.elicit_forecast(llm, QWEN8B, TASK, "no_thinking", "anonymous")
    assert fc.kind == "anonymous"
    assert "An AI coding agent" in llm.calls[0]["prompt"]


def test_the_retired_cross_kind_is_rejected():
    """"cross" once meant anonymous; it must not quietly resolve to either kind."""
    llm = FakeLLM([_good_forecast_json()])
    with pytest.raises(ValueError, match="kind"):
        forecasting.elicit_forecast(llm, QWEN8B, TASK, "no_thinking", "cross")


def test_elicit_broken_forecast_flagged():
    # Single-pass recovery (sec 10.9): one elicit call + one fallback call; both
    # fail -> the sample is flagged broken with empty predictions.
    llm = FakeLLM(["total garbage no json", json.dumps({"broken": True})])
    fc = forecasting.elicit_forecast(llm, QWEN8B, TASK, "no_thinking", "self")
    assert fc.broken
    assert fc.predictions == {}
    assert len(llm.calls) == 2


def test_elicit_recovers_via_fallback():
    # The raw reply fails the programmatic parse; the single cheap-model fallback
    # call recovers the forecast.
    good = json.loads(_good_forecast_json())
    llm = FakeLLM(["garbage, no parseable forecast",
                   json.dumps({"broken": False, "forecast": good})])
    fc = forecasting.elicit_forecast(llm, QWEN8B, TASK, "no_thinking", "self")
    assert not fc.broken
    assert set(fc.predictions) == set(config.DIMENSION_NAMES)
    assert len(llm.calls) == 2


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


# --------------------------------------------------------------------------
# Named cross forecast (the target is told to the peer)
# --------------------------------------------------------------------------
GLM5 = config.model_by_id("z-ai/glm-5")     # the designated cross-predictor


class FakeChatLLM(FakeLLM):
    """FakeLLM plus the multi-turn entry point the primed self-forecast uses."""

    def __init__(self, contents):
        super().__init__(contents)
        self.chats = []

    def chat(self, model, messages, **kwargs):
        self.chats.append({"model": model, "messages": messages, "kwargs": kwargs})
        return type("R", (), {"content": self.contents.pop(0)})()


def test_named_cross_prompt_names_the_target():
    prompt = forecasting.build_cross_prompt(TASK, GLM5, "no_thinking", target=QWEN8B)
    assert QWEN8B.name in prompt
    assert "{{MODEL}}" not in prompt


def test_named_cross_prompt_names_the_target_not_the_forecaster():
    """The subject is who is being predicted, never who is doing the predicting."""
    prompt = forecasting.build_cross_prompt(TASK, GLM5, "no_thinking", target=QWEN8B)
    assert GLM5.name not in prompt


def test_anonymous_prompt_survives_the_named_edit():
    """The RQ1 baseline must stay runnable after cross_forecast.txt was renamed."""
    prompt = forecasting.build_cross_prompt(TASK, GLM5, "no_thinking")
    assert "An AI coding agent" in prompt
    assert QWEN8B.name not in prompt
    assert "{{MODEL}}" not in prompt


def test_both_cross_variants_ask_for_every_dimension():
    """Naming the target must not have quietly dropped part of the battery."""
    for target in (None, QWEN8B):
        prompt = forecasting.build_cross_prompt(TASK, GLM5, "no_thinking", target=target)
        for dim in config.DIMENSION_NAMES:
            assert dim in prompt, (target, dim)
        assert TASK.problem_statement in prompt
        assert "{{AGENT_SCAFFOLD}}" not in prompt


def test_named_and_anonymous_cross_prompts_actually_differ():
    anon = forecasting.build_cross_prompt(TASK, GLM5, "no_thinking")
    named = forecasting.build_cross_prompt(TASK, GLM5, "no_thinking", target=QWEN8B)
    assert anon != named


def test_elicit_cross_named_records_the_target():
    llm = FakeLLM([_good_forecast_json()])
    fc = forecasting.elicit_forecast(
        llm, GLM5, TASK, "no_thinking", "cross_named", target=QWEN8B)
    assert fc.kind == "cross_named"
    assert fc.model == GLM5.openrouter_id        # forecaster
    assert fc.target == QWEN8B.openrouter_id     # subject
    assert QWEN8B.name in llm.calls[0]["prompt"]


def test_cross_named_without_a_target_is_a_loud_failure():
    """Silently falling back to anonymous would mislabel the estimand."""
    llm = FakeLLM([_good_forecast_json()])
    with pytest.raises(ValueError, match="target"):
        forecasting.elicit_forecast(llm, GLM5, TASK, "no_thinking", "cross_named")


def test_anonymous_forecast_refuses_a_target():
    llm = FakeLLM([_good_forecast_json()])
    with pytest.raises(ValueError, match="target"):
        forecasting.elicit_forecast(
            llm, GLM5, TASK, "no_thinking", "anonymous", target=QWEN8B)


# --------------------------------------------------------------------------
# Identity-primed self forecast
# --------------------------------------------------------------------------
def test_identity_question_is_the_shared_template():
    """forecasting and check_model_identity must ask the identical question."""
    import check_model_identity as cmi
    assert forecasting.identity_question() == cmi.load_identity_question()
    assert "what model you are" in forecasting.identity_question().lower()


def test_primed_messages_are_three_turns_in_order():
    msgs = forecasting.build_primed_self_messages(TASK, QWEN8B, "no_thinking", "I am Qwen3-8B.")
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0]["content"] == forecasting.identity_question()
    assert msgs[1]["content"] == "I am Qwen3-8B."


def test_primed_forecast_request_is_byte_identical_to_the_unprimed_one():
    """Priming is the only treatment -- a reworded prompt would confound it."""
    msgs = forecasting.build_primed_self_messages(TASK, QWEN8B, "thinking", "I am Qwen3-8B.")
    assert msgs[-1]["content"] == forecasting.build_self_prompt(TASK, QWEN8B, "thinking")


def test_elicit_primed_self_asks_identity_first_then_forecasts():
    llm = FakeChatLLM(["I am Qwen3-8B, a Qwen model.", _good_forecast_json()])
    fc = forecasting.elicit_forecast(llm, QWEN8B, TASK, "no_thinking", "self_primed")
    assert fc.kind == "self_primed"
    assert fc.identity_reply == "I am Qwen3-8B, a Qwen model."
    assert not fc.broken
    assert set(fc.predictions) == set(config.DIMENSION_NAMES)
    # one single-turn call for the identity question, one multi-turn call for the forecast
    assert len(llm.calls) == 1 and len(llm.chats) == 1
    assert llm.calls[0]["prompt"] == forecasting.identity_question()
    assert llm.chats[0]["messages"][1]["content"] == "I am Qwen3-8B, a Qwen model."


def test_primed_forecast_is_extracted_from_the_second_reply():
    """The forecast must come from the forecast turn, not the identity turn."""
    llm = FakeChatLLM(["I am Qwen3-8B.", _good_forecast_json(p_solve=0.11)])
    fc = forecasting.elicit_forecast(llm, QWEN8B, TASK, "no_thinking", "self_primed")
    assert fc.predictions["p_solve"] == pytest.approx(0.11)
    assert "I am Qwen3-8B." not in fc.raw_response


def test_priming_turn_uses_the_same_reasoning_and_temperature_as_the_forecast():
    llm = FakeChatLLM(["I am Qwen3-8B.", _good_forecast_json()])
    forecasting.elicit_forecast(llm, QWEN8B, TASK, "thinking", "self_primed")
    assert llm.calls[0]["kwargs"]["reasoning"] == {"enabled": True}
    assert llm.chats[0]["kwargs"]["reasoning"] == {"enabled": True}
    assert llm.calls[0]["kwargs"]["temperature"] == pytest.approx(config.FORECAST_TEMPERATURE)


def test_empty_identity_reply_is_recorded_not_dropped():
    """A model that says nothing was still primed; the record must show it."""
    llm = FakeChatLLM(["", _good_forecast_json()])
    fc = forecasting.elicit_forecast(llm, QWEN8B, TASK, "no_thinking", "self_primed")
    assert fc.identity_reply == ""
    assert fc.kind == "self_primed"


def test_unprimed_self_forecast_carries_no_identity_reply():
    llm = FakeLLM([_good_forecast_json()])
    fc = forecasting.elicit_forecast(llm, QWEN8B, TASK, "no_thinking", "self")
    assert fc.identity_reply is None
    assert fc.target is None
