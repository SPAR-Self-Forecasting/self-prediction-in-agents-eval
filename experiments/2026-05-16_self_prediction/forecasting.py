"""Forecast elicitation -- self and cross, two conditions (EVAL_SPEC_V1.md sec 4, 5).

Builds the self / cross forecast prompts from the templates in ``prompts/``,
calls the forecaster through the OpenRouter client with the per-model reasoning
control of sec 10.4, and converts the response to a :class:`Forecast` via the
two-stage extractor (sec 10.9).

- **Self** -- naturalistic: "You are about to attempt ... predict your attempt."
- **Cross** -- anonymous: the target model is never named (sec 5); one forecast
  set is produced by the designated cross-predictor and reused across targets.
- **Conditions** -- hybrids vary only the API ``reasoning`` parameter; the 4
  non-reasoning models get the sec 4 step-by-step instruction in the prompt.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import config
from dataset import Task
from src.extraction import extract_forecast
from src.types import Forecast

PROMPTS_DIR = Path(__file__).parent / "prompts"


@lru_cache(maxsize=None)
def _load_template(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------
def build_agent_scaffold(task: Task) -> str:
    """The verbatim mini-swe-agent scaffold the executor receives for ``task``.

    Renders ``prompts/agent_scaffold.txt`` -- a vendored, verbatim copy of the
    ``swebench.yaml`` ``system_template`` + ``instance_template`` the rollout
    runs under (rollout.py ``-c swebench.yaml``) -- substituting the real
    ``problem_statement`` into the ``{{task}}`` slot exactly as the agent sees
    it. The yaml's executor ``model_name`` is deliberately not part of this text,
    so the scaffold is safe to show in the anonymous cross prompt (sec 5).

    Vendored rather than read from the installed package so forecasting needs no
    ``minisweagent``/``yaml`` at import time; ``test_forecasting`` asserts the
    vendored copy still matches the installed ``swebench.yaml`` (drift guard).
    """
    return _load_template("agent_scaffold.txt").replace("{{TASK}}", task.problem_statement)


def build_self_prompt(task: Task, model: config.ModelSpec, condition: str) -> str:
    """The naturalistic self-forecast prompt for ``model`` in ``condition`` (sec 4)."""
    template = _load_template("self_forecast.txt")
    # AGENT_SCAFFOLD is substituted last so task-derived text it carries is never
    # re-scanned for other placeholders.
    return (
        template
        .replace("{{CONDITION_INSTRUCTION}}", config.forecast_condition_suffix(model, condition))
        .replace("{{AGENT_SCAFFOLD}}", build_agent_scaffold(task))
    )


def build_cross_prompt(task: Task, cross_model: config.ModelSpec, condition: str) -> str:
    """The anonymous cross-forecast prompt (sec 5).

    ``cross_model`` is the designated cross-predictor; it drives the thinking
    control only -- the target identity is never written into the prompt.
    """
    template = _load_template("cross_forecast.txt")
    return (
        template
        .replace("{{CONDITION_INSTRUCTION}}", config.forecast_condition_suffix(cross_model, condition))
        .replace("{{AGENT_SCAFFOLD}}", build_agent_scaffold(task))
    )


# --------------------------------------------------------------------------
# Forecast value validation (passed to the two-stage extractor)
# --------------------------------------------------------------------------
def forecast_value_validator(predictions: dict) -> dict:
    """Normalise extracted forecast values to their valid ranges.

    Binary-dimension forecasts and ``explore_edit_ratio`` are clamped to [0, 1];
    the count dimensions (``avg_tokens_step``, ``total_expected_steps``) are
    clamped to be non-negative. Never rejects -- always returns a cleaned dict.
    """
    cleaned: dict[str, float] = {}
    for name, value in predictions.items():
        dim = config.DIMENSION_BY_NAME.get(name)
        if dim is not None and (dim.is_binary or name == "explore_edit_ratio"):
            cleaned[name] = min(1.0, max(0.0, float(value)))
        else:  # non-negative count dimensions
            cleaned[name] = max(0.0, float(value))
    return cleaned


# --------------------------------------------------------------------------
# Elicitation
# --------------------------------------------------------------------------
def elicit_forecast(
    llm_client,
    forecaster: config.ModelSpec,
    task: Task,
    condition: str,
    kind: str,
    *,
    fallback_model: str = config.FALLBACK_MODEL,
    fallback_template: str | None = None,
    sample_idx: int = 0,
    temperature: float | None = None,
) -> Forecast:
    """Elicit one forecast sample and return it as a :class:`Forecast`.

    ``forecaster`` is the model doing the predicting (the target itself for a
    self forecast; the designated cross-predictor for a cross forecast).
    ``sample_idx`` identifies this sample within its (forecaster, task,
    condition) cell; the caller produces ``config.N_FORECAST_SAMPLES`` per cell
    (multi-sample amendment, sec 9.6). ``temperature`` defaults to
    ``config.FORECAST_TEMPERATURE`` -- non-zero so the N samples for a
    deterministic-mode forecaster carry real diversity.

    Single-pass recovery (sec 10.9, simplified for the MVP): elicit the
    forecaster once, then convert the reply to numbers via the two-step
    extractor -- a programmatic JSON parse, then one cheap-model fallback call.
    If both fail the sample is flagged ``broken`` and treated as missing data
    (sec 9.6); with N samples per cell, one broken sample is absorbed by the
    others.
    """
    if kind == "self":
        prompt = build_self_prompt(task, forecaster, condition)
    elif kind == "cross":
        prompt = build_cross_prompt(task, forecaster, condition)
    else:
        raise ValueError(f"kind must be 'self' or 'cross', got {kind!r}")

    reasoning = config.forecast_reasoning(forecaster, condition)
    temp = config.FORECAST_TEMPERATURE if temperature is None else temperature

    response = llm_client.complete(
        forecaster.openrouter_id, prompt, reasoning=reasoning, temperature=temp,
    )
    result = extract_forecast(
        response.content,
        list(config.DIMENSION_NAMES),
        llm_client=llm_client,
        fallback_model=fallback_model,
        fallback_template=fallback_template,
        value_validator=forecast_value_validator,
    )

    return Forecast(
        model=forecaster.openrouter_id,
        task_id=task.instance_id,
        condition=condition,
        kind=kind,
        predictions=result.predictions or {},
        broken=bool(result.broken),
        raw_response=response.content,
        sample_idx=sample_idx,
    )
