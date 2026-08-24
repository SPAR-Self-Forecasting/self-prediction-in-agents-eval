"""Forecast elicitation -- self forecasts, two conditions (EVAL_SPEC_V1.md sec 4).

Builds the self forecast prompts from the templates in ``prompts/``, calls the
forecaster through the OpenRouter client with the per-model reasoning control
of sec 10.4, and converts the response to a :class:`Forecast` via the two-stage
extractor (sec 10.9).

- **Self** -- naturalistic: "You are about to attempt ... predict your attempt."
- **Self, primed** -- the same self prompt, verbatim, but preceded by the identity
  question and the model's own answer to it.
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
    """The verbatim mini-swe-agent scaffold the executor receives for ``task``."""
    return _load_template("agent_scaffold.txt").replace("{{TASK}}", task.problem_statement)


def build_self_prompt(task: Task, model: config.ModelSpec, condition: str) -> str:
    """The naturalistic self-forecast prompt for ``model`` in ``condition`` (sec 4)."""
    template = _load_template("self_forecast.txt")
    return (
        template
        .replace("{{CONDITION_INSTRUCTION}}", config.forecast_condition_suffix(model, condition))
        .replace("{{AGENT_SCAFFOLD}}", build_agent_scaffold(task))
    )


def identity_question() -> str:
    """The identity question, shared with ``check_model_identity.py``."""
    return _load_template("identity_check.txt").strip()


def build_primed_self_messages(
    task: Task,
    model: config.ModelSpec,
    condition: str,
    identity_reply: str,
) -> list[dict]:
    """Message list for the identity-primed self-forecast."""
    return [
        {"role": "user", "content": identity_question()},
        {"role": "assistant", "content": identity_reply},
        {"role": "user", "content": build_self_prompt(task, model, condition)},
    ]


# --------------------------------------------------------------------------
# Forecast value validation (passed to the two-stage extractor)
# --------------------------------------------------------------------------
def forecast_value_validator(predictions: dict) -> dict:
    """Normalise extracted forecast values to their valid ranges."""
    cleaned: dict[str, float] = {}
    for name, value in predictions.items():
        dim = config.DIMENSION_BY_NAME.get(name)
        if dim is not None and (dim.is_binary or name == "explore_edit_ratio"):
            cleaned[name] = min(1.0, max(0.0, float(value)))
        else:
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

    ``kind`` must be ``self`` or ``self_primed``.
    """
    if kind not in ("self", "self_primed"):
        raise ValueError(f"kind must be 'self' or 'self_primed', got {kind!r}")

    reasoning = config.forecast_reasoning(forecaster, condition)
    temp = config.FORECAST_TEMPERATURE if temperature is None else temperature

    identity_reply: str | None = None
    if kind == "self_primed":
        identity_reply = llm_client.complete(
            forecaster.openrouter_id, identity_question(),
            reasoning=reasoning, temperature=temp,
        ).content or ""
        response = llm_client.chat(
            forecaster.openrouter_id,
            build_primed_self_messages(task, forecaster, condition, identity_reply),
            reasoning=reasoning, temperature=temp,
        )
    else:
        prompt = build_self_prompt(task, forecaster, condition)
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
        identity_reply=identity_reply,
    )
