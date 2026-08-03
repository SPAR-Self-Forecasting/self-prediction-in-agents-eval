"""Forecast elicitation -- self and cross, two conditions (EVAL_SPEC_V1.md sec 4, 5).

Builds the self / cross forecast prompts from the templates in ``prompts/``,
calls the forecaster through the OpenRouter client with the per-model reasoning
control of sec 10.4, and converts the response to a :class:`Forecast` via the
two-stage extractor (sec 10.9).

- **Self** -- naturalistic: "You are about to attempt ... predict your attempt."
- **Self, primed** -- the same self prompt, verbatim, but preceded by the identity
  question and the model's own answer to it, so the forecast is made by a model
  that has just said out loud which model it thinks it is.
- **Anonymous** -- the peer forecast in which the subject is never named (sec 5);
  one set is produced by the designated cross-predictor and reused across subjects.
- **Cross, named** -- the target *is* named. This makes the forecast
  target-specific (so it can no longer be shared across targets) and measures
  something narrower than anonymous cross: a peer told it is predicting a small
  model can price that reputation without any behavioural insight.
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


def identity_question() -> str:
    """The identity question, shared with ``check_model_identity.py``.

    Read from ``prompts/identity_check.txt`` so the priming turn and the
    standalone capability check provably ask the same thing.
    """
    return _load_template("identity_check.txt").strip()


def build_primed_self_messages(
    task: Task,
    model: config.ModelSpec,
    condition: str,
    identity_reply: str,
) -> list[dict]:
    """Message list for the identity-primed self-forecast.

    Three turns: the identity question, the model's own answer to it, then the
    unmodified ``self_forecast.txt`` prompt. The forecast text is byte-identical
    to the unprimed condition -- the only difference is that the model has just
    stated who it thinks it is, in the same context. That isolates the priming as
    the treatment instead of confounding it with a reworded request.
    """
    return [
        {"role": "user", "content": identity_question()},
        {"role": "assistant", "content": identity_reply},
        {"role": "user", "content": build_self_prompt(task, model, condition)},
    ]


def build_cross_prompt(
    task: Task,
    cross_model: config.ModelSpec,
    condition: str,
    *,
    target: config.ModelSpec | None = None,
) -> str:
    """A cross-forecast prompt -- named when ``target`` is given, else anonymous.

    ``cross_model`` is the designated cross-predictor; it drives the thinking
    control only. ``target`` is who is being forecast:

    * ``target=None`` renders ``cross_forecast_anonymous.txt`` (sec 5) -- the
      target is never named, and one forecast per task serves every target.
    * ``target=spec`` renders ``cross_forecast.txt`` with ``{{MODEL}}`` filled in,
      which makes the forecast target-specific: the peer now knows whose
      behaviour it is predicting, so a forecast must be elicited per target.

    The two are different estimands, not two spellings of one. Anonymous cross is
    the RQ1 baseline ("can a model predict itself better than an outsider who
    doesn't know who it is?"); named cross prices what the identity alone is worth,
    since a peer told it is forecasting a weak model can lean on reputation.
    """
    name = "cross_forecast.txt" if target is not None else "cross_forecast_anonymous.txt"
    template = _load_template(name)
    rendered = template.replace(
        "{{CONDITION_INSTRUCTION}}", config.forecast_condition_suffix(cross_model, condition))
    if target is not None:
        rendered = rendered.replace("{{MODEL}}", target.name)
    # AGENT_SCAFFOLD last, so task-derived text is never re-scanned for placeholders.
    return rendered.replace("{{AGENT_SCAFFOLD}}", build_agent_scaffold(task))


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
    target: config.ModelSpec | None = None,
) -> Forecast:
    """Elicit one forecast sample and return it as a :class:`Forecast`.

    ``forecaster`` is the model doing the predicting (the target itself for a
    self forecast; the designated cross-predictor for a cross forecast).
    ``sample_idx`` identifies this sample within its (forecaster, task,
    condition) cell; the caller produces ``config.N_FORECAST_SAMPLES`` per cell
    (multi-sample amendment, sec 9.6). ``temperature`` defaults to
    ``config.FORECAST_TEMPERATURE`` -- non-zero so the N samples for a
    deterministic-mode forecaster carry real diversity.

    Four ``kind`` values:

    * ``self``        -- ``self_forecast.txt``, one call (sec 4).
    * ``self_primed`` -- the identity question, the model's own answer, then the
      byte-identical ``self_forecast.txt`` prompt in the same context. Two calls;
      the priming answer is kept on the returned Forecast.
    * ``anonymous``   -- the anonymous peer forecast, subject never named (sec 5).
    * ``cross_named`` -- the peer forecast with ``target`` named in the prompt;
      requires ``target``.

    Single-pass recovery (sec 10.9, simplified for the MVP): elicit the
    forecaster once, then convert the reply to numbers via the two-step
    extractor -- a programmatic JSON parse, then one cheap-model fallback call.
    If both fail the sample is flagged ``broken`` and treated as missing data
    (sec 9.6); with N samples per cell, one broken sample is absorbed by the
    others.
    """
    if kind not in ("self", "self_primed", "anonymous", "cross_named"):
        raise ValueError(
            "kind must be one of 'self', 'self_primed', 'anonymous', 'cross_named', "
            f"got {kind!r}"
        )
    if (kind == "cross_named") != (target is not None):
        raise ValueError(
            f"target is required for kind='cross_named' and forbidden otherwise; "
            f"got kind={kind!r}, target={target!r}"
        )

    reasoning = config.forecast_reasoning(forecaster, condition)
    temp = config.FORECAST_TEMPERATURE if temperature is None else temperature

    identity_reply: str | None = None
    if kind == "self_primed":
        # The priming turn runs at the same temperature as the forecast: it is part
        # of this sample's context, so it should vary across samples the way the
        # forecast does rather than being pinned to one canned answer.
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
        if kind == "self":
            prompt = build_self_prompt(task, forecaster, condition)
        else:
            prompt = build_cross_prompt(task, forecaster, condition, target=target)
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
        target=target.openrouter_id if target is not None else None,
        identity_reply=identity_reply,
    )
