"""LLM-judge ground truth for the 3 judge-based dimensions (EVAL_SPEC_V1.md sec 10.1).

Two judges, one call each per trajectory:

- **Judge 1** reads the full patch diff -> ``documentation`` (dim 3) and
  ``defensive_code`` (dim 4).
- **Judge 2** reads the concatenated reasoning text -> ``express_uncertainty``
  (dim 5).

The judge model is ``gpt-4o-mini`` -- cheap, and not one of the 14 executors, so
no circularity (sec 10.6). Judge outputs are accepted as-is, with no human
validation (sec 10.1). A judge reply that cannot be parsed yields ``None`` for the
affected dimension -- handled as missing data (sec 9.6), never silently defaulted.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import config
from src.extraction import find_json_objects

PROMPTS_DIR = Path(__file__).parent / "prompts"

_TRUE_STRINGS = {"true", "yes", "1"}
_FALSE_STRINGS = {"false", "no", "0"}


@lru_cache(maxsize=None)
def _load_template(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------
def build_patch_judge_prompt(patch_diff: str) -> str:
    """Judge 1 prompt: documentation + defensive_code from the full diff (sec 10.1)."""
    return _load_template("judge_patch.txt").replace("{{PATCH_DIFF}}", patch_diff)


def build_reasoning_judge_prompt(reasoning_text: str) -> str:
    """Judge 2 prompt: express_uncertainty from the concatenated reasoning (sec 10.1)."""
    return _load_template("judge_reasoning.txt").replace("{{REASONING_TEXT}}", reasoning_text)


# --------------------------------------------------------------------------
# Reply parsing
# --------------------------------------------------------------------------
def _first_parsed_object(content: str) -> dict:
    """Parse the last JSON object in a judge reply; {} if none parses."""
    import json

    for snippet in reversed(find_json_objects(content or "")):
        try:
            obj = json.loads(snippet)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return {}


def _extract_bool(parsed: dict, key: str) -> float | None:
    """Pull a boolean field out of a judge reply as 1.0 / 0.0, tolerant of forms.

    Accepts ``{key: {"bool": true, ...}}`` (the sec 10.1 schema) and the flatter
    ``{key: true}``. Returns None if the field is absent or not interpretable.
    """
    value = parsed.get(key)
    if isinstance(value, dict):
        value = value.get("bool")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in _TRUE_STRINGS:
            return 1.0
        if s in _FALSE_STRINGS:
            return 0.0
    return None


# --------------------------------------------------------------------------
# Judges
# --------------------------------------------------------------------------
# Cap judge inputs: a runaway rollout can emit a multi-MB patch/reasoning that
# blows past the judge model's context window (a ~780K-token diff once raised a
# non-retryable HTTP 400 that crashed the whole run). 120k chars is ~30k tokens,
# safely under the 128k limit, and the head of a patch/trace is enough to judge
# documentation/defensive-code/uncertainty. The try/except is a backstop so any
# other LLM failure degrades to missing data, never a crash.
_MAX_JUDGE_CHARS = 120_000


def _truncate(text: str) -> str:
    return text if len(text) <= _MAX_JUDGE_CHARS else text[:_MAX_JUDGE_CHARS] + "\n...[truncated]"


def judge_patch(llm_client, patch_diff: str, judge_model: str = config.JUDGE_MODEL) -> dict:
    """Judge 1: documentation + defensive_code (dims 3-4).

    An empty patch is short-circuited to False/False -- nothing was added, so no
    documentation and no defensive code, and an LLM call would be wasted. Oversized
    input is truncated to the head; a failed call yields None -> missing data
    (sec 9.6), never a crash.
    """
    if not (patch_diff or "").strip():
        return {"documentation": 0.0, "defensive_code": 0.0}
    try:
        response = llm_client.complete(judge_model, build_patch_judge_prompt(_truncate(patch_diff)))
    except Exception:
        return {"documentation": None, "defensive_code": None}
    parsed = _first_parsed_object(response.content)
    return {
        "documentation": _extract_bool(parsed, "documentation"),
        "defensive_code": _extract_bool(parsed, "defensive_code"),
    }


def judge_reasoning(llm_client, reasoning_text: str, judge_model: str = config.JUDGE_MODEL) -> dict:
    """Judge 2: express_uncertainty (dim 5). Oversized input truncated; failed -> None."""
    if not (reasoning_text or "").strip():
        return {"express_uncertainty": 0.0}
    try:
        response = llm_client.complete(judge_model, build_reasoning_judge_prompt(_truncate(reasoning_text)))
    except Exception:
        return {"express_uncertainty": None}
    parsed = _first_parsed_object(response.content)
    return {"express_uncertainty": _extract_bool(parsed, "express_uncertainty")}


def judge_trajectory(
    llm_client, patch_diff: str, reasoning_text: str, judge_model: str = config.JUDGE_MODEL
) -> dict[str, float | None]:
    """Run both judges; return the 3 judge-based dimensions for one rollout."""
    out: dict[str, float | None] = {}
    out.update(judge_patch(llm_client, patch_diff, judge_model))
    out.update(judge_reasoning(llm_client, reasoning_text, judge_model))
    return out
