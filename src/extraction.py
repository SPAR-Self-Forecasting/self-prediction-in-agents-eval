"""Two-stage forecast-JSON extractor (EVAL_SPEC_V1.md sec 10.9).

Every forecast response -- self and cross, both conditions -- is converted to a
numeric forecast object here:

1. **Programmatic parse.** Scan the raw response for the JSON object that
   carries all required dimension keys, tolerant of surrounding prose, leaked
   reasoning text, and markdown code fences. When several brace-objects appear
   (reasoning leakage), the *trailing* one wins -- the forecast prompt asks the
   model to end its reply with the JSON.
2. **LLM fallback.** If stage 1 fails -- no parseable JSON, a missing dimension,
   or a malformed (string / boolean / non-finite) value -- a single call to a
   cheap model (gpt-4o-mini) either recovers the forecast or returns an explicit
   ``broken`` flag.

Parse failures are never silent: an unrecoverable forecast is flagged ``broken``
and handled as missing data by the caller (sec 9.6).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ExtractionResult:
    """Outcome of two-stage extraction for one forecast response."""

    predictions: dict | None       # {dimension: float} or None when broken
    broken: bool
    used_fallback: bool
    stage: str                     # "programmatic" | "llm_fallback" | "failed"


def find_json_objects(text: str) -> list[str]:
    """Return the top-level ``{...}`` substrings of ``text``, in source order.

    Brace counting is string-literal aware, so braces inside JSON string values
    do not corrupt the balance.
    """
    objs: list[str] = []
    depth = 0
    start: int | None = None
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(text[start : i + 1])
                    start = None
    return objs


def _coerce_numeric(value) -> float | None:
    """Return ``value`` as a finite float, or None if it is not a well-typed number.

    Booleans are JSON numbers in Python's type lattice but are *not* valid
    forecast values -- rejected explicitly so stage 1 routes them to the fallback.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    return None


def _parse_candidate(
    snippet: str,
    dimension_names: list[str],
    value_validator: Callable[[dict], dict | None] | None,
) -> dict | None:
    """Parse one JSON snippet into a validated forecast, or None if it does not qualify."""
    try:
        obj = json.loads(snippet)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    coerced: dict[str, float] = {}
    for dim in dimension_names:
        if dim not in obj:
            return None
        num = _coerce_numeric(obj[dim])
        if num is None:
            return None
        coerced[dim] = num
    if value_validator is not None:
        coerced = value_validator(coerced)
        if coerced is None:
            return None
    return coerced


def _programmatic(
    raw: str,
    dimension_names: list[str],
    value_validator: Callable[[dict], dict | None] | None,
) -> dict | None:
    """Stage 1: recover a forecast from the raw response without an LLM call."""
    candidates = find_json_objects(raw)
    # Prefer the trailing JSON object (the forecast follows any leaked reasoning).
    for snippet in reversed(candidates):
        parsed = _parse_candidate(snippet, dimension_names, value_validator)
        if parsed is not None:
            return parsed
    # Last resort: the whole response, stripped.
    return _parse_candidate(raw.strip(), dimension_names, value_validator)


def build_fallback_prompt(raw: str, dimension_names: list[str], template: str | None) -> str:
    """Build the stage-2 prompt; uses ``template`` (with ``{{RAW_RESPONSE}}``) if given."""
    if template is not None:
        return template.replace("{{RAW_RESPONSE}}", raw)
    keys = ", ".join(dimension_names)
    return (
        "A model was asked to produce a JSON forecast with exactly these numeric "
        f"keys: {keys}.\nIts raw response is below; it may contain reasoning text, "
        "markdown fences, prose, or truncation.\n\n"
        f"[RAW RESPONSE]\n{raw}\n\n[TASK]\n"
        'If a forecast for ALL keys is recoverable, output exactly '
        '{"broken": false, "forecast": {<each key with a numeric value>}}.\n'
        'Otherwise output exactly {"broken": true}.\nOutput ONLY the JSON object.'
    )


def _parse_fallback(
    content: str,
    dimension_names: list[str],
    value_validator: Callable[[dict], dict | None] | None,
) -> tuple[bool, dict | None]:
    """Interpret the fallback model's reply. Returns (broken, predictions)."""
    for snippet in reversed(find_json_objects(content)):
        try:
            obj = json.loads(snippet)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict) or "broken" not in obj:
            continue
        if obj.get("broken") is True:
            return True, None
        forecast = obj.get("forecast")
        if isinstance(forecast, dict):
            parsed = _parse_candidate(json.dumps(forecast), dimension_names, value_validator)
            if parsed is not None:
                return False, parsed
        return True, None  # claimed not broken but no recoverable forecast
    return True, None  # no interpretable fallback reply


def extract_forecast(
    raw: str,
    dimension_names: list[str],
    *,
    llm_client=None,
    fallback_model: str = "openai/gpt-4o-mini",
    fallback_template: str | None = None,
    value_validator: Callable[[dict], dict | None] | None = None,
    fallback_attempts: int = 1,
) -> ExtractionResult:
    """Convert a raw forecast response to an :class:`ExtractionResult` (sec 10.9).

    ``fallback_attempts``: up to this many independent calls to ``fallback_model``
    before declaring the sample broken (default 1 = a single fallback call). The
    fallback prompt is identical across attempts; diversity comes from the
    fallback model's default temperature (server-side, ~1.0).
    """
    programmatic = _programmatic(raw, dimension_names, value_validator)
    if programmatic is not None:
        return ExtractionResult(programmatic, broken=False, used_fallback=False, stage="programmatic")

    if llm_client is None:
        return ExtractionResult(None, broken=True, used_fallback=False, stage="failed")

    prompt = build_fallback_prompt(raw, dimension_names, fallback_template)
    for _ in range(max(1, fallback_attempts)):
        response = llm_client.complete(fallback_model, prompt)
        broken, predictions = _parse_fallback(
            getattr(response, "content", "") or "", dimension_names, value_validator
        )
        if not broken:
            return ExtractionResult(predictions, broken=False, used_fallback=True, stage="llm_fallback")
    return ExtractionResult(None, broken=True, used_fallback=True, stage="llm_fallback")
