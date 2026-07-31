"""Response extraction for the {"severity": 0-5 int, "action": str} schema.

Reuses ``src.extraction.find_json_objects`` (which already returns candidates
in document order; we scan from the LAST object backwards -- kept from exp1 as
free robustness even though no executor emits chain-of-thought text before its
JSON anymore). The exp1 ``extract_forecast`` itself is numeric-per-key and
cannot carry the free-text ``action`` field, hence this thin adapter
(build decision Q5a -- mechanism identical, key list adapted).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.extraction import find_json_objects

from . import config


@dataclass(frozen=True)
class PredictionResult:
    severity: int | None
    action: str | None
    broken: bool
    stage: str  # "programmatic" | "llm_fallback" | "failed"


def _coerce_severity(value) -> int | None:
    """Accept 3, 3.0, "3"; reject 2.5, "high", None, out-of-range."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        if float(value) != int(value):
            return None
        iv = int(value)
        if 0 <= iv <= 5:
            return iv
    return None


def _parse_candidate(text: str) -> tuple[int, str] | None:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    severity = _coerce_severity(obj.get("severity"))
    action = obj.get("action")
    if severity is None or not isinstance(action, str) or not action.strip():
        return None
    return severity, action.strip()


def parse_prediction(raw: str) -> tuple[int, str] | None:
    """Programmatic parse: the LAST valid JSON object wins (exp1 convention)."""
    for candidate in reversed(find_json_objects(raw or "")):
        parsed = _parse_candidate(candidate)
        if parsed is not None:
            return parsed
    return None


def _fallback_prompt(raw: str) -> str:
    template = (config.PROMPTS_DIR / "extraction_fallback.txt").read_text(encoding="utf-8")
    return template.replace("{{RAW_RESPONSE}}", raw or "")


def extract_prediction(
    raw: str,
    *,
    llm_client=None,
    fallback_model: str = config.FALLBACK_MODEL,
) -> PredictionResult:
    """Programmatic parse, then (optionally) one cheap-model fallback call.

    An unrecoverable sample is flagged broken and treated as missing data --
    never silently defaulted (spec §3 step 5).
    """
    parsed = parse_prediction(raw)
    if parsed is not None:
        return PredictionResult(*parsed, broken=False, stage="programmatic")
    if llm_client is None:
        return PredictionResult(None, None, broken=True, stage="failed")

    reply = llm_client.complete(fallback_model, _fallback_prompt(raw))
    for candidate in reversed(find_json_objects(getattr(reply, "content", "") or "")):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("broken") is True:
            break
        pred = obj.get("prediction")
        if isinstance(pred, dict):
            parsed = _parse_candidate(json.dumps(pred))
            if parsed is not None:
                return PredictionResult(*parsed, broken=False, stage="llm_fallback")
    return PredictionResult(None, None, broken=True, stage="llm_fallback")
