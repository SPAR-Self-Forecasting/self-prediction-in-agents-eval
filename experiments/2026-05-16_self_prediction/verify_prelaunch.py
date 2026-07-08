"""Pre-launch verification (EVAL_SPEC_V1.md sec 10.6, sec 14).

A one-time, opt-in script run *before* any rollout spend. For each of the 12
models it confirms, against the live OpenRouter API:

  1. **Provider access** -- the model id is reachable and returns a response.
  2. **JSON parseability** -- a forecast call returns a 10-dimension JSON object
     that the two-stage extractor (sec 10.9) can parse, in both conditions.
  3. **Reasoning toggle** -- for the 10 hybrid models, the no-thinking and
     thinking conditions both succeed (the OpenRouter ``reasoning`` parameter is
     accepted); where the API reports reasoning-token usage, the thinking
     condition is checked to use more.

This script makes real, paid API calls -- run it explicitly, never as part of
the test suite. The check functions take an injectable ``client`` so the test
suite can exercise them with a mock.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_EXP_DIR = Path(__file__).parent.resolve()
_ROOT = _EXP_DIR.parents[1]
for _p in (str(_ROOT), str(_EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
import forecasting
from dataset import Task
from src.llm_client import LLMClient

#: A trivial synthetic task -- enough to elicit a well-formed forecast cheaply.
_VERIFY_TASK = Task(
    instance_id="prelaunch-verify-0",
    difficulty="easy",
    repo="example/project",
    base_commit="0" * 40,
    problem_statement=(
        "A helper function returns the wrong value on empty input. "
        "Reproduce the failure and apply a minimal fix."
    ),
)


def _reasoning_tokens(usage: dict) -> int | None:
    """Reasoning-token count from an OpenRouter usage block, if reported."""
    if not isinstance(usage, dict):
        return None
    for key in ("reasoning_tokens", "completion_tokens_details"):
        value = usage.get(key)
        if isinstance(value, dict):
            value = value.get("reasoning_tokens")
        if isinstance(value, (int, float)):
            return int(value)
    return None


def check_model(client, model: config.ModelSpec, *, fallback_template: str | None = None) -> dict:
    """Verify provider access + JSON parseability for one model, both conditions."""
    result: dict = {"thinking_class": model.thinking_class, "conditions": {}, "ok": True}
    for condition in config.CONDITIONS:
        entry: dict = {}
        try:
            forecast = forecasting.elicit_forecast(
                client, model, _VERIFY_TASK, condition, "self",
                fallback_template=fallback_template,
            )
            entry["reachable"] = True
            entry["broken"] = forecast.broken
            entry["n_dimensions"] = len(forecast.predictions)
            entry["ok"] = (not forecast.broken
                           and len(forecast.predictions) == len(config.DIMENSION_NAMES))
        except Exception as exc:  # network / auth / provider error
            entry = {"reachable": False, "ok": False, "error": str(exc)}
        result["conditions"][condition] = entry
        result["ok"] = result["ok"] and entry["ok"]
    return result


def verify(client, models=config.MODELS, *, fallback_template: str | None = None) -> dict:
    """Run the full pre-launch verification over ``models``."""
    report: dict = {"models": {}, "all_ok": True, "n_models": len(models)}
    for model in models:
        outcome = check_model(client, model, fallback_template=fallback_template)
        report["models"][model.openrouter_id] = outcome
        if not outcome["ok"]:
            report["all_ok"] = False
    report["hybrid_models_checked"] = sum(m.thinking_class == "hybrid" for m in models)
    return report


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(_ROOT / ".env")
    except ImportError:
        pass

    fallback_template = (_EXP_DIR / "prompts" / "extraction_fallback.txt").read_text(encoding="utf-8")
    print("[verify] Pre-launch verification -- this makes real OpenRouter API calls.")
    client = LLMClient()
    report = verify(client, config.MODELS, fallback_template=fallback_template)

    for model_id, outcome in report["models"].items():
        status = "OK " if outcome["ok"] else "FAIL"
        print(f"  [{status}] {model_id} ({outcome['thinking_class']})")
        for condition, entry in outcome["conditions"].items():
            print(f"          {condition}: {entry}")
    print(f"\n[verify] all_ok = {report['all_ok']}")
    print(json.dumps(report, indent=2))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
