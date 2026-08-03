"""Does each model know which model it is? -- a standalone precondition check.

Asks every model in ``config.MODELS`` the question in
``prompts/identity_check.txt`` and records the raw reply. This is a *precondition*
check, not an experiment: the identity-primed self-forecast condition only means
anything if the models can actually answer it, so this script exists to be run
first and read by a human.

Deliberately does no scoring. It reports a keyword hint -- whether the reply
contains the model's family and the distinctive tokens of its name -- but the hint
is advisory only. "I am Claude, made by Anthropic" and "I am a large language
model" both fail a keyword test, and only one of them is a refusal to self-
identify; a regex cannot tell the difference, so the raw reply is always kept and
printed.

    python check_model_identity.py                      # all 12 models
    python check_model_identity.py --models qwen/qwen3-8b z-ai/glm-5
    python check_model_identity.py --dry-run             # no API calls
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

_EXP_DIR = Path(__file__).parent.resolve()
_ROOT = _EXP_DIR.parents[1]
for _p in (str(_ROOT), str(_EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402

PROMPTS_DIR = _EXP_DIR / "prompts"

#: Identity answers are a capability check, not a sample -- keep them repeatable.
IDENTITY_TEMPERATURE = 0.0

#: Tokens too generic to count as self-identification ("3" matches "GPT-3", "8b"
#: appears in unrelated sizes, and every model is an "instruct" of something).
_UNINFORMATIVE_TOKENS = frozenset({
    "a", "v", "lite", "flash", "instruct", "preview", "chat", "mini", "pro",
})


def load_identity_question() -> str:
    """The question, from ``prompts/identity_check.txt``.

    Lives in ``prompts/`` rather than as a literal so this script and the
    identity-primed forecast condition provably ask the same thing.
    """
    return (PROMPTS_DIR / "identity_check.txt").read_text(encoding="utf-8").strip()


def _name_tokens(spec: config.ModelSpec) -> list[str]:
    """Distinctive lowercase tokens of a model's display name."""
    raw = re.split(r"[^0-9a-zA-Z.]+", spec.name.lower())
    return [t for t in raw if len(t) > 1 and t not in _UNINFORMATIVE_TOKENS]


def keyword_hint(spec: config.ModelSpec, reply: str) -> dict:
    """Advisory keyword overlap between a reply and the model's own name.

    Never a verdict -- see the module docstring. ``matched`` lists which of the
    name's distinctive tokens appear, so a human can see *why* the hint fired.
    """
    low = (reply or "").lower()
    tokens = _name_tokens(spec)
    matched = [t for t in tokens if t in low]
    return {
        "mentions_family": spec.family.lower() in low,
        "name_tokens": tokens,
        "matched_tokens": matched,
        "looks_self_identified": bool(matched) or spec.family.lower() in low,
    }


def ask_one(llm, spec: config.ModelSpec, question: str) -> dict:
    """Ask one model the identity question; never raises."""
    record = {
        "model": spec.openrouter_id,
        "name": spec.name,
        "family": spec.family,
        "question": question,
        "reply": None,
        "error": None,
    }
    started = time.time()
    try:
        response = llm.chat(
            spec.openrouter_id,
            [{"role": "user", "content": question}],
            temperature=IDENTITY_TEMPERATURE,
        )
        record["reply"] = response.content
    except Exception as exc:            # noqa: BLE001 -- one bad model must not abort the sweep
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = round(time.time() - started, 2)
    if record["reply"] is not None:
        record.update(keyword_hint(spec, record["reply"]))
    return record


class _StubClient:
    """Offline stand-in: echoes each model's own name back."""

    def chat(self, model: str, messages, **kwargs):
        spec = config.MODEL_BY_ID[model]

        class _R:
            content = f"I am {spec.name}, a model from {spec.family}."
        return _R()


def check_identities(models=None, *, dry_run: bool = False) -> list[dict]:
    """Ask every requested model the identity question, in list order."""
    specs = [config.MODEL_BY_ID[m] for m in models] if models else list(config.MODELS)
    question = load_identity_question()
    llm = _StubClient() if dry_run else LLMClient()
    return [ask_one(llm, spec, question) for spec in specs]


def format_report(records: list[dict], *, width: int = 96) -> str:
    """Human-readable summary: one block per model, raw reply included."""
    lines = [f"Identity check — {len(records)} models", ""]
    for r in records:
        if r["error"]:
            mark, detail = "ERR ", r["error"]
        else:
            mark = "yes " if r.get("looks_self_identified") else "?   "
            detail = " ".join((r["reply"] or "").split())
            if len(detail) > width:
                detail = detail[: width - 1] + "…"
        lines.append(f"[{mark}] {r['name']}  ({r['model']})")
        lines.append(f"        {detail}")
        if r.get("matched_tokens"):
            lines.append(f"        matched: {', '.join(r['matched_tokens'])}")
        lines.append("")
    ok = sum(1 for r in records if r.get("looks_self_identified"))
    errs = sum(1 for r in records if r["error"])
    lines.append(f"keyword hint fired for {ok}/{len(records)} models; {errs} errored")
    lines.append("The hint is advisory — read the replies before trusting it.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask each model whether it knows which model it is.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="OpenRouter ids to check (default: all of config.MODELS).")
    parser.add_argument("--out", default=str(_EXP_DIR / "results" / "identity_check.json"),
                        help="Where to write the raw replies as JSON.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use a stub client; makes no API calls.")
    args = parser.parse_args(argv)

    unknown = [m for m in (args.models or []) if m not in config.MODEL_BY_ID]
    if unknown:
        parser.error(f"not in config.MODELS: {', '.join(unknown)}")

    records = check_identities(args.models, dry_run=args.dry_run)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "question": load_identity_question(),
        "temperature": IDENTITY_TEMPERATURE,
        "dry_run": bool(args.dry_run),
        "records": records,
    }, indent=2) + "\n", encoding="utf-8")

    print(format_report(records))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
