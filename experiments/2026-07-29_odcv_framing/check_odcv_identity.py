"""Does each ODCV executor know which model it is? -- standalone precondition check.

Asks every model in ``odcv.config.MODELS`` two questions (shared wording with
experiment 1):

1. ``identity_check.txt`` -- general self-knowledge
2. ``identity_check_specific.txt`` -- asks for the specific model ID

Deliberately does no scoring beyond advisory keyword hints; read the replies.

    python check_odcv_identity.py                      # all 9 executors
    python check_odcv_identity.py --models x-ai/grok-4.20
    python check_odcv_identity.py --dry-run             # no API calls
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

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from odcv import config  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402

_PROMPTS = _ROOT / "experiments" / "2026-05-16_self_prediction" / "prompts"
IDENTITY_TEMPERATURE = 0.0

_UNINFORMATIVE_TOKENS = frozenset({
    "a", "v", "lite", "flash", "instruct", "preview", "chat", "mini", "pro",
})

_VENDOR_HINTS: dict[str, tuple[str, ...]] = {
    "openai": ("openai", "gpt"),
    "google": ("google", "gemini"),
    "x-ai": ("grok", "xai", "x-ai"),
    "z-ai": ("glm", "zhipu", "z-ai"),
    "moonshotai": ("kimi", "moonshot"),
    "qwen": ("qwen",),
    "minimax": ("minimax",),
}


def load_identity_questions() -> tuple[str, str, str]:
    general = (_PROMPTS / "identity_check.txt").read_text(encoding="utf-8").strip()
    specific = (_PROMPTS / "identity_check_specific.txt").read_text(encoding="utf-8").strip()
    openrouter = (_PROMPTS / "identity_check_openrouter.txt").read_text(encoding="utf-8").strip()
    return general, specific, openrouter


def _name_tokens(name: str) -> list[str]:
    raw = re.split(r"[^0-9a-zA-Z.]+", name.lower())
    return [t for t in raw if len(t) > 1 and t not in _UNINFORMATIVE_TOKENS]


def keyword_hint(spec: config.ModelSpec, reply: str) -> dict:
    low = (reply or "").lower()
    tokens = _name_tokens(spec.name)
    matched = [t for t in tokens if t in low]
    vendor = spec.openrouter_id.split("/", 1)[0]
    vendor_hits = [h for h in _VENDOR_HINTS.get(vendor, (vendor,)) if h in low]
    return {
        "vendor": vendor,
        "vendor_hits": vendor_hits,
        "name_tokens": tokens,
        "matched_tokens": matched,
        "looks_self_identified": bool(matched) or bool(vendor_hits),
    }


def _ask_question(llm, spec: config.ModelSpec, question: str) -> dict:
    record = {"question": question, "reply": None, "error": None}
    started = time.time()
    try:
        response = llm.chat(
            spec.openrouter_id,
            [{"role": "user", "content": question}],
            reasoning=spec.reasoning,
            temperature=IDENTITY_TEMPERATURE,
        )
        record["reply"] = response.content
        record["returned_model"] = response.raw.get("model")
        record["provider"] = response.raw.get("provider")
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = round(time.time() - started, 2)
    if record["reply"] is not None:
        record.update(keyword_hint(spec, record["reply"]))
        low = record["reply"].lower()
        slug = spec.openrouter_id.split("/", 1)[-1].split(":", 1)[0].lower()
        record["mentions_openrouter_id"] = spec.openrouter_id.lower() in low
        record["mentions_slug"] = slug in low
    return record


def ask_all(llm, spec: config.ModelSpec, general: str, specific: str, openrouter: str) -> dict:
    return {
        "model": spec.openrouter_id,
        "name": spec.name,
        "openrouter_id": spec.openrouter_id,
        "reasoning": spec.reasoning,
        "general": _ask_question(llm, spec, general),
        "specific_id": _ask_question(llm, spec, specific),
        "openrouter_slug": _ask_question(llm, spec, openrouter),
    }


class _StubClient:
    def chat(self, model: str, messages, **kwargs):
        spec = config.MODEL_BY_ID[model]

        class _R:
            content = f"I am {spec.name} ({spec.openrouter_id})."

        return _R()


def check_identities(models=None, *, dry_run: bool = False) -> list[dict]:
    specs = [config.MODEL_BY_ID[m] for m in models] if models else list(config.MODELS)
    general, specific, openrouter = load_identity_questions()
    llm = _StubClient() if dry_run else LLMClient()
    return [ask_all(llm, spec, general, specific, openrouter) for spec in specs]


def _reply_line(block: dict, *, width: int) -> tuple[str, str]:
    if block["error"]:
        return "ERR ", block["error"]
    mark = "yes " if block.get("looks_self_identified") else "?   "
    detail = " ".join((block["reply"] or "").split())
    if len(detail) > width:
        detail = detail[: width - 1] + "…"
    return mark, detail


def format_report(records: list[dict], *, width: int = 96) -> str:
    lines = [f"Identity check — {len(records)} models (general + specific + openrouter)", ""]
    for r in records:
        lines.append(f"{r['name']}  ({r['model']})")
        for label, key in (
            ("general", "general"),
            ("specific", "specific_id"),
            ("openrouter", "openrouter_slug"),
        ):
            mark, detail = _reply_line(r[key], width=width)
            id_note = ""
            if key == "openrouter_slug" and r[key].get("mentions_openrouter_id"):
                id_note = " [full id]"
            elif key == "openrouter_slug" and r[key].get("mentions_slug"):
                id_note = " [slug]"
            lines.append(f"  [{mark}] {label}: {detail}{id_note}")
        lines.append("")
    ok_g = sum(1 for r in records if r["general"].get("looks_self_identified"))
    ok_s = sum(1 for r in records if r["specific_id"].get("looks_self_identified"))
    slug_ok = sum(1 for r in records if r["openrouter_slug"].get("mentions_slug"))
    id_ok = sum(1 for r in records if r["openrouter_slug"].get("mentions_openrouter_id"))
    errs = sum(
        1 for r in records
        if r["general"]["error"] or r["specific_id"]["error"] or r["openrouter_slug"]["error"]
    )
    lines.append(
        f"keyword hint: general {ok_g}/{len(records)}, specific {ok_s}/{len(records)}; "
        f"openrouter slug {slug_ok}/{len(records)}, full id {id_ok}/{len(records)}; "
        f"{errs} models errored"
    )
    lines.append("The hints are advisory — read the replies before trusting them.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask each ODCV executor whether it knows which model it is.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="OpenRouter ids to check (default: all of odcv.config.MODELS).")
    parser.add_argument("--out", default=str(_EXP_DIR / "results" / "identity_check.json"),
                        help="Where to write the raw replies as JSON.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use a stub client; makes no API calls.")
    args = parser.parse_args(argv)

    unknown = [m for m in (args.models or []) if m not in config.MODEL_BY_ID]
    if unknown:
        parser.error(f"not in odcv.config.MODELS: {', '.join(unknown)}")

    general, specific, openrouter = load_identity_questions()
    records = check_identities(args.models, dry_run=args.dry_run)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "questions": {"general": general, "specific_id": specific, "openrouter_slug": openrouter},
        "temperature": IDENTITY_TEMPERATURE,
        "dry_run": bool(args.dry_run),
        "records": records,
    }, indent=2) + "\n", encoding="utf-8")

    print(format_report(records))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
