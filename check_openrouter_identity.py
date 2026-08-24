#!/usr/bin/env python3
"""Identity check for arbitrary OpenRouter models (three questions per model).

Questions (shared with experiment scripts):
  1. identity_check.txt           — general self-knowledge
  2. identity_check_specific.txt  — ask for specific model ID
  3. identity_check_openrouter.txt — ask for OpenRouter slug

    python check_openrouter_identity.py
    python check_openrouter_identity.py --models-file openrouter_top50_models.txt
    python check_openrouter_identity.py --models openai/gpt-4o-mini
    python check_openrouter_identity.py --pin-providers auto
    python check_openrouter_identity.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROMPTS = ROOT / "experiments" / "2026-05-16_self_prediction" / "prompts"
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.llm_client import LLMClient  # noqa: E402

IDENTITY_TEMPERATURE = 0.0
IDENTITY_MAX_RETRIES = 2
IDENTITY_BACKOFF_CAP = 5.0
PER_MODEL_TIMEOUT_SECONDS = 90.0
STUCK_ERROR_MARKERS = ("429", "retries", "timeout", "503", "502", "504", "overloaded")
OPENROUTER_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model_id}/endpoints"

# First-party / preferred upstream hosts per OpenRouter vendor prefix. Tried in order.
VENDOR_PREFERRED: dict[str, tuple[str, ...]] = {
    "openai": ("OpenAI",),
    "anthropic": ("Anthropic",),
    "google": ("Google",),
    "deepseek": ("DeepSeek", "DeepInfra"),
    "moonshotai": ("Moonshot AI", "Moonshot"),
    "x-ai": ("xAI",),
    "minimax": ("Minimax",),
    "z-ai": ("Z.ai", "Zhipu"),
    "qwen": ("Alibaba",),
    "nvidia": ("NVIDIA",),
    "xiaomi": ("Xiaomi",),
    "tencent": ("Tencent",),
    "stepfun": ("StepFun",),
    "cohere": ("Cohere",),
    "mistralai": ("Mistral",),
    "inclusionai": ("InclusionAI",),
    "poolside": ("Poolside",),
}

# Deprioritize hosts that showed identity drift in the unpinned sweep.
AVOID_PROVIDERS: frozenset[str] = frozenset({"Baidu"})
_UNINFORMATIVE = frozenset({
    "a", "v", "lite", "flash", "instruct", "preview", "chat", "mini", "pro", "free",
})

_VENDOR_HINTS: dict[str, tuple[str, ...]] = {
    "openai": ("openai", "gpt", "chatgpt"),
    "anthropic": ("anthropic", "claude"),
    "google": ("google", "gemini", "gemma"),
    "deepseek": ("deepseek",),
    "moonshotai": ("kimi", "moonshot"),
    "x-ai": ("grok", "xai"),
    "z-ai": ("glm", "zhipu", "z.ai"),
    "qwen": ("qwen",),
    "minimax": ("minimax",),
    "nvidia": ("nemotron", "nvidia"),
    "xiaomi": ("mimo", "xiaomi"),
    "tencent": ("hy", "tencent"),
    "stepfun": ("step", "stepfun"),
    "inclusionai": ("ling",),
    "poolside": ("laguna", "poolside"),
    "cohere": ("cohere", "north"),
    "mistralai": ("mistral",),
}


@dataclass(frozen=True)
class ModelEntry:
    openrouter_id: str
    display_name: str | None = None


@dataclass
class PinConfig:
    """OpenRouter provider routing for one model."""

    pinned_provider: str | None
    available_providers: list[str] = field(default_factory=list)
    provider_payload: dict | None = None
    pin_error: str | None = None


def _vendor_prefix(openrouter_id: str) -> str:
    return openrouter_id.split("/", 1)[0]


@lru_cache(maxsize=256)
def fetch_endpoint_providers(model_id: str) -> tuple[str, ...]:
    """Return provider_name values available for a model on OpenRouter."""
    import os
    import requests

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return ()
    url = OPENROUTER_ENDPOINTS_URL.format(model_id=model_id)
    resp = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
    if resp.status_code >= 400:
        return ()
    try:
        endpoints = resp.json()["data"]["endpoints"]
    except (KeyError, TypeError, ValueError):
        return ()
    names: list[str] = []
    seen: set[str] = set()
    for ep in endpoints:
        name = ep.get("provider_name")
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return tuple(names)


def choose_pinned_provider(model_id: str) -> PinConfig:
    """Pick a first-party provider when possible; avoid known-bad hosts."""
    available = list(fetch_endpoint_providers(model_id))
    if not available:
        return PinConfig(
            pinned_provider=None,
            available_providers=[],
            pin_error="no endpoints returned from OpenRouter",
        )

    preferred = VENDOR_PREFERRED.get(_vendor_prefix(model_id), ())
    candidates = [p for p in available if p not in AVOID_PROVIDERS]
    if not candidates:
        candidates = available

    for name in preferred:
        if name in candidates:
            return PinConfig(
                pinned_provider=name,
                available_providers=available,
                provider_payload={
                    "order": [name],
                    "allow_fallbacks": False,
                },
            )

    pinned = candidates[0]
    return PinConfig(
        pinned_provider=pinned,
        available_providers=available,
        provider_payload={
            "order": [pinned],
            "allow_fallbacks": False,
        },
    )


def load_questions() -> tuple[str, str, str]:
    general = (PROMPTS / "identity_check.txt").read_text(encoding="utf-8").strip()
    specific = (PROMPTS / "identity_check_specific.txt").read_text(encoding="utf-8").strip()
    openrouter = (PROMPTS / "identity_check_openrouter.txt").read_text(encoding="utf-8").strip()
    return general, specific, openrouter


def parse_models_file(path: Path) -> list[ModelEntry]:
    text = path.read_text(encoding="utf-8")
    entries: list[ModelEntry] = []
    for line in text.splitlines():
        m = re.match(r"^\s*\d+\.\s+(\S+)\s+(.+?)\s+\d+k", line)
        if m:
            entries.append(ModelEntry(m.group(1), m.group(2).strip()))
    if entries:
        return entries
    # Fallback: IDs-only block
    ids_only = False
    for line in text.splitlines():
        if line.strip().startswith("IDs only"):
            ids_only = True
            continue
        if ids_only and "/" in line:
            oid = line.strip()
            if oid:
                entries.append(ModelEntry(oid))
    return entries


def _matches_exclude(openrouter_id: str, excludes: list[str]) -> bool:
    for pattern in excludes:
        if pattern.endswith("/*"):
            prefix = pattern[:-2]
            if openrouter_id.startswith(f"{prefix}/"):
                return True
        elif openrouter_id == pattern:
            return True
    return False


def filter_entries(
    entries: list[ModelEntry],
    *,
    excludes: list[str] | None = None,
    skip_ids: set[str] | None = None,
) -> list[ModelEntry]:
    out: list[ModelEntry] = []
    for entry in entries:
        if skip_ids and entry.openrouter_id in skip_ids:
            continue
        if excludes and _matches_exclude(entry.openrouter_id, excludes):
            continue
        out.append(entry)
    return out


def load_resume_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("records") or [])


def _slug(openrouter_id: str) -> str:
    base = openrouter_id.split("/", 1)[-1]
    return base.split(":", 1)[0]


def _name_tokens(display_name: str | None, openrouter_id: str) -> list[str]:
    source = display_name or _slug(openrouter_id)
    raw = re.split(r"[^0-9a-zA-Z.]+", source.lower())
    return [t for t in raw if len(t) > 1 and t not in _UNINFORMATIVE]


def keyword_hint(entry: ModelEntry, reply: str) -> dict:
    low = (reply or "").lower()
    tokens = _name_tokens(entry.display_name, entry.openrouter_id)
    matched = [t for t in tokens if t in low]
    vendor = entry.openrouter_id.split("/", 1)[0]
    vendor_hits = [h for h in _VENDOR_HINTS.get(vendor, (vendor.replace("-", " "),)) if h in low]
    slug = _slug(entry.openrouter_id).lower()
    full = entry.openrouter_id.lower()
    return {
        "vendor": vendor,
        "vendor_hits": vendor_hits,
        "name_tokens": tokens,
        "matched_tokens": matched,
        "mentions_slug": slug in low,
        "mentions_openrouter_id": full in low,
        "looks_self_identified": bool(matched) or bool(vendor_hits),
    }


def _summarize(reply: str | None, width: int = 88) -> str:
    if not reply:
        return ""
    one = " ".join(reply.split())
    return one if len(one) <= width else one[: width - 1] + "…"


def _is_stuck_error(error: str | None) -> bool:
    if not error:
        return False
    low = error.lower()
    return any(marker in low for marker in STUCK_ERROR_MARKERS)


def _skipped_block(question: str, reason: str, *, pin: PinConfig | None = None) -> dict:
    block = {
        "question": question,
        "reply": None,
        "error": reason,
        "skipped": True,
    }
    if pin is not None:
        block["pinned_provider"] = pin.pinned_provider
        block["available_providers"] = pin.available_providers
        if pin.pin_error:
            block["pin_error"] = pin.pin_error
    return block


def ask_question(
    llm,
    entry: ModelEntry,
    question: str,
    key: str,
    *,
    pin: PinConfig | None = None,
    deadline: float | None = None,
) -> dict:
    record = {"question": question, "reply": None, "error": None}
    if pin is not None:
        record["pinned_provider"] = pin.pinned_provider
        record["available_providers"] = pin.available_providers
        if pin.pin_error:
            record["pin_error"] = pin.pin_error
    if deadline is not None and time.time() >= deadline:
        record["error"] = "skipped: model timeout"
        record["skipped"] = True
        return record
    started = time.time()
    try:
        extra = {"provider": pin.provider_payload} if pin and pin.provider_payload else None
        response = llm.chat(
            entry.openrouter_id,
            [{"role": "user", "content": question}],
            temperature=IDENTITY_TEMPERATURE,
            extra=extra,
        )
        record["reply"] = response.content
        record["returned_model"] = response.raw.get("model")
        record["provider"] = response.raw.get("provider")
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = round(time.time() - started, 2)
    if record["reply"] is not None:
        record.update(keyword_hint(entry, record["reply"]))
    return record


def ask_all(
    llm,
    entry: ModelEntry,
    questions: tuple[str, str, str],
    *,
    pin: PinConfig | None = None,
    skip_on_stuck: bool = True,
) -> dict:
    general, specific, openrouter = questions
    deadline = time.time() + PER_MODEL_TIMEOUT_SECONDS
    blocks: list[tuple[str, str, str]] = [
        ("general", general, "general"),
        ("specific_id", specific, "specific_id"),
        ("openrouter_slug", openrouter, "openrouter_slug"),
    ]
    result: dict = {
        "model": entry.openrouter_id,
        "display_name": entry.display_name,
    }
    if pin is not None:
        result["pinned_provider"] = pin.pinned_provider
        result["available_providers"] = pin.available_providers
        if pin.pin_error:
            result["pin_error"] = pin.pin_error

    for idx, (key, question, _label) in enumerate(blocks):
        block = ask_question(llm, entry, question, key, pin=pin, deadline=deadline)
        result[key] = block
        if not skip_on_stuck:
            continue
        if block.get("skipped") or _is_stuck_error(block.get("error")):
            reason = block.get("error") or "skipped: stuck"
            for rest_key, rest_question, _ in blocks[idx + 1:]:
                result[rest_key] = _skipped_block(
                    rest_question,
                    f"skipped: {reason}",
                    pin=pin,
                )
            result["skipped_stuck"] = True
            break
    return result


class _StubClient:
    def chat(self, model: str, messages, **kwargs):
        from src.llm_client import LLMResponse
        return LLMResponse(
            content=f"I am {model}.",
            model=model,
            usage={},
            cost=0.0,
            raw={"model": model, "provider": "stub"},
        )


def run(
    entries: list[ModelEntry],
    *,
    dry_run: bool = False,
    out: Path | None = None,
    pin_providers: bool = False,
    initial_records: list[dict] | None = None,
) -> list[dict]:
    questions = load_questions()
    llm = (
        _StubClient()
        if dry_run
        else LLMClient(max_retries=IDENTITY_MAX_RETRIES, backoff_cap=IDENTITY_BACKOFF_CAP)
    )
    records: list[dict] = list(initial_records or [])
    base_count = len(records)
    skipped_stuck: list[str] = []
    for i, entry in enumerate(entries, 1):
        pin = choose_pinned_provider(entry.openrouter_id) if pin_providers else None
        pin_note = f" -> {pin.pinned_provider}" if pin and pin.pinned_provider else ""
        print(f"[{base_count + i}/{base_count + len(entries)}] {entry.openrouter_id}{pin_note}", flush=True)
        record = ask_all(llm, entry, questions, pin=pin, skip_on_stuck=True)
        if record.get("skipped_stuck"):
            skipped_stuck.append(entry.openrouter_id)
            print(f"  skipped remaining questions for {entry.openrouter_id} (stuck)", flush=True)
        records.append(record)
        if out and not dry_run:
            payload = {
                "questions": {
                    "general": questions[0],
                    "specific_id": questions[1],
                    "openrouter_slug": questions[2],
                },
                "temperature": IDENTITY_TEMPERATURE,
                "dry_run": dry_run,
                "pin_providers": pin_providers,
                "skipped_stuck": skipped_stuck,
                "source": str(out),
                "records": records,
            }
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return records[base_count:], skipped_stuck


def format_report(records: list[dict], *, pinned: bool = False) -> str:
    title = "OpenRouter identity check"
    if pinned:
        title += " (pinned providers)"
    lines = [f"{title} — {len(records)} models", ""]
    for r in records:
        name = r.get("display_name") or r["model"]
        pin = r.get("pinned_provider")
        header = f"{name}  ({r['model']})"
        if pin:
            header += f"  [pinned: {pin}]"
        lines.append(header)
        for label, key in (
            ("general", "general"),
            ("specific", "specific_id"),
            ("openrouter", "openrouter_slug"),
        ):
            block = r[key]
            if block.get("error"):
                detail = block["error"]
                mark = "ERR"
            else:
                mark = "yes" if block.get("mentions_openrouter_id") or block.get("mentions_slug") else "?"
                detail = _summarize(block.get("reply"), 80)
            prov = block.get("provider") or "?"
            lines.append(f"  [{mark}] {label} ({prov}): {detail}")
        lines.append("")
    slug_ok = sum(1 for r in records if r["openrouter_slug"].get("mentions_slug"))
    id_ok = sum(1 for r in records if r["openrouter_slug"].get("mentions_openrouter_id"))
    errs = sum(
        1 for r in records
        if any(r[k].get("error") for k in ("general", "specific_id", "openrouter_slug"))
    )
    lines.append(f"openrouter slug mentioned: {slug_ok}/{len(records)}; full id: {id_ok}/{len(records)}; errors: {errs}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Three-question identity check for OpenRouter models.")
    parser.add_argument("--models-file", default=str(ROOT / "openrouter_top50_models.txt"))
    parser.add_argument("--models", nargs="+", default=None, help="Override model ids.")
    parser.add_argument("--out", default=str(ROOT / "results" / "openrouter_top50_identity_check.json"))
    parser.add_argument("--pin-providers", choices=("auto", "off"), default="off",
                        help="Pin each model to a first-party OpenRouter provider (no fallbacks).")
    parser.add_argument("--exclude", nargs="+", default=None,
                        help="Skip model ids or vendor prefixes (e.g. deepseek/*).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip models already present in --out and append new records.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    pin_providers = args.pin_providers == "auto"
    if pin_providers and args.out == str(ROOT / "results" / "openrouter_top50_identity_check.json"):
        args.out = str(ROOT / "results" / "openrouter_top50_identity_check_pinned.json")

    if args.models:
        entries = [ModelEntry(m) for m in args.models]
    else:
        entries = parse_models_file(Path(args.models_file))
    if not entries:
        parser.error("no models found")

    out = Path(args.out)
    existing = load_resume_records(out) if args.resume else []
    done_ids = {r["model"] for r in existing}
    entries = filter_entries(entries, excludes=args.exclude, skip_ids=done_ids)
    if not entries and not existing:
        parser.error("no models left to run after excludes/resume")

    general, specific, openrouter = load_questions()
    new_records, skipped_stuck = (
        run(
            entries,
            dry_run=args.dry_run,
            out=out,
            pin_providers=pin_providers,
            initial_records=existing,
        )
        if entries
        else ([], [])
    )
    records = existing + new_records
    all_skipped = sorted({
        *(r["model"] for r in records if r.get("skipped_stuck")),
        *skipped_stuck,
    })
    payload = {
        "questions": {"general": general, "specific_id": specific, "openrouter_slug": openrouter},
        "temperature": IDENTITY_TEMPERATURE,
        "dry_run": bool(args.dry_run),
        "pin_providers": pin_providers,
        "excluded": args.exclude,
        "skipped_stuck": all_skipped,
        "resumed_from": str(out) if args.resume and existing else None,
        "models_file": args.models_file if not args.models else None,
        "records": records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(format_report(records, pinned=pin_providers))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
