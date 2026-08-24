#!/usr/bin/env python3
"""Identity check via direct vendor API endpoints (three questions per model).

Compare with OpenRouter results from ``check_openrouter_identity.py``. Uses the
same prompts but calls each vendor's native API instead of OpenRouter routing.

    python check_direct_identity.py --priority p0
    python check_direct_identity.py --models openai/gpt-oss-120b deepseek/deepseek-v3.2
    python check_direct_identity.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
PROMPTS = ROOT / "experiments" / "2026-05-16_self_prediction" / "prompts"
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.llm_client import LLMClient, LLMError, LLMResponse, OPENROUTER_URL  # noqa: E402

IDENTITY_TEMPERATURE = 0.0

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
}

P0_MODELS = (
    "minimax/minimax-m3",
    "openai/gpt-oss-120b",
    "deepseek/deepseek-v3.2",
)

P1_MODELS = (
    "stepfun/step-3.7-flash",
    "x-ai/grok-4.20",
    "x-ai/grok-4.5",
    "z-ai/glm-5.2",
)

# OpenRouter id -> direct endpoint config.
# ``model_ids`` is tried in order until one succeeds (handles naming uncertainty).
@dataclass(frozen=True)
class DirectSpec:
    openrouter_id: str
    display_name: str
    provider: str
    env_var: str
    api_style: str  # openai | anthropic | google
    base_url: str | None
    model_ids: tuple[str, ...]
    priority: str = "p2"
    temperature: float | None = None  # None = use IDENTITY_TEMPERATURE


DIRECT_REGISTRY: tuple[DirectSpec, ...] = (
    DirectSpec(
        "openai/gpt-oss-120b", "OpenAI gpt-oss-120b", "openai", "OPENAI_API_KEY",
        "openai", "https://api.openai.com/v1/chat/completions",
        ("gpt-oss-120b", "gpt-oss-120b:free"),
        priority="p0",
    ),
    DirectSpec(
        "openai/gpt-oss-20b", "OpenAI gpt-oss-20b", "openai", "OPENAI_API_KEY",
        "openai", "https://api.openai.com/v1/chat/completions",
        ("gpt-oss-20b",),
        priority="p0",
    ),
    DirectSpec(
        "deepseek/deepseek-v3.2", "DeepSeek V3.2", "deepseek", "DEEPSEEK_API_KEY",
        "openai", "https://api.deepseek.com/v1/chat/completions",
        ("deepseek-chat", "deepseek-v3.2"),
        priority="p0",
    ),
    DirectSpec(
        "minimax/minimax-m3", "MiniMax M3 (Together)", "together", "TOGETHER_API_KEY",
        "openai", "https://api.together.xyz/v1/chat/completions",
        ("MiniMaxAI/MiniMax-M3",),
        priority="p0",
    ),
    DirectSpec(
        "minimax/minimax-m2.7", "MiniMax M2.7", "minimax", "MINIMAX_API_KEY",
        "openai", "https://api.minimax.io/v1/text/chatcompletion_v2",
        ("MiniMax-M2.7", "minimax-m2.7"),
        priority="p1",
    ),
    DirectSpec(
        "x-ai/grok-4.20", "Grok 4.20", "xai", "XAI_API_KEY",
        "openai", "https://api.x.ai/v1/chat/completions",
        ("grok-4-20", "grok-4.20", "grok-4-latest", "grok-4"),
        priority="p1",
    ),
    DirectSpec(
        "x-ai/grok-4.5", "Grok 4.5", "xai", "XAI_API_KEY",
        "openai", "https://api.x.ai/v1/chat/completions",
        ("grok-4-5", "grok-4.5", "grok-4-latest"),
        priority="p1",
    ),
    DirectSpec(
        "moonshotai/kimi-k2.5", "Kimi K2.5", "moonshot", "MOONSHOT_API_KEY",
        "openai", "https://api.moonshot.ai/v1/chat/completions",
        ("kimi-k2.5", "moonshot-v1-128k"),
        priority="p1",
    ),
    DirectSpec(
        "moonshotai/kimi-k2.6", "Kimi K2.6", "moonshot", "MOONSHOT_API_KEY",
        "openai", "https://api.moonshot.ai/v1/chat/completions",
        ("kimi-k2.6",),
        priority="p1",
        temperature=1.0,
    ),
    DirectSpec(
        "google/gemini-3.1-pro-preview", "Gemini 3.1 Pro", "google", "GOOGLE_API_KEY",
        "google", None,
        ("gemini-3.1-pro-preview",),
        priority="p1",
    ),
    DirectSpec(
        "google/gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite", "google", "GOOGLE_API_KEY",
        "google", None,
        ("gemini-2.5-flash-lite", "gemini-2.0-flash-lite"),
        priority="p2",
    ),
    DirectSpec(
        "google/gemini-3.1-flash-lite", "Gemini 3.1 Flash Lite", "google", "GOOGLE_API_KEY",
        "google", None,
        ("gemini-3.1-flash-lite",),
        priority="p2",
    ),
    DirectSpec(
        "qwen/qwen3.6-27b", "Qwen3.6 27B", "dashscope", "DASHSCOPE_API_KEY",
        "openai", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        ("qwen3.6-27b", "qwen-plus"),
        priority="p1",
    ),
    DirectSpec(
        "qwen/qwen3.6-max-preview", "Qwen3.6 Max", "dashscope", "DASHSCOPE_API_KEY",
        "openai", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        ("qwen3.6-max-preview", "qwen-max"),
        priority="p1",
    ),
    DirectSpec(
        "qwen/qwen3-8b", "Qwen3 8B", "dashscope", "DASHSCOPE_API_KEY",
        "openai", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        ("qwen3-8b",),
        priority="p2",
    ),
    DirectSpec(
        "z-ai/glm-5.1", "GLM 5.1", "zhipu", "ZHIPU_API_KEY",
        "openai", "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        ("glm-5", "glm-4-plus"),
        priority="p1",
    ),
    DirectSpec(
        "z-ai/glm-5.2", "GLM 5.2", "zhipu", "ZHIPU_API_KEY",
        "openai", "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        ("glm-5", "glm-4-plus"),
        priority="p1",
    ),
    DirectSpec(
        "z-ai/glm-5.2", "GLM 5.2 (Together)", "together", "TOGETHER_API_KEY",
        "openai", "https://api.together.xyz/v1/chat/completions",
        ("zai-org/GLM-5.2", "zai-org/GLM-5.2-FP8"),
        priority="p1",
    ),
    DirectSpec(
        "stepfun/step-3.7-flash", "Step 3.7 Flash", "stepfun", "STEPFUN_API_KEY",
        "openai", "https://api.stepfun.com/v1/chat/completions",
        ("step-3.7-flash", "step-3"),
        priority="p1",
    ),
)

REGISTRY_BY_OR: dict[str, list[DirectSpec]] = {}
for _spec in DIRECT_REGISTRY:
    REGISTRY_BY_OR.setdefault(_spec.openrouter_id, []).append(_spec)


def load_questions() -> tuple[str, str, str]:
    general = (PROMPTS / "identity_check.txt").read_text(encoding="utf-8").strip()
    specific = (PROMPTS / "identity_check_specific.txt").read_text(encoding="utf-8").strip()
    openrouter = (PROMPTS / "identity_check_openrouter.txt").read_text(encoding="utf-8").strip()
    return general, specific, openrouter


def _slug(openrouter_id: str) -> str:
    return openrouter_id.split("/", 1)[-1].split(":", 1)[0]


def keyword_hint(openrouter_id: str, display_name: str, reply: str) -> dict:
    low = (reply or "").lower()
    source = display_name or _slug(openrouter_id)
    tokens = [t for t in re.split(r"[^0-9a-zA-Z.]+", source.lower()) if len(t) > 1 and t not in _UNINFORMATIVE]
    matched = [t for t in tokens if t in low]
    vendor = openrouter_id.split("/", 1)[0]
    vendor_hits = [h for h in _VENDOR_HINTS.get(vendor, (vendor.replace("-", " "),)) if h in low]
    slug = _slug(openrouter_id).lower()
    return {
        "vendor": vendor,
        "vendor_hits": vendor_hits,
        "name_tokens": tokens,
        "matched_tokens": matched,
        "mentions_slug": slug in low,
        "mentions_openrouter_id": openrouter_id.lower() in low,
        "looks_self_identified": bool(matched) or bool(vendor_hits),
    }


class GoogleDirectClient:
    """Minimal Gemini REST client."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def chat(self, model: str, messages: list[dict], **kwargs) -> LLMResponse:
        import requests

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={self.api_key}"
        )
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": kwargs.get("temperature", 0.0)},
        }
        resp = requests.post(url, json=payload, timeout=120)
        body = resp.json()
        if resp.status_code >= 400:
            raise LLMError(f"HTTP {resp.status_code}: {body}")
        try:
            text = body["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Malformed Gemini response: {body}") from exc
        return LLMResponse(content=text, model=model, usage={}, cost=0.0, raw=body)


class AnthropicDirectClient:
    """Minimal Anthropic Messages API client."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def chat(self, model: str, messages: list[dict], **kwargs) -> LLMResponse:
        import requests

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        body = resp.json()
        if resp.status_code >= 400:
            raise LLMError(f"HTTP {resp.status_code}: {body}")
        try:
            text = body["content"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Malformed Anthropic response: {body}") from exc
        return LLMResponse(content=text, model=body.get("model", model), usage={}, cost=0.0, raw=body)


def _make_client(spec: DirectSpec) -> tuple[Any, str | None]:
    """Return (client, error_if_key_missing)."""
    key = (os.environ.get(spec.env_var) or "").strip()
    if not key:
        return None, f"missing env var {spec.env_var}"
    if spec.api_style == "google":
        return GoogleDirectClient(key), None
    if spec.api_style == "anthropic":
        return AnthropicDirectClient(key), None
    return LLMClient(api_key=key, url=spec.base_url or OPENROUTER_URL), None


def _auth_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in ("401", "403", "invalid api key", "authentication", "unauthorized", "permission"))


def ask_with_fallback(spec: DirectSpec, question: str) -> dict:
    record: dict[str, Any] = {
        "question": question,
        "reply": None,
        "error": None,
        "endpoint": spec.provider,
        "env_var": spec.env_var,
        "direct_model_id": None,
        "returned_model": None,
    }
    client, key_err = _make_client(spec)
    if key_err:
        record["error"] = key_err
        return record

    started = time.time()
    last_err: Exception | None = None
    temperature = spec.temperature if spec.temperature is not None else IDENTITY_TEMPERATURE
    for model_id in spec.model_ids:
        try:
            response = client.chat(
                model_id,
                [{"role": "user", "content": question}],
                temperature=temperature,
                max_tokens=1024,
            )
            record["direct_model_id"] = model_id
            record["returned_model"] = response.model
            record["reply"] = response.content
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if _auth_error(exc):
                record["error"] = f"auth failure ({spec.env_var}): {type(exc).__name__}: {exc}"
                break
    else:
        record["error"] = f"{type(last_err).__name__}: {last_err}" if last_err else "no model ids"

    record["elapsed_seconds"] = round(time.time() - started, 2)
    if record["reply"] is not None:
        record.update(keyword_hint(spec.openrouter_id, spec.display_name, record["reply"]))
    return record


def pick_spec(openrouter_id: str) -> DirectSpec | None:
    candidates = REGISTRY_BY_OR.get(openrouter_id, [])
    for spec in candidates:
        key = (os.environ.get(spec.env_var) or "").strip()
        if key:
            return spec
    return candidates[0] if candidates else None


def ask_all(spec: DirectSpec, questions: tuple[str, str, str]) -> dict:
    general, specific, openrouter = questions
    return {
        "openrouter_id": spec.openrouter_id,
        "display_name": spec.display_name,
        "provider": spec.provider,
        "env_var": spec.env_var,
        "priority": spec.priority,
        "general": ask_with_fallback(spec, general),
        "specific_id": ask_with_fallback(spec, specific),
        "openrouter_slug": ask_with_fallback(spec, openrouter),
    }


def resolve_models(*, priority: str | None, models: list[str] | None) -> list[DirectSpec]:
    if models:
        out: list[DirectSpec] = []
        for mid in models:
            spec = pick_spec(mid)
            if spec is None:
                raise SystemExit(f"no direct endpoint mapping for {mid!r}")
            out.append(spec)
        return out
    wanted: list[DirectSpec] = []
    seen: set[str] = set()
    for spec in DIRECT_REGISTRY:
        if spec.openrouter_id in seen:
            continue
        if priority == "p0" and spec.priority != "p0":
            continue
        if priority == "p1" and spec.priority not in ("p0", "p1"):
            continue
        seen.add(spec.openrouter_id)
        chosen = pick_spec(spec.openrouter_id)
        if chosen:
            wanted.append(chosen)
    return wanted


class _StubClient:
    def chat(self, model: str, messages, **kwargs):
        return LLMResponse(content=f"[stub {model}]", model=model, usage={}, cost=0.0, raw={})


def run(specs: list[DirectSpec], *, dry_run: bool = False) -> list[dict]:
    questions = load_questions()
    records: list[dict] = []
    for i, spec in enumerate(specs, 1):
        print(f"[{i}/{len(specs)}] {spec.openrouter_id} via {spec.provider} ({spec.env_var})", flush=True)
        if dry_run:
            records.append({
                "openrouter_id": spec.openrouter_id,
                "display_name": spec.display_name,
                "provider": spec.provider,
                "env_var": spec.env_var,
                "priority": spec.priority,
                "general": {"reply": "stub", "direct_model_id": spec.model_ids[0]},
                "specific_id": {"reply": "stub", "direct_model_id": spec.model_ids[0]},
                "openrouter_slug": {"reply": "stub", "direct_model_id": spec.model_ids[0]},
            })
            continue
        records.append(ask_all(spec, questions))
    return records


def _summarize(reply: str | None, width: int = 80) -> str:
    if not reply:
        return ""
    one = " ".join(reply.split())
    return one if len(one) <= width else one[: width - 1] + "…"


def format_report(records: list[dict]) -> str:
    lines = [f"Direct vendor identity check — {len(records)} models", ""]
    auth_failures: dict[str, str] = {}
    missing_keys: set[str] = set()

    for r in records:
        lines.append(f"{r['display_name']}  ({r['openrouter_id']})  [{r['provider']}]")
        for label, key in (
            ("general", "general"),
            ("specific", "specific_id"),
            ("openrouter", "openrouter_slug"),
        ):
            block = r[key]
            if block.get("error"):
                detail = block["error"]
                mark = "ERR"
                if detail.startswith("missing env var"):
                    missing_keys.add(block.get("env_var") or r["env_var"])
                elif "auth failure" in detail:
                    auth_failures[block.get("env_var") or r["env_var"]] = detail
            else:
                mark = "yes" if block.get("mentions_slug") or block.get("mentions_openrouter_id") else "?"
                detail = _summarize(block.get("reply"))
            mid = block.get("direct_model_id") or "?"
            lines.append(f"  [{mark}] {label} (model={mid}): {detail}")
        lines.append("")

    if missing_keys:
        lines.append("Missing API keys (set in .env):")
        for k in sorted(missing_keys):
            lines.append(f"  - {k}")
        lines.append("")
    if auth_failures:
        lines.append("Auth failures (keys present but rejected):")
        for k, msg in sorted(auth_failures.items()):
            lines.append(f"  - {k}: {msg}")
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Identity check via direct vendor APIs.")
    parser.add_argument("--priority", choices=("p0", "p1", "all"), default="p0")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--out", default=str(ROOT / "results" / "direct_identity_check.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    priority = None if args.priority == "all" else args.priority
    specs = resolve_models(priority=priority, models=args.models)
    if not specs:
        parser.error("no models resolved")

    general, specific, openrouter = load_questions()
    records = run(specs, dry_run=args.dry_run)
    out = Path(args.out)
    payload = {
        "arm": "direct_vendor",
        "questions": {"general": general, "specific_id": specific, "openrouter_slug": openrouter},
        "temperature": IDENTITY_TEMPERATURE,
        "dry_run": bool(args.dry_run),
        "priority": args.priority,
        "records": records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(format_report(records))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
