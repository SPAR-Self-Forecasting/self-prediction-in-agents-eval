"""OpenRouter chat client.

A thin, dependency-light client over OpenRouter's OpenAI-compatible
``/chat/completions`` endpoint, with the four features the experiment needs
(EVAL_SPEC_V1.md sec 10.4, sec 10.5, sec 12):

- **Reasoning-parameter control** -- the per-model thinking toggle (sec 10.4) is
  passed straight through as OpenRouter's unified top-level ``reasoning`` object.
- **Robust retry strategy** -- transient failures (HTTP 429 / 5xx and network
  errors) are retried with exponential backoff, capped at ``backoff_cap`` so a
  long run cannot spin at 256-second sleeps. ``Retry-After`` is honored when
  the server provides it. Non-retryable 4xx errors abort immediately.
- **Thread-safe cost & token accounting** -- accumulated across calls under a
  lock so the parallel forecasting / judge stages don't drop updates.
- **Structured retry logging** -- each retry emits a WARNING log so a babysitter
  can spot a misbehaving model in the run log.

The HTTP transport is injectable (``request_fn``) so the test suite never
touches the network. ``request_fn`` may return either ``(status, body)`` (the
v0 contract) or ``(status, headers, body)`` (the v1 contract needed for
``Retry-After`` honoring); both are accepted for backward compatibility.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised on a non-recoverable LLM request failure."""


@dataclass
class LLMResponse:
    """A parsed chat completion."""

    content: str
    model: str
    usage: dict
    cost: float
    raw: dict = field(repr=False, default_factory=dict)


#: Module-level ``requests.Session``, lazily created. Reused across every
#: ``LLMClient`` that uses the default transport so HTTPS connections to
#: ``openrouter.ai`` are pooled across the ~7,500 calls of a real run
#: (Fix 5: saves TCP + TLS handshake on every call). The session is created
#: on first use under a lock to avoid races between the parallel stages.
_session_lock = threading.Lock()
_session = None


def _get_session():
    """Lazy global ``requests.Session`` for the default transport."""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                import requests

                _session = requests.Session()
    return _session


def _default_request_fn(url: str, headers: dict, payload: dict) -> tuple[int, dict, dict]:
    """Real HTTP transport. Returns ``(status_code, response_headers, body)``.

    ``requests`` is imported lazily via :func:`_get_session` so the test suite
    -- which always injects a fake transport -- need not depend on it.
    """
    resp = _get_session().post(url, headers=headers, json=payload, timeout=600)
    try:
        body = resp.json()
    except ValueError:
        body = {"error": resp.text}
    return resp.status_code, dict(resp.headers), body


def _parse_retry_after(value: str | None) -> float | None:
    """Parse the ``Retry-After`` header.

    Per RFC 7231 the value is either a non-negative integer (seconds) or an
    HTTP-date. Returns the wait in seconds, or ``None`` if the value is
    missing / unparseable (callers then fall back to their backoff schedule).
    """
    if not value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        dt = parsedate_to_datetime(value)
        if dt is None:
            return None
        import datetime as _dt

        now = _dt.datetime.now(tz=dt.tzinfo) if dt.tzinfo else _dt.datetime.now()
        return max(0.0, (dt - now).total_seconds())
    except Exception:  # pragma: no cover - defensive against weird header values
        return None


class LLMClient:
    """Stateful OpenRouter client; tracks cumulative cost and token usage."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        url: str = OPENROUTER_URL,
        request_fn: Callable | None = None,
        max_retries: int = 8,
        backoff_base: float = 0.5,
        backoff_cap: float = 60.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.url = url
        self._uses_default_transport = request_fn is None
        self._request_fn = request_fn or _default_request_fn
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self._sleep_fn = sleep_fn
        # accounting -- guarded by a lock because the forecasting / judge
        # stages drive the client from a ThreadPoolExecutor (run.py stages 3+5).
        self._account_lock = threading.Lock()
        self.total_cost = 0.0
        self.call_count = 0
        self.total_tokens = {"prompt": 0, "completion": 0, "total": 0}

    # ------------------------------------------------------------------
    def complete(self, model: str, prompt: str, **kwargs) -> LLMResponse:
        """Convenience wrapper: send ``prompt`` as a single user message."""
        return self.chat(model, [{"role": "user", "content": prompt}], **kwargs)

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        reasoning: dict | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict | None = None,
        extra: dict | None = None,
    ) -> LLMResponse:
        """Send a chat completion request and return the parsed :class:`LLMResponse`."""
        if self._uses_default_transport and not self.api_key:
            raise LLMError(
                "No OpenRouter API key. Set OPENROUTER_API_KEY or pass api_key=..."
            )

        payload: dict = {"model": model, "messages": messages, "usage": {"include": True}}
        if reasoning is not None:
            payload["reasoning"] = reasoning
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if response_format is not None:
            payload["response_format"] = response_format
        if extra:
            payload.update(extra)

        body = self._request_with_retries(payload, model)
        return self._parse(body, model)

    # ------------------------------------------------------------------
    def _invoke_transport(self, payload: dict) -> tuple[int, dict, dict]:
        """Call the transport and normalise to a 3-tuple ``(status, headers, body)``.

        Tests still use the v0 ``(status, body)`` 2-tuple contract; production
        and the new Retry-After tests use the v1 3-tuple. Both are accepted.
        """
        headers_out = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        result = self._request_fn(self.url, headers_out, payload)
        if isinstance(result, tuple) and len(result) == 3:
            return result  # (status, response_headers, body)
        if isinstance(result, tuple) and len(result) == 2:
            status, body = result
            return status, {}, body
        raise LLMError(f"request_fn returned an unrecognised shape: {type(result)}")

    def _request_with_retries(self, payload: dict, model: str) -> dict:
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            transient = False
            retry_after: float | None = None
            try:
                status, resp_headers, body = self._invoke_transport(payload)
            except Exception as exc:  # network-level failure -> transient
                last_err, transient = exc, True
            else:
                if 200 <= status < 300:
                    return body
                if status == 429 or status >= 500:
                    last_err, transient = LLMError(f"HTTP {status}: {body}"), True
                    if status == 429:
                        retry_after = _parse_retry_after(
                            resp_headers.get("Retry-After")
                            or resp_headers.get("retry-after"),
                        )
                else:
                    raise LLMError(f"HTTP {status} (non-retryable): {body}")
            if not transient:  # pragma: no cover - defensive
                break
            if attempt < self.max_retries:
                if retry_after is not None:
                    wait = retry_after
                    reason = f"Retry-After={retry_after:g}s"
                else:
                    wait = min(self.backoff_cap, self.backoff_base * (2 ** attempt))
                    reason = f"backoff={wait:g}s"
                logger.warning(
                    "LLM transient failure on %s (attempt %d/%d): %s -- retrying after %s",
                    model, attempt + 1, self.max_retries + 1, last_err, reason,
                )
                self._sleep_fn(wait)
        raise LLMError(
            f"Request failed after {self.max_retries} retries on {model}: {last_err}"
        )

    def _parse(self, body: dict, model: str) -> LLMResponse:
        try:
            choice = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Malformed response (no choices): {body}") from exc
        content = choice.get("content")
        if content is None:
            content = ""
        usage = body.get("usage", {}) or {}
        cost = float(usage.get("cost", 0.0) or 0.0)

        # Account under the lock so concurrent stages can't drop increments.
        with self._account_lock:
            self.call_count += 1
            self.total_cost += cost
            self.total_tokens["prompt"] += int(usage.get("prompt_tokens", 0) or 0)
            self.total_tokens["completion"] += int(usage.get("completion_tokens", 0) or 0)
            self.total_tokens["total"] += int(usage.get("total_tokens", 0) or 0)

        return LLMResponse(
            content=content,
            model=body.get("model", model),
            usage=usage,
            cost=cost,
            raw=body,
        )

    def accounting(self) -> dict:
        """A snapshot of cumulative usage for the run report."""
        with self._account_lock:
            return {
                "calls": self.call_count,
                "total_cost_usd": round(self.total_cost, 4),
                "tokens": dict(self.total_tokens),
            }
