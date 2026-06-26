"""Tests for src/llm_client.py -- OpenRouter client. No real network calls."""

import threading

import pytest

from src.llm_client import LLMClient, LLMError


def _ok_body(content="hello", *, cost=0.0, ptok=10, ctok=5):
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "model": "test/model",
        "usage": {
            "prompt_tokens": ptok,
            "completion_tokens": ctok,
            "total_tokens": ptok + ctok,
            "cost": cost,
        },
    }


class FakeTransport:
    """Scripted transport: records payloads, replays (status, body) or (status, headers, body)."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []  # list of payload dicts

    def __call__(self, url, headers, payload):
        self.calls.append(payload)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item  # (status_code, body) -- legacy 2-tuple still supported


# --------------------------------------------------------------------------
# Basic request / response
# --------------------------------------------------------------------------
def test_chat_returns_parsed_content():
    t = FakeTransport([(200, _ok_body("the answer"))])
    client = LLMClient(api_key="k", request_fn=t)
    resp = client.chat("openai/gpt-4o-mini", [{"role": "user", "content": "hi"}])
    assert resp.content == "the answer"
    assert resp.usage["completion_tokens"] == 5


def test_payload_includes_model_and_messages():
    t = FakeTransport([(200, _ok_body())])
    LLMClient(api_key="k", request_fn=t).chat("m", [{"role": "user", "content": "q"}])
    assert t.calls[0]["model"] == "m"
    assert t.calls[0]["messages"] == [{"role": "user", "content": "q"}]


def test_reasoning_param_threaded_into_payload():
    t = FakeTransport([(200, _ok_body())])
    client = LLMClient(api_key="k", request_fn=t)
    client.chat("m", [{"role": "user", "content": "q"}], reasoning={"enabled": False})
    assert t.calls[0]["reasoning"] == {"enabled": False}


def test_reasoning_omitted_when_none():
    t = FakeTransport([(200, _ok_body())])
    LLMClient(api_key="k", request_fn=t).chat("m", [{"role": "user", "content": "q"}])
    assert "reasoning" not in t.calls[0]


def test_optional_params_included_and_omitted():
    t = FakeTransport([(200, _ok_body()), (200, _ok_body())])
    client = LLMClient(api_key="k", request_fn=t)
    client.chat("m", [{"role": "user", "content": "q"}], max_tokens=128, temperature=0.3)
    assert t.calls[0]["max_tokens"] == 128
    assert t.calls[0]["temperature"] == 0.3
    client.chat("m", [{"role": "user", "content": "q"}])
    assert "max_tokens" not in t.calls[1] and "temperature" not in t.calls[1]


def test_complete_wraps_single_user_message():
    t = FakeTransport([(200, _ok_body("y"))])
    resp = LLMClient(api_key="k", request_fn=t).complete("m", "a prompt")
    assert resp.content == "y"
    assert t.calls[0]["messages"] == [{"role": "user", "content": "a prompt"}]


def test_null_content_becomes_empty_string():
    t = FakeTransport([(200, _ok_body(content=None))])
    resp = LLMClient(api_key="k", request_fn=t).complete("m", "p")
    assert resp.content == ""


# --------------------------------------------------------------------------
# Retries & backoff
# --------------------------------------------------------------------------
def test_retries_on_503_then_succeeds():
    sleeps = []
    t = FakeTransport([(503, {"error": "busy"}), (503, {"error": "busy"}), (200, _ok_body("ok"))])
    client = LLMClient(api_key="k", request_fn=t, max_retries=3, sleep_fn=sleeps.append)
    resp = client.chat("m", [{"role": "user", "content": "q"}])
    assert resp.content == "ok"
    assert len(sleeps) == 2  # slept before each of the two retries


def test_retries_on_network_exception():
    t = FakeTransport([ConnectionError("dropped"), (200, _ok_body("recovered"))])
    client = LLMClient(api_key="k", request_fn=t, max_retries=3, sleep_fn=lambda s: None)
    assert client.chat("m", [{"role": "user", "content": "q"}]).content == "recovered"


def test_retry_exhaustion_raises():
    t = FakeTransport([(500, {}), (500, {}), (500, {}), (500, {})])
    client = LLMClient(api_key="k", request_fn=t, max_retries=3, sleep_fn=lambda s: None)
    with pytest.raises(LLMError):
        client.chat("m", [{"role": "user", "content": "q"}])


def test_non_retryable_4xx_raises_immediately():
    t = FakeTransport([(400, {"error": "bad request"}), (200, _ok_body())])
    client = LLMClient(api_key="k", request_fn=t, max_retries=3, sleep_fn=lambda s: None)
    with pytest.raises(LLMError):
        client.chat("m", [{"role": "user", "content": "q"}])
    assert len(t.calls) == 1  # did not retry


def test_default_max_retries_is_8():
    """Production default -- max 9 attempts (1 + 8 retries) to ride out OpenRouter blips."""
    client = LLMClient(api_key="k", request_fn=FakeTransport([(200, _ok_body())]))
    assert client.max_retries == 8


def test_backoff_caps_at_ceiling():
    """Exponential backoff must cap (0.5 * 2**attempt would hit 128s on attempt 8)."""
    sleeps: list[float] = []
    # 8 transient failures (the max we'll retry through) then a success.
    t = FakeTransport([(500, {})] * 8 + [(200, _ok_body())])
    client = LLMClient(api_key="k", request_fn=t, backoff_cap=10.0,
                       sleep_fn=sleeps.append)
    client.chat("m", [{"role": "user", "content": "q"}])
    assert max(sleeps) <= 10.0 + 1e-9, "no sleep may exceed the configured cap"


def test_retry_after_header_seconds_honored():
    """A 429 with ``Retry-After: 7`` must sleep for 7 seconds before retrying."""
    sleeps: list[float] = []
    t = FakeTransport([
        (429, {"Retry-After": "7"}, {"error": "rate limited"}),
        (200, {}, _ok_body("ok")),
    ])
    client = LLMClient(api_key="k", request_fn=t, max_retries=3, sleep_fn=sleeps.append)
    assert client.chat("m", [{"role": "user", "content": "q"}]).content == "ok"
    assert sleeps == [7.0], "Retry-After must override exponential backoff"


def test_retry_after_header_case_insensitive():
    """HTTP headers are case-insensitive; the lowercase variant must also be honored."""
    sleeps: list[float] = []
    t = FakeTransport([
        (429, {"retry-after": "4"}, {"error": "rl"}),
        (200, {}, _ok_body()),
    ])
    LLMClient(api_key="k", request_fn=t, sleep_fn=sleeps.append).chat(
        "m", [{"role": "user", "content": "q"}],
    )
    assert sleeps == [4.0]


def test_retry_after_falls_back_to_backoff_when_unparseable():
    """A garbage Retry-After value falls back to exponential backoff, not crash."""
    sleeps: list[float] = []
    t = FakeTransport([
        (429, {"Retry-After": "not-a-number"}, {"error": "rl"}),
        (200, {}, _ok_body()),
    ])
    LLMClient(api_key="k", request_fn=t, backoff_base=0.5, sleep_fn=sleeps.append).chat(
        "m", [{"role": "user", "content": "q"}],
    )
    assert sleeps == [0.5]  # 0.5 * 2**0


def test_legacy_two_tuple_transport_still_works():
    """Backward compatibility: existing test transports returning (status, body) still work."""
    t = FakeTransport([(200, _ok_body("legacy"))])
    assert LLMClient(api_key="k", request_fn=t).chat(
        "m", [{"role": "user", "content": "q"}],
    ).content == "legacy"


def test_retry_logs_emitted_on_each_retry(caplog):
    import logging

    t = FakeTransport([(503, {}), (503, {}), (200, _ok_body("ok"))])
    client = LLMClient(api_key="k", request_fn=t, max_retries=3, sleep_fn=lambda s: None)
    with caplog.at_level(logging.WARNING, logger="src.llm_client"):
        client.chat("m", [{"role": "user", "content": "q"}])
    retry_msgs = [r for r in caplog.records if "retry" in r.message.lower()]
    assert len(retry_msgs) == 2, "one warning log per retry attempt"


# --------------------------------------------------------------------------
# Thread-safe accounting (stages 3 + 5 will fan out concurrent LLM calls)
# --------------------------------------------------------------------------
def test_accounting_is_threadsafe_under_concurrent_calls():
    """The accounting block must use a lock so call_count + cost don't drop updates."""
    n_threads = 12
    n_calls_per_thread = 50
    n_total = n_threads * n_calls_per_thread

    class ThreadsafeTransport:
        def __init__(self):
            self.lock = threading.Lock()
            self.served = 0

        def __call__(self, url, headers, payload):
            with self.lock:
                self.served += 1
            return (200, _ok_body(cost=0.001, ptok=1, ctok=1))

    t = ThreadsafeTransport()
    client = LLMClient(api_key="k", request_fn=t)

    def worker():
        for _ in range(n_calls_per_thread):
            client.chat("m", [{"role": "user", "content": "q"}])

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert client.call_count == n_total, "no concurrent update may be lost"
    assert client.total_cost == pytest.approx(0.001 * n_total)
    assert client.total_tokens["completion"] == n_total


# --------------------------------------------------------------------------
# Cost & token accounting
# --------------------------------------------------------------------------
def test_cost_and_token_accounting_accumulates():
    t = FakeTransport([(200, _ok_body(cost=0.01, ptok=100, ctok=50)),
                       (200, _ok_body(cost=0.02, ptok=200, ctok=20))])
    client = LLMClient(api_key="k", request_fn=t)
    client.chat("m", [{"role": "user", "content": "q"}])
    client.chat("m", [{"role": "user", "content": "q"}])
    assert client.total_cost == pytest.approx(0.03)
    assert client.call_count == 2
    assert client.total_tokens["completion"] == 70
    assert client.total_tokens["prompt"] == 300


# --------------------------------------------------------------------------
# API-key handling
# --------------------------------------------------------------------------
def test_missing_api_key_with_default_transport_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = LLMClient(api_key=None)  # default (real) transport
    with pytest.raises(LLMError):
        client.chat("m", [{"role": "user", "content": "q"}])
