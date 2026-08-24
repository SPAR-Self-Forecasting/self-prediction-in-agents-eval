"""Slice-3b tests: call plans, checkpoint/resume, and the full loop on a mock
transport (zero API calls -- LLMClient's injectable request_fn)."""

import json

import pytest

from src.llm_client import LLMClient

from odcv import config, elicit, fidelity_gate


# --------------------------------------------------------------------------
# Call plans (spec §3 steps 4 + 5 counts)
# --------------------------------------------------------------------------
def test_full_plan_counts():
    plan = elicit.build_call_plan("full")
    assert len(plan) == 3_580
    by_condition = {}
    for c in plan:
        by_condition[c["condition"]] = by_condition.get(c["condition"], 0) + 1
    assert by_condition == {"self": 1_790, "simulated_cross": 1_790}


def test_smoke_plan_counts():
    plan = elicit.build_call_plan("smoke")
    assert len(plan) == 25  # 20 + 5 probe
    assert sum(c["model"] == elicit.PROBE_MODEL for c in plan) == 5


def test_smoke_scenarios_span_severity():
    chosen = elicit.smoke_scenarios()
    assert len(chosen) == len(set(chosen)) == 5


def test_grok_carries_reasoning_enable():
    plan = elicit.build_call_plan("smoke")
    grok = [c for c in plan if c["model"] == "x-ai/grok-4.20"]
    assert grok and all(c["reasoning"] == {"enabled": True} for c in grok)
    glm = [c for c in plan if c["model"] == "z-ai/glm-5.1"]
    assert glm and all(c["reasoning"] is None for c in glm)


# --------------------------------------------------------------------------
# Mock-transport end-to-end (checkpoint, resume, record schema)
# --------------------------------------------------------------------------
def _mock_request_fn(url, headers, payload):
    body = {
        "choices": [{"message": {
            "content": '{"severity": 1, "action": "Does the honest analysis and reports it."}',
            "reasoning": "mock chain of thought",
        }}],
        "model": payload["model"],
        "provider": "MockProvider",
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cost": 0.001},
    }
    return 200, {}, body


@pytest.fixture()
def mock_client():
    return LLMClient(api_key="test-key", request_fn=_mock_request_fn)


def test_run_smoke_on_mock(tmp_path, mock_client):
    path = tmp_path / "predictions.jsonl"
    summary = elicit.run(
        "smoke", client=mock_client, summaries=fidelity_gate.fixture_summaries(),
        max_workers=2, path=path,
    )
    assert summary["ran"] == 25 and summary["broken"] == 0
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 25
    r = records[0]
    for field in ("model", "scenario", "condition", "sample_idx", "raw_response",
                  "reasoning_content", "severity", "action", "broken", "stage",
                  "provider", "prompt_tokens", "completion_tokens", "cost"):
        assert field in r, field
    assert r["provider"] == "MockProvider"          # D34
    assert r["reasoning_content"] == "mock chain of thought"
    assert r["severity"] == 1 and r["broken"] is False


def test_resume_skips_done(tmp_path, mock_client):
    path = tmp_path / "predictions.jsonl"
    fixtures = fidelity_gate.fixture_summaries()
    elicit.run("smoke", client=mock_client, summaries=fixtures, max_workers=2, path=path)
    calls_before = mock_client.call_count
    again = elicit.run("smoke", client=mock_client, summaries=fixtures, max_workers=2, path=path)
    assert again["ran"] == 0
    assert mock_client.call_count == calls_before  # zero re-spend (C5)


def test_temperature_and_fresh_context(monkeypatch):
    """Every call is a single-message conversation at TEMPERATURE."""
    seen = []

    def spy_request_fn(url, headers, payload):
        seen.append(payload)
        return _mock_request_fn(url, headers, payload)

    client = LLMClient(api_key="k", request_fn=spy_request_fn)
    call = elicit.build_call_plan("smoke")[0]
    elicit.run_one(call, fidelity_gate.fixture_summaries(), client)
    payload = seen[0]
    assert payload["temperature"] == config.TEMPERATURE == 0.7
    assert len(payload["messages"]) == 1  # fresh context, no carried framings
