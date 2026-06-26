"""Tests for verify_prelaunch.py -- the sec 10.6 pre-launch verification.

The check functions take an injectable client, so this exercises them with the
mock client and never touches the network.
"""

import config
import verify_prelaunch
from run import MockLLMClient


class FailingClient:
    """A client that always errors -- simulates an unreachable provider."""

    def complete(self, model, prompt, **kwargs):
        raise ConnectionError(f"provider {model} unreachable")


def test_verify_passes_with_mock_client():
    report = verify_prelaunch.verify(MockLLMClient(), config.MODELS[:4])
    assert report["all_ok"] is True
    assert report["n_models"] == 4
    for outcome in report["models"].values():
        assert set(outcome["conditions"]) == {"no_thinking", "thinking"}
        for entry in outcome["conditions"].values():
            assert entry["ok"] and entry["n_dimensions"] == 10   # A6: 10 dims (8 + mean_action_block_size, n_constraint_violations)


def test_verify_flags_unreachable_provider():
    report = verify_prelaunch.verify(FailingClient(), config.MODELS[:2])
    assert report["all_ok"] is False
    for outcome in report["models"].values():
        assert outcome["ok"] is False
        assert all(not e["reachable"] for e in outcome["conditions"].values())


def test_check_model_reports_both_conditions():
    result = verify_prelaunch.check_model(MockLLMClient(), config.model_by_id("qwen/qwen3-8b"))
    assert result["thinking_class"] == "hybrid"
    assert result["ok"] is True
