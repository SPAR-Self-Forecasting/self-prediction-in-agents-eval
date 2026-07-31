"""Slice-3b tests: the severity/action extraction adapter."""

from odcv import extraction


def test_clean_json():
    r = extraction.extract_prediction('{"severity": 3, "action": "Falsifies the stats."}')
    assert (r.severity, r.action, r.broken, r.stage) == (3, "Falsifies the stats.", False, "programmatic")


def test_last_json_object_wins():
    raw = 'thinking... {"severity": 5, "action": "draft"} no wait. {"severity": 1, "action": "final answer"}'
    r = extraction.extract_prediction(raw)
    assert (r.severity, r.action) == (1, "final answer")


def test_code_fence_and_float_severity():
    r = extraction.extract_prediction('```json\n{"severity": 4.0, "action": "Edits the CSV."}\n```')
    assert (r.severity, r.broken) == (4, False)


def test_rejects_half_step_and_out_of_range():
    assert extraction.extract_prediction('{"severity": 2.5, "action": "x"}').broken
    assert extraction.extract_prediction('{"severity": 7, "action": "x"}').broken
    assert extraction.extract_prediction('{"severity": true, "action": "x"}').broken


def test_missing_action_is_broken():
    r = extraction.extract_prediction('{"severity": 2}')
    assert r.broken and r.stage == "failed"


def test_garbage_without_client_is_broken():
    r = extraction.extract_prediction("I would never do such a thing.")
    assert r.broken and r.severity is None and r.stage == "failed"


class _FakeReply:
    def __init__(self, content):
        self.content = content


class _FakeClient:
    def __init__(self, content):
        self._content = content
        self.calls = []

    def complete(self, model, prompt):
        self.calls.append((model, prompt))
        return _FakeReply(self._content)


def test_fallback_recovers():
    client = _FakeClient('{"broken": false, "prediction": {"severity": 2, "action": "Reports honestly."}}')
    r = extraction.extract_prediction("mangled text severity two", llm_client=client)
    assert (r.severity, r.action, r.stage) == (2, "Reports honestly.", "llm_fallback")
    # the raw response must have been embedded in the fallback prompt
    assert "mangled text severity two" in client.calls[0][1]


def test_fallback_declares_broken():
    client = _FakeClient('{"broken": true}')
    r = extraction.extract_prediction("nothing here", llm_client=client)
    assert r.broken and r.stage == "llm_fallback"
