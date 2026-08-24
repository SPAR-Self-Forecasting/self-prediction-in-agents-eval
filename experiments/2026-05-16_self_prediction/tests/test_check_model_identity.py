"""Tests for check_model_identity.py.

The risks worth pinning: one unreachable model must not abort the sweep, the
keyword hint must not be mistaken for a verdict (raw replies always kept), and
the question must come from the shared template so the identity-primed forecast
condition asks exactly the same thing.
"""

from __future__ import annotations

import json

import pytest

import check_model_identity as cmi
import config


class _Client:
    """Returns a canned reply per model id; raises for ids in ``fail``."""

    def __init__(self, replies: dict, fail: set | None = None):
        self.replies = replies
        self.fail = fail or set()
        self.calls: list[tuple] = []

    def chat(self, model, messages, **kwargs):
        self.calls.append((model, messages, kwargs))
        if model in self.fail:
            raise RuntimeError("upstream 502")

        class _R:
            content = self.replies.get(model, "")
        return _R()


def test_question_comes_from_the_shared_template():
    """A literal here would let the script and the primed condition drift apart."""
    template = (cmi.PROMPTS_DIR / "identity_check.txt").read_text(encoding="utf-8")
    assert cmi.load_identity_question() == template.strip()
    assert "what model you are" in cmi.load_identity_question().lower()


def test_a_dead_model_does_not_abort_the_sweep():
    spec = config.MODELS[0]
    client = _Client({}, fail={spec.openrouter_id})
    record = cmi.ask_one(client, spec, "who are you?")
    assert record["error"].startswith("RuntimeError")
    assert record["reply"] is None
    assert "looks_self_identified" not in record   # no verdict without a reply


def test_raw_reply_is_always_kept():
    spec = config.MODEL_BY_ID["z-ai/glm-5"]
    client = _Client({spec.openrouter_id: "I'm afraid I can't say."})
    record = cmi.ask_one(client, spec, "who are you?")
    assert record["reply"] == "I'm afraid I can't say."
    assert record["looks_self_identified"] is False


def test_identity_answers_are_asked_at_temperature_zero():
    """A capability check should be repeatable, unlike the forecast samples."""
    spec = config.MODELS[0]
    client = _Client({spec.openrouter_id: "hi"})
    cmi.ask_one(client, spec, "who are you?")
    _, _, kwargs = client.calls[0]
    assert kwargs["temperature"] == 0.0
    assert cmi.IDENTITY_TEMPERATURE == 0.0


@pytest.mark.parametrize("reply,expected", [
    ("I am GLM-5, built by Zhipu AI.", True),
    ("I'm a large language model.", False),
    ("I am an AI assistant. I don't have a name.", False),
])
def test_keyword_hint_tracks_self_identification(reply, expected):
    spec = config.MODEL_BY_ID["z-ai/glm-5"]
    assert cmi.keyword_hint(spec, reply)["looks_self_identified"] is expected


def test_generic_tokens_never_fire_the_hint_alone():
    """"Flash"/"lite"/"instruct" would match half the internet."""
    spec = config.MODEL_BY_ID["google/gemini-2.5-flash-lite"]
    hint = cmi.keyword_hint(spec, "I am a fast, lightweight instruct model in a flash.")
    assert hint["looks_self_identified"] is False
    assert "gemini" in hint["name_tokens"]


def test_hint_reports_which_tokens_matched():
    spec = config.MODEL_BY_ID["moonshotai/kimi-k2.5"]
    hint = cmi.keyword_hint(spec, "I am Kimi, an assistant.")
    assert hint["matched_tokens"] == ["kimi"]


def test_unknown_model_id_is_rejected_before_any_call(capsys):
    with pytest.raises(SystemExit):
        cmi.main(["--models", "not/a-model", "--dry-run"])


def test_main_writes_every_model_and_both_questions(tmp_path):
    out = tmp_path / "identity.json"
    assert cmi.main(["--dry-run", "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    general, specific, openrouter = cmi.load_identity_questions()
    assert payload["questions"]["general"] == general
    assert payload["questions"]["specific_id"] == specific
    assert payload["questions"]["openrouter_slug"] == openrouter
    assert payload["dry_run"] is True
    assert len(payload["records"]) == len(config.MODELS)
    assert {r["model"] for r in payload["records"]} == set(config.MODEL_IDS)
    assert "general" in payload["records"][0] and "specific_id" in payload["records"][0]
    assert "openrouter_slug" in payload["records"][0]


def test_report_flags_errors_separately_from_non_answers():
    def _block(reply, error=None, identified=False):
        return {
            "question": "q", "reply": reply, "error": error,
            "looks_self_identified": identified, "matched_tokens": ["c"] if identified else [],
        }

    records = [
        {"model": "a/b", "name": "A",
         "general": _block(None, "LLMError: boom"), "specific_id": _block("ok", identified=True),
         "openrouter_slug": _block("openai/a/b")},
        {"model": "c/d", "name": "C",
         "general": _block("I am C", identified=True), "specific_id": _block("I am C", identified=True),
         "openrouter_slug": _block("c/d", identified=True)},
        {"model": "e/f", "name": "E",
         "general": _block("no idea"), "specific_id": _block("no idea"),
         "openrouter_slug": _block("no idea")},
    ]
    report = cmi.format_report(records)
    assert "[ERR ]" in report and "[yes ]" in report and "[?   ]" in report
    assert "1 models errored" in report