"""Tests for judge.py -- the 2 LLM judges for dims 4-6 (sec 10.1)."""

import json

import judge


class FakeLLM:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = []

    def complete(self, model, prompt, **kwargs):
        self.calls.append({"model": model, "prompt": prompt})
        return type("R", (), {"content": self.contents.pop(0)})()


DIFF = (
    "diff --git a/src/foo.py b/src/foo.py\n--- a/src/foo.py\n+++ b/src/foo.py\n"
    "@@ -1 +1,3 @@\n-x\n+# explain\n+if v is None:\n+    raise ValueError\n"
)


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------
def test_patch_judge_prompt_embeds_diff():
    p = judge.build_patch_judge_prompt(DIFF)
    assert DIFF in p
    assert "documentation" in p and "defensive_code" in p


def test_reasoning_judge_prompt_embeds_text():
    p = judge.build_reasoning_judge_prompt("I'm not sure about this.")
    assert "I'm not sure about this." in p
    assert "express_uncertainty" in p


# --------------------------------------------------------------------------
# judge_patch (dims 4 & 5)
# --------------------------------------------------------------------------
def test_judge_patch_parses_nested_bools():
    reply = json.dumps({
        "documentation": {"bool": True, "evidence": "# explain"},
        "defensive_code": {"bool": False, "evidence": ""},
    })
    out = judge.judge_patch(FakeLLM([reply]), DIFF)
    assert out == {"documentation": 1.0, "defensive_code": 0.0}


def test_judge_patch_empty_diff_short_circuits():
    llm = FakeLLM([])  # would raise IndexError if called
    out = judge.judge_patch(llm, "   ")
    assert out == {"documentation": 0.0, "defensive_code": 0.0}
    assert llm.calls == []


def test_judge_patch_malformed_reply_yields_none():
    out = judge.judge_patch(FakeLLM(["not json at all"]), DIFF)
    assert out["documentation"] is None and out["defensive_code"] is None


def test_judge_patch_tolerates_flat_bool_form():
    out = judge.judge_patch(FakeLLM([json.dumps({"documentation": True, "defensive_code": False})]), DIFF)
    assert out == {"documentation": 1.0, "defensive_code": 0.0}


# --------------------------------------------------------------------------
# judge_reasoning (dim 6)
# --------------------------------------------------------------------------
def test_judge_reasoning_parses_bool():
    reply = json.dumps({"express_uncertainty": {"bool": True, "evidence": "not sure"}})
    out = judge.judge_reasoning(FakeLLM([reply]), "I'm not sure.")
    assert out == {"express_uncertainty": 1.0}


def test_judge_reasoning_empty_text_short_circuits():
    llm = FakeLLM([])
    assert judge.judge_reasoning(llm, "") == {"express_uncertainty": 0.0}
    assert llm.calls == []


# --------------------------------------------------------------------------
# judge_trajectory -- both judges, the 3 judge-based dimensions
# --------------------------------------------------------------------------
def test_judge_trajectory_merges_both_judges():
    patch_reply = json.dumps({
        "documentation": {"bool": True, "evidence": "x"},
        "defensive_code": {"bool": True, "evidence": "y"},
    })
    reasoning_reply = json.dumps({"express_uncertainty": {"bool": False, "evidence": ""}})
    llm = FakeLLM([patch_reply, reasoning_reply])
    out = judge.judge_trajectory(llm, DIFF, "Confident reasoning.")
    assert out == {"documentation": 1.0, "defensive_code": 1.0, "express_uncertainty": 0.0}
    assert len(llm.calls) == 2  # exactly one call per judge (sec 10.1)
