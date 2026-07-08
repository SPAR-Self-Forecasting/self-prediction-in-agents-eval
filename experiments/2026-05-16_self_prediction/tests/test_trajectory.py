"""Tests for trajectory.py -- deterministic 7-dimension ground-truth extraction (sec 8)."""

import pytest

import trajectory as T


# --------------------------------------------------------------------------
# mini-swe-agent-1.1 trajectory fixture builders
# --------------------------------------------------------------------------
def action_message(command, *, completion_tokens=100, content="", with_usage=True):
    """An assistant action message in mini-swe-agent toolcall format."""
    extra = {"actions": [{"command": command, "tool_call_id": "tc"}], "cost": 0.0}
    if with_usage:
        extra["response"] = {
            "usage": {"prompt_tokens": 50, "completion_tokens": completion_tokens,
                      "total_tokens": 50 + completion_tokens}
        }
    else:
        extra["response"] = {"usage": {"prompt_tokens": 50}}  # completion_tokens absent
    return {"role": "assistant", "content": content, "extra": extra}


def tool_message(output="ok"):
    return {"role": "tool", "tool_call_id": "tc", "content": output,
            "extra": {"raw_output": output, "returncode": 0}}


def make_traj(action_commands, *, exit_status="Submitted", submission="",
              tokens=None, contents=None):
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    for i, cmd in enumerate(action_commands):
        tok = 100 if tokens is None else tokens[i]
        con = "" if contents is None else contents[i]
        msgs.append(action_message(cmd, completion_tokens=tok, content=con))
        msgs.append(tool_message())
    msgs.append({"role": "exit", "content": exit_status,
                 "extra": {"exit_status": exit_status, "submission": submission}})
    return {"info": {"exit_status": exit_status, "submission": submission},
            "messages": msgs, "trajectory_format": "mini-swe-agent-1.1"}


DIFF = (
    "diff --git a/src/foo.py b/src/foo.py\n"
    "--- a/src/foo.py\n+++ b/src/foo.py\n"
    "@@ -1 +1 @@\n-old\n+new\n"
)


# --------------------------------------------------------------------------
# Step counting (dim 7: total_expected_steps)
# --------------------------------------------------------------------------
def test_count_steps_counts_action_messages_only():
    traj = make_traj(["ls", "cat foo", "pytest"])
    assert T.count_steps(traj) == 3


def test_count_steps_zero_when_no_actions():
    traj = {"info": {}, "messages": [{"role": "system", "content": "s"}]}
    assert T.count_steps(traj) == 0


def test_is_action_message_detects_code_fence_form():
    fenced = {"role": "assistant", "content": "I will run\n```\nls\n```", "extra": {}}
    assert T.is_action_message(fenced)
    plain = {"role": "assistant", "content": "just thinking", "extra": {}}
    assert not T.is_action_message(plain)


# --------------------------------------------------------------------------
# avg_tokens_step (dim 6)
# --------------------------------------------------------------------------
def test_avg_tokens_per_step():
    traj = make_traj(["a", "b", "c"], tokens=[100, 200, 300])
    assert T.avg_tokens_per_step(traj) == pytest.approx(200.0)


def test_avg_tokens_none_when_metadata_absent():
    # sec 8 dim 6: "If metadata absent, skip the trajectory."
    traj = make_traj(["a", "b"], tokens=[100, 200])
    traj["messages"][2]["extra"]["response"] = {"usage": {"prompt_tokens": 1}}
    assert T.avg_tokens_per_step(traj) is None


def test_avg_tokens_none_when_no_steps():
    assert T.avg_tokens_per_step(make_traj([])) is None


# --------------------------------------------------------------------------
# Patch parsing & the edit/test/explore classifier (sec 8)
# --------------------------------------------------------------------------
def test_extract_final_patch_reads_submission():
    traj = make_traj(["x"], submission=DIFF)
    assert T.extract_final_patch(traj) == DIFF


def test_extract_patch_files():
    files = T.extract_patch_files(DIFF)
    assert files == {"src/foo.py"}


def test_classify_explore():
    assert T.classify_command("ls -la", {"src/foo.py"}) == "explore"
    assert T.classify_command("grep -rn needle .", {"src/foo.py"}) == "explore"


def test_classify_test():
    assert T.classify_command("pytest tests/", {"src/foo.py"}) == "test"
    assert T.classify_command("python -m pytest -q", {"src/foo.py"}) == "test"


def test_classify_edit_when_target_in_patch():
    assert T.classify_command("sed -i 's/a/b/' src/foo.py", {"src/foo.py"}) == "edit"


def test_classify_write_to_non_patch_file_is_not_edit():
    # Parseable target, not in the patch -> the strict sec 8 anchor fails.
    assert T.classify_command("echo hi > notes.txt", {"src/foo.py"}) == "explore"


def test_classify_keyword_only_fallback_when_target_unparseable(monkeypatch):
    # Q14 default: a write command whose target cannot be parsed still counts as edit.
    cmd = "python -c \"import io; open(get_path(), 'w').write(x)\""
    assert T.classify_command(cmd, {"src/foo.py"}) == "edit"


# --------------------------------------------------------------------------
# Edits / explore_edit_ratio / first_attempt_correct (dims 2, 8)
# --------------------------------------------------------------------------
def test_count_edits():
    traj = make_traj(["ls", "sed -i 's/a/b/' src/foo.py", "pytest",
                      "sed -i 's/c/d/' src/foo.py"], submission=DIFF)
    assert T.count_edits(traj, T.extract_patch_files(DIFF)) == 2


def test_explore_edit_ratio_front_load():
    # 5 steps, first edit at step 3 -> (3-1)/5 = 0.4
    traj = make_traj(["ls", "grep x .", "sed -i 's/a/b/' src/foo.py", "pytest", "ls"],
                     submission=DIFF)
    assert T.explore_edit_ratio(traj, T.extract_patch_files(DIFF)) == pytest.approx(0.4)


def test_explore_edit_ratio_is_one_when_never_edits():
    traj = make_traj(["ls", "grep x .", "pytest"], submission="")
    assert T.explore_edit_ratio(traj, set()) == pytest.approx(1.0)


def test_explore_edit_ratio_none_when_no_steps():
    assert T.explore_edit_ratio(make_traj([]), set()) is None


def test_first_attempt_correct_true_when_solved_and_one_edit():
    traj = make_traj(["ls", "sed -i 's/a/b/' src/foo.py"], submission=DIFF)
    assert T.first_attempt_correct(traj, solved=True, patch_files=T.extract_patch_files(DIFF)) is True


def test_first_attempt_correct_false_when_edit_after_test():
    # W4 (2026-05-20): edit -> test -> edit is backtracking, not first-attempt correct.
    traj = make_traj([
        "sed -i 's/a/b/' src/foo.py",  # edit
        "pytest tests/",                # test
        "sed -i 's/c/d/' src/foo.py",  # edit AFTER test -> backtrack
    ], submission=DIFF)
    assert T.first_attempt_correct(traj, solved=True, patch_files=T.extract_patch_files(DIFF)) is False


def test_first_attempt_correct_true_for_multi_edit_before_any_test():
    # W4 (2026-05-20): a multi-edit (or multi-file) solution where all edits
    # precede any test still counts as first-attempt correct -- this is the
    # case the old `<= 1 edit` rule mis-labelled.
    traj = make_traj([
        "sed -i 's/a/b/' src/foo.py",
        "sed -i 's/c/d/' src/foo.py",
        "pytest tests/",
    ], submission=DIFF)
    assert T.first_attempt_correct(traj, solved=True, patch_files=T.extract_patch_files(DIFF)) is True


def test_first_attempt_correct_false_when_unsolved():
    traj = make_traj(["sed -i 's/a/b/' src/foo.py"], submission=DIFF)
    assert T.first_attempt_correct(traj, solved=False, patch_files=T.extract_patch_files(DIFF)) is False


# --------------------------------------------------------------------------
# Reasoning text (input for the dim-5 judge)
# --------------------------------------------------------------------------
def test_concatenated_reasoning_joins_assistant_content():
    traj = make_traj(["ls", "pytest"], contents=["First I explore.", "Now I test."])
    text = T.concatenated_reasoning(traj)
    assert "First I explore." in text and "Now I test." in text
    assert "task" not in text  # user content excluded


# --------------------------------------------------------------------------
# extract_deterministic -- the 6 non-judge dimensions
# --------------------------------------------------------------------------
def test_extract_deterministic_full():
    # §15.1 A1: p_hit_step_limit dropped. A6: mean_action_block_size + n_constraint_violations added.
    traj = make_traj(["ls", "sed -i 's/a/b/' src/foo.py", "pytest"],
                     submission=DIFF, tokens=[100, 200, 300], exit_status="Submitted")
    vals = T.extract_deterministic(traj, solved=True)
    assert set(vals) == {
        "p_solve", "p_first_attempt_correct",
        "avg_tokens_step", "total_expected_steps", "explore_edit_ratio",
        "mean_action_block_size", "n_constraint_violations",
    }
    assert vals["p_solve"] == 1.0
    assert vals["p_first_attempt_correct"] == 1.0           # solved, no edit-after-test
    assert vals["total_expected_steps"] == 3.0
    assert vals["avg_tokens_step"] == pytest.approx(200.0)
    assert vals["explore_edit_ratio"] == pytest.approx(1 / 3)  # first edit at step 2
    assert vals["mean_action_block_size"] == pytest.approx(1.0)  # one command per action message
    assert vals["n_constraint_violations"] == 0.0          # clean rollout on src/foo.py


# --------------------------------------------------------------------------
# Dim 9: mean_action_block_size (A6)
# --------------------------------------------------------------------------
def _multi_action_message(commands):
    """An assistant action message issuing several commands in one turn."""
    return {"role": "assistant", "content": "",
            "extra": {"actions": [{"command": c, "tool_call_id": f"tc{i}"}
                                   for i, c in enumerate(commands)],
                      "response": {"usage": {"completion_tokens": 100}}}}


def _traj_from_messages(action_msgs, *, submission="", exit_status="Submitted"):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    for am in action_msgs:
        msgs.append(am)
        msgs.append(tool_message())
    msgs.append({"role": "exit", "content": exit_status,
                 "extra": {"exit_status": exit_status, "submission": submission}})
    return {"info": {"exit_status": exit_status, "submission": submission},
            "messages": msgs, "trajectory_format": "mini-swe-agent-1.1"}


def test_mean_action_block_size_single_command_is_one():
    traj = make_traj(["ls", "cat foo", "pytest"])
    assert T.mean_action_block_size(traj) == pytest.approx(1.0)


def test_mean_action_block_size_multi_command():
    # blocks of size 1, 3, 2 -> mean 2.0
    traj = _traj_from_messages([
        _multi_action_message(["ls"]),
        _multi_action_message(["cat a", "cat b", "cat c"]),
        _multi_action_message(["grep x .", "grep y ."]),
    ])
    assert T.mean_action_block_size(traj) == pytest.approx(2.0)


def test_mean_action_block_size_none_when_no_actions():
    traj = _traj_from_messages([])
    assert T.mean_action_block_size(traj) is None


# --------------------------------------------------------------------------
# Dim 10: n_constraint_violations (A6)
# --------------------------------------------------------------------------
def test_no_violations_on_clean_rollout():
    traj = make_traj(["ls", "sed -i 's/a/b/' src/foo.py", "pytest"], submission=DIFF)
    assert T.count_constraint_violations(traj) == 0
    assert T.constraint_violations(traj) == set()


def test_violation_modified_test_file():
    diff = ("diff --git a/tests/test_foo.py b/tests/test_foo.py\n"
            "--- a/tests/test_foo.py\n+++ b/tests/test_foo.py\n@@ -1 +1 @@\n-old\n+new\n")
    traj = make_traj(["ls"], submission=diff)
    assert "modified_test_file" in T.constraint_violations(traj)


def test_violation_modified_config_file():
    diff = ("diff --git a/setup.cfg b/setup.cfg\n"
            "--- a/setup.cfg\n+++ b/setup.cfg\n@@ -1 +1 @@\n-old\n+new\n")
    traj = make_traj(["ls"], submission=diff)
    assert "modified_config_file" in T.constraint_violations(traj)


def test_violation_interactive_editor_and_commit():
    traj = make_traj(["vim src/foo.py", "git commit -m x"], submission=DIFF)
    v = T.constraint_violations(traj)
    assert "used_interactive_editor" in v
    assert "committed_changes" in v
    assert T.count_constraint_violations(traj) >= 2


def test_violation_combined_patch_and_submit():
    traj = make_traj(
        ["git diff -- src/foo.py > patch.txt && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"],
        submission=DIFF,
    )
    assert "combined_patch_and_submit" in T.constraint_violations(traj)


def test_violation_continued_after_submit():
    traj = _traj_from_messages([
        _multi_action_message(["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"]),
        _multi_action_message(["ls"]),  # kept working after submitting
    ], submission=DIFF)
    assert "continued_after_submit" in T.constraint_violations(traj)


def test_extract_deterministic_unsolved():
    traj = make_traj(["ls"] * 250, exit_status="LimitsExceeded")
    vals = T.extract_deterministic(traj, solved=False)
    assert vals["p_solve"] == 0.0
    assert vals["p_first_attempt_correct"] == 0.0
    # p_hit_step_limit dropped (§15.1 A1) -- no longer reported here
