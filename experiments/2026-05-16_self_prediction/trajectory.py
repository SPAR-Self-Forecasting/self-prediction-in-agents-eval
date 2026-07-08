"""Deterministic ground-truth extraction from mini-swe-agent trajectories (sec 8).

Reads the canonical ``mini-swe-agent-1.1`` trajectory structure -- ``info`` +
``messages``, with assistant action messages carrying ``extra.actions`` and
``extra.response.usage`` -- and extracts the 7 deterministically-measured
dimensions:

    p_solve, p_first_attempt_correct, avg_tokens_step, total_expected_steps,
    explore_edit_ratio, mean_action_block_size, n_constraint_violations

The 3 judge-based dimensions (documentation, defensive_code, express_uncertainty)
are produced separately by ``judge.py``; this module also supplies that judge
its inputs -- the final patch diff and the concatenated reasoning text.

The edit / test / explore command classifier (sec 8) is shared by the
``first_attempt_correct`` and ``explore_edit_ratio`` extractors. Edit detection
anchors a write command to a file in the final patch where the target is
parseable, and falls back to keyword-only when it is not (Phase-1 finding /
clarification Q14).
"""

from __future__ import annotations

import os
import re

# --------------------------------------------------------------------------
# Command-classifier regexes
# --------------------------------------------------------------------------
# A keyword is a *command* (not an argument) only at the start of a command:
# string start, or after a newline / ; / | / & / && / ||.  This is what keeps
# `cat patch.txt` (patch as an argument) from being read as a write command.
_CMD_START = r"(?:^|[\n;|&])\s*(?:sudo\s+)?"

_RE_SED_I = re.compile(_CMD_START + r"sed\s+-i", re.MULTILINE)
_RE_WRITE_CMD = re.compile(_CMD_START + r"(?:patch|tee|vim|nano|emacs|dd)\b", re.MULTILINE)
_RE_GIT_APPLY = re.compile(_CMD_START + r"git\s+apply\b", re.MULTILINE)
_RE_APPLY_PATCH = re.compile(r"\bapply_patch\b")
# A real output redirection: `>` or `>>` not part of an fd-dup like `2>&1`.
_RE_REDIRECT = re.compile(r"(?<![0-9&>])>>?(?![&>])")
_RE_PY_OPEN_WRITE = re.compile(r"open\s*\([^)]*['\"][wax]")
_RE_PY_WRITE = re.compile(r"\.write(?:lines)?\s*\(")

_RE_TEST = re.compile(
    r"\bpytest\b|\bunittest\b|\bnosetests\b|\btox\b|run_tests|reproduce\S*\.py", re.IGNORECASE
)

# target extraction
_RE_T_REDIRECT = re.compile(r"(?<![0-9&>])>>?(?![&>])\s*([^\s;&|<>]+)")
_RE_T_TEE = re.compile(r"\btee\s+(?:-\S+\s+)*([^\s;&|]+)")
_RE_T_EDITOR = re.compile(r"\b(?:vim|nano|emacs)\s+(?:-\S+\s+)*([^\s;&|]+)")
_RE_T_SED = re.compile(r"\bsed\s+-i\S*\s+(.+)")

_RE_CODE_FENCE = re.compile(r"```(?:[^\n`]*)\n?(.*?)```", re.DOTALL)
_RE_DIFF_GIT = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.MULTILINE)
_RE_DIFF_FILE = re.compile(r"^[+-]{3} [ab]/(\S+)", re.MULTILINE)

_ASSISTANT_ROLES = ("assistant", "model")


# --------------------------------------------------------------------------
# Trajectory accessors
# --------------------------------------------------------------------------
def get_messages(traj: dict) -> list[dict]:
    return traj.get("messages", []) or []


def get_info(traj: dict) -> dict:
    return traj.get("info", {}) or {}


def _has_code_fence(content) -> bool:
    return isinstance(content, str) and content.count("```") >= 2


def is_action_message(msg: dict) -> bool:
    """True iff ``msg`` is an assistant message that issued at least one action.

    Canonically detected via a non-empty ``extra.actions`` (mini-swe-agent v2
    toolcalls); a markdown code fence is the v1 / text-based fallback (sec 8).
    """
    if msg.get("role") not in _ASSISTANT_ROLES:
        return False
    if msg.get("extra", {}).get("actions"):
        return True
    return _has_code_fence(msg.get("content"))


def action_messages(traj: dict) -> list[dict]:
    return [m for m in get_messages(traj) if is_action_message(m)]


def commands_of(msg: dict) -> list[str]:
    """The bash command strings issued by an action message."""
    actions = msg.get("extra", {}).get("actions") or []
    if actions:
        return [a.get("command", "") for a in actions]
    return [m.strip() for m in _RE_CODE_FENCE.findall(msg.get("content") or "")]


# --------------------------------------------------------------------------
# Dim 7: total_expected_steps
# --------------------------------------------------------------------------
def count_steps(traj: dict) -> int:
    """Count of assistant action messages (sec 2 dim 7)."""
    return len(action_messages(traj))


# --------------------------------------------------------------------------
# Dim 6: avg_tokens_step
# --------------------------------------------------------------------------
def _completion_tokens(msg: dict) -> int | None:
    usage = msg.get("extra", {}).get("response", {}).get("usage", {}) or {}
    tok = usage.get("completion_tokens")
    return int(tok) if isinstance(tok, (int, float)) and not isinstance(tok, bool) else None


def avg_tokens_per_step(traj: dict) -> float | None:
    """Mean completion tokens per action message, or None.

    Per sec 8, if usage metadata is absent the trajectory is skipped for this
    dimension -- any action message lacking ``completion_tokens`` (or no action
    messages at all) yields None.
    """
    msgs = action_messages(traj)
    if not msgs:
        return None
    tokens = [_completion_tokens(m) for m in msgs]
    if any(t is None for t in tokens):
        return None
    return float(sum(tokens)) / len(tokens)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Patch parsing
# --------------------------------------------------------------------------
def extract_final_patch(traj: dict) -> str:
    """The final submitted patch diff (``info.submission``)."""
    submission = get_info(traj).get("submission")
    if submission:
        return str(submission)
    for msg in reversed(get_messages(traj)):
        if msg.get("role") == "exit":
            return str(msg.get("extra", {}).get("submission", "") or "")
    return ""


def extract_patch_files(diff: str) -> set[str]:
    """The set of file paths touched by a unified-diff patch."""
    files: set[str] = set()
    for a, b in _RE_DIFF_GIT.findall(diff or ""):
        files.update((a, b))
    files.update(_RE_DIFF_FILE.findall(diff or ""))
    files.discard("dev/null")
    return files


# --------------------------------------------------------------------------
# Edit / test / explore command classifier (sec 8)
# --------------------------------------------------------------------------
def _is_write_command(cmd: str) -> bool:
    return bool(
        _RE_SED_I.search(cmd)
        or _RE_WRITE_CMD.search(cmd)
        or _RE_GIT_APPLY.search(cmd)
        or _RE_APPLY_PATCH.search(cmd)
        or _RE_REDIRECT.search(cmd)
        or _RE_PY_OPEN_WRITE.search(cmd)
        or _RE_PY_WRITE.search(cmd)
    )


def _is_test_command(cmd: str) -> bool:
    return bool(_RE_TEST.search(cmd))


def _extract_write_target(cmd: str) -> str | None:
    """Best-effort parse of the file a write command targets; None if unparseable."""
    for regex in (_RE_T_REDIRECT, _RE_T_TEE, _RE_T_EDITOR):
        m = regex.search(cmd)
        if m:
            return m.group(1).strip("'\"")
    m = _RE_T_SED.search(cmd)
    if m:
        tokens = m.group(1).split()
        if tokens:
            return tokens[-1].strip("'\"")
    return None


def _target_matches_patch(target: str, patch_files: set[str]) -> bool:
    t = target.lstrip("./")
    base = os.path.basename(t)
    for f in patch_files:
        if f == t or f.endswith("/" + t) or t.endswith(f) or os.path.basename(f) == base:
            return True
    return False


def classify_command(command: str, patch_files: set[str]) -> str:
    """Classify a bash command as ``edit`` | ``test`` | ``explore`` (sec 8).

    A write command counts as an *edit* when its target is a file in the final
    patch, or when the target cannot be parsed at all (keyword-only fallback,
    Q14). A write to a parseable non-patch file is not an edit and falls through
    to the test / explore check.
    """
    # A trajectory step occasionally encodes its command as a list of tokens (or
    # some other non-str); coerce so the regex classifiers never hit a TypeError.
    if isinstance(command, list):
        command = " ".join(str(c) for c in command)
    cmd = command if isinstance(command, str) else ("" if not command else str(command))
    if _is_write_command(cmd):
        target = _extract_write_target(cmd)
        if target is None:
            return "edit"  # keyword-only fallback (Q14)
        if _target_matches_patch(target, patch_files):
            return "edit"
        # parseable target, not in patch -> not an edit; fall through
    if _is_test_command(cmd):
        return "test"
    return "explore"


def _step_is_edit(msg: dict, patch_files: set[str]) -> bool:
    return any(classify_command(c, patch_files) == "edit" for c in commands_of(msg))


def _step_is_test(msg: dict, patch_files: set[str]) -> bool:
    return any(classify_command(c, patch_files) == "test" for c in commands_of(msg))


# --------------------------------------------------------------------------
# Dims 2 & 8 (sec 2): first_attempt_correct & explore_edit_ratio
# --------------------------------------------------------------------------
def count_edits(traj: dict, patch_files: set[str]) -> int:
    """Number of action messages that issued at least one edit command."""
    return sum(_step_is_edit(m, patch_files) for m in action_messages(traj))


def first_edit_step(traj: dict, patch_files: set[str]) -> int | None:
    """1-indexed position of the first edit step among action messages; None if no edit."""
    for i, msg in enumerate(action_messages(traj), start=1):
        if _step_is_edit(msg, patch_files):
            return i
    return None


def explore_edit_ratio(traj: dict, patch_files: set[str]) -> float | None:
    """Front-load ratio ``(first_edit_step - 1) / total_steps`` (sec 8).

    1.0 when the rollout never edits (fully exploratory); None when there are no
    steps at all (the ratio is undefined).
    """
    total = count_steps(traj)
    if total == 0:
        return None
    fes = first_edit_step(traj, patch_files)
    if fes is None:
        return 1.0
    return (fes - 1) / total


def first_attempt_correct(traj: dict, solved: bool, patch_files: set[str]) -> bool:
    """Heuristic for sec 8 dim 2 (W4 redefinition, 2026-05-20).

    ``solved`` AND **no edit step runs after any test step** -- i.e. no
    edit -> test -> edit backtracking. Multi-file (or multi-edit) patches whose
    edits all occur before the first test still count as first-attempt correct.

    Supersedes the v1.0 ``<= 1 edit`` rule (sec 15.4 W4), which mis-labelled
    clean multi-file solutions as failures by conflating *files touched* with
    *attempts*.
    """
    if not solved:
        return False
    seen_test = False
    for msg in action_messages(traj):
        if seen_test and _step_is_edit(msg, patch_files):
            return False
        if _step_is_test(msg, patch_files):
            seen_test = True
    return True


# --------------------------------------------------------------------------
# Reasoning text (input for the dim-5 judge, sec 8)
# --------------------------------------------------------------------------
def concatenated_reasoning(traj: dict) -> str:
    """All assistant message text content, concatenated (sec 8 express_uncertainty input)."""
    parts: list[str] = []
    for msg in get_messages(traj):
        if msg.get("role") in _ASSISTANT_ROLES:
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
            elif isinstance(content, list):  # multimodal content blocks
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        parts.append(block["text"].strip())
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Dim 9: mean_action_block_size (amendment 2026-05-21)
# --------------------------------------------------------------------------
def mean_action_block_size(traj: dict) -> float | None:
    """Mean number of bash commands per assistant action message (sec 2 dim 9).

    The scaffold permits "one or more" commands per response, so the batch size
    is a model-driven command-batching style, not a scaffold-imposed constant.
    Averages ``len(commands_of(m))`` over all action messages. None when there
    are no action messages (the mean is undefined).
    """
    msgs = action_messages(traj)
    if not msgs:
        return None
    counts = [len(commands_of(m)) for m in msgs]
    return float(sum(counts)) / len(counts)


# --------------------------------------------------------------------------
# Dim 10: n_constraint_violations (amendment 2026-05-21)
# --------------------------------------------------------------------------
# The swebench.yaml scaffold states explicit constraints; this dimension counts
# how many *distinct* ones the rollout breaches -- deliberately, by confusion,
# or by ingrained habit. A pure model-compliance signal: env-independent (the
# constraints are always in-context), orthogonal to the content/effort dims,
# and a deviation-from-norm (so the *violation* is never scaffold-prompted).
_RE_INTERACTIVE_EDITOR = re.compile(_CMD_START + r"(?:vi|vim|nano|emacs)\b", re.MULTILINE)
_RE_GIT_COMMIT = re.compile(_CMD_START + r"git\s+commit\b", re.MULTILINE)
_SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
# Test files / config-&-build files the scaffold forbids modifying.
_RE_TEST_PATH = re.compile(r"(?:^|/)(?:tests?/|test_|conftest\.py|.*_test\.py)", re.IGNORECASE)
_RE_CONFIG_PATH = re.compile(
    r"(?:^|/)(?:pyproject\.toml|setup\.py|setup\.cfg|tox\.ini|setup_cfg\.py|MANIFEST\.in)$",
    re.IGNORECASE,
)


def constraint_violations(traj: dict) -> set[str]:
    """The set of distinct scaffold-constraint violation tags in a rollout (sec 2 dim 10).

    Deterministic checklist (swebench.yaml):
      - ``modified_test_file``      -- patch touches a test file / conftest
      - ``modified_config_file``    -- patch touches pyproject/setup/tox/MANIFEST
      - ``used_interactive_editor`` -- a command invokes vi/vim/nano/emacs
      - ``committed_changes``       -- a ``git commit`` was issued ("Do NOT commit")
      - ``combined_patch_and_submit`` -- the submit marker shares a command with
                                       patch creation (must be separate commands)
      - ``continued_after_submit``  -- an action message issues commands *after*
                                       the submit marker ("You CANNOT continue")
    """
    violations: set[str] = set()

    patch_files = extract_patch_files(extract_final_patch(traj))
    if any(_RE_TEST_PATH.search(f) for f in patch_files):
        violations.add("modified_test_file")
    if any(_RE_CONFIG_PATH.search(f) for f in patch_files):
        violations.add("modified_config_file")

    submit_seen = False
    for msg in get_messages(traj):
        if msg.get("role") not in _ASSISTANT_ROLES:
            continue
        cmds = commands_of(msg)
        # Ordering check uses submit_seen from *prior* messages (set at end below).
        if submit_seen and cmds:
            violations.add("continued_after_submit")
        for c in cmds:
            if _RE_INTERACTIVE_EDITOR.search(c):
                violations.add("used_interactive_editor")
            if _RE_GIT_COMMIT.search(c):
                violations.add("committed_changes")
            if _SUBMIT_MARKER in c and ("git diff" in c or "> patch.txt" in c or ">patch.txt" in c):
                violations.add("combined_patch_and_submit")
        if any(_SUBMIT_MARKER in c for c in cmds):
            submit_seen = True
    return violations


def count_constraint_violations(traj: dict) -> int:
    """Number of distinct scaffold-constraint violations (sec 2 dim 10)."""
    return len(constraint_violations(traj))


# --------------------------------------------------------------------------
# Orchestration: the 7 deterministic dimensions
# --------------------------------------------------------------------------
def extract_deterministic(traj: dict, solved: bool | None) -> dict[str, float | None]:
    """Extract the 7 deterministically-measured dimensions of one rollout (sec 8, §15.1 A1).

    The 3 judge-based dimensions (documentation, defensive_code, express_uncertainty)
    are added separately by ``judge.py``.
    """
    patch_files = extract_patch_files(extract_final_patch(traj))
    p_solve = None if solved is None else (1.0 if solved else 0.0)
    if solved is None:
        fac = None
    else:
        fac = 1.0 if first_attempt_correct(traj, solved, patch_files) else 0.0
    return {
        "p_solve": p_solve,
        "p_first_attempt_correct": fac,
        "avg_tokens_step": avg_tokens_per_step(traj),
        "total_expected_steps": float(count_steps(traj)),
        "explore_edit_ratio": explore_edit_ratio(traj, patch_files),
        "mean_action_block_size": mean_action_block_size(traj),
        "n_constraint_violations": float(count_constraint_violations(traj)),
    }
