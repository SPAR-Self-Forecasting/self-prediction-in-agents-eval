"""Tests for dataset.py -- SWE-bench Verified task selection (sec 6)."""

import pytest

import config
import dataset


def _fake_rows(n_easy, n_medium, n_hard):
    """Synthetic SWE-bench-Verified-shaped rows with a human difficulty field."""
    diff_label = {"easy": "<15 min fix", "medium": "15 min - 1 hour", "hard": "1-4 hours"}
    rows = []
    for bucket, n in (("easy", n_easy), ("medium", n_medium), ("hard", n_hard)):
        for i in range(n):
            rows.append({
                "instance_id": f"repo__proj-{bucket}-{i:03d}",
                "repo": "repo/proj",
                "base_commit": "0" * 40,
                "problem_statement": f"A {bucket} issue number {i}.",
                "difficulty": diff_label[bucket],
            })
    return rows


# --------------------------------------------------------------------------
# select_tasks
# --------------------------------------------------------------------------
def test_select_30_tasks_10_each():
    tasks = dataset.select_tasks(seed=42, loader=lambda: _fake_rows(20, 18, 25))
    assert len(tasks) == config.N_TASKS
    counts = {b: sum(t.difficulty == b for t in tasks) for b in ("easy", "medium", "hard")}
    assert counts == {"easy": 10, "medium": 10, "hard": 10}


def test_selection_is_deterministic_under_seed():
    rows = _fake_rows(20, 20, 20)
    a = [t.instance_id for t in dataset.select_tasks(seed=42, loader=lambda: rows)]
    b = [t.instance_id for t in dataset.select_tasks(seed=42, loader=lambda: rows)]
    assert a == b


def test_different_seed_changes_selection():
    rows = _fake_rows(40, 40, 40)
    a = {t.instance_id for t in dataset.select_tasks(seed=1, loader=lambda: rows)}
    b = {t.instance_id for t in dataset.select_tasks(seed=2, loader=lambda: rows)}
    assert a != b


def test_exact_pool_selects_everything():
    # A bucket with exactly 10 -> all 10 chosen regardless of seed.
    rows = _fake_rows(10, 10, 10)
    ids = {t.instance_id for t in dataset.select_tasks(seed=99, loader=lambda: rows)}
    assert len(ids) == 30
    assert ids == {r["instance_id"] for r in rows}


def test_selected_tasks_are_a_subset_of_the_pool():
    rows = _fake_rows(15, 15, 15)
    pool = {r["instance_id"] for r in rows}
    assert all(t.instance_id in pool for t in dataset.select_tasks(seed=42, loader=lambda: rows))


def test_raises_when_bucket_too_small():
    with pytest.raises(ValueError):
        dataset.select_tasks(seed=42, loader=lambda: _fake_rows(10, 9, 10))


def test_raises_on_unknown_difficulty_label():
    bad = _fake_rows(10, 10, 10)
    bad[0]["difficulty"] = "unknowable"
    with pytest.raises(ValueError):
        dataset.select_tasks(seed=42, loader=lambda: bad)


# --------------------------------------------------------------------------
# mock_tasks (dry-run / tests)
# --------------------------------------------------------------------------
def test_mock_tasks_shape():
    tasks = dataset.mock_tasks()
    assert len(tasks) == 30
    counts = {b: sum(t.difficulty == b for t in tasks) for b in ("easy", "medium", "hard")}
    assert counts == {"easy": 10, "medium": 10, "hard": 10}
    assert len({t.instance_id for t in tasks}) == 30


def test_mock_tasks_deterministic():
    a = [t.instance_id for t in dataset.mock_tasks()]
    b = [t.instance_id for t in dataset.mock_tasks()]
    assert a == b


# --------------------------------------------------------------------------
# Task
# --------------------------------------------------------------------------
def test_task_roundtrip():
    t = dataset.mock_tasks()[0]
    assert dataset.Task.from_dict(t.to_dict()) == t


def test_task_forecast_description_includes_problem_statement():
    t = dataset.Task(
        instance_id="x", difficulty="easy", repo="a/b",
        base_commit="c", problem_statement="The bug is here.",
    )
    desc = t.forecast_description()
    assert "The bug is here." in desc
    assert "a/b" in desc
