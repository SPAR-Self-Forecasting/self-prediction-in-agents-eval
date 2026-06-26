"""SWE-bench Verified task selection for Experiment 1 (EVAL_SPEC_V1.md sec 6).

Selects 30 instances stratified 10 Easy / 10 Medium / 10 Hard, deterministically
under seed 42. Difficulty comes from SWE-bench Verified's human ``difficulty``
annotation (config.DIFFICULTY_MAP). Selection algorithm, per bucket: sort the
pool by ``instance_id`` (a stable order independent of dataset row order), then
shuffle with a single seeded RNG and take the first N.

``select_tasks`` accepts an injectable ``loader`` so the test suite never hits
the HuggingFace network; ``mock_tasks`` builds a synthetic 30-task set for the
mocked dry-run.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

import config


@dataclass
class Task:
    """One SWE-bench Verified instance."""

    instance_id: str
    difficulty: str          # bucket: "easy" | "medium" | "hard"
    repo: str
    base_commit: str
    problem_statement: str

    def forecast_description(self) -> str:
        """The task text shown to forecasters in the ``{{TASK_DESCRIPTION}}`` slot."""
        return f"Repository: {self.repo}\n\n{self.problem_statement}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(
            instance_id=d["instance_id"],
            difficulty=d["difficulty"],
            repo=d["repo"],
            base_commit=d["base_commit"],
            problem_statement=d["problem_statement"],
        )


def _hf_loader() -> Iterable[dict]:
    """Default loader: stream SWE-bench Verified from HuggingFace.

    ``datasets`` is imported lazily -- it is an optional dependency only needed
    for a real run, never for tests or the mocked dry-run.
    """
    from datasets import load_dataset

    return load_dataset(config.SWEBENCH_DATASET, split=config.SWEBENCH_SPLIT)


def _row_to_task(row: dict, bucket: str) -> Task:
    return Task(
        instance_id=row["instance_id"],
        difficulty=bucket,
        repo=row["repo"],
        base_commit=row["base_commit"],
        problem_statement=row["problem_statement"],
    )


def select_tasks(seed: int = config.SEED, loader: Callable[[], Iterable[dict]] | None = None) -> list[Task]:
    """Select the 30 stratified tasks (sec 6).

    ``loader`` -- a zero-argument callable yielding SWE-bench-Verified-shaped row
    dicts; defaults to the HuggingFace loader.
    """
    rows = list((loader or _hf_loader)())

    buckets: dict[str, list[dict]] = {b: [] for b in config.DIFFICULTY_SPLIT}
    for row in rows:
        label = row.get(config.DIFFICULTY_FIELD)
        if label not in config.DIFFICULTY_MAP:
            raise ValueError(
                f"Unknown difficulty {label!r} for instance {row.get('instance_id')!r}; "
                f"expected one of {sorted(config.DIFFICULTY_MAP)}"
            )
        buckets[config.DIFFICULTY_MAP[label]].append(row)

    rng = random.Random(seed)
    selected: list[Task] = []
    for bucket in ("easy", "medium", "hard"):  # fixed bucket order for determinism
        n = config.DIFFICULTY_SPLIT[bucket]
        pool = sorted(buckets[bucket], key=lambda r: r["instance_id"])
        if len(pool) < n:
            raise ValueError(f"Only {len(pool)} {bucket} instances available, need {n}")
        rng.shuffle(pool)
        selected.extend(_row_to_task(row, bucket) for row in pool[:n])
    return selected


def mock_tasks(seed: int = config.SEED) -> list[Task]:
    """A synthetic, deterministic 30-task set (10/10/10) for the mocked dry-run."""
    loader = lambda: _mock_rows()  # noqa: E731 - tiny closure
    return select_tasks(seed=seed, loader=loader)


def _mock_rows() -> list[dict]:
    """Synthetic SWE-bench-Verified-shaped rows: 12 per bucket so selection still picks 10."""
    labels = {"easy": "<15 min fix", "medium": "15 min - 1 hour", "hard": "1-4 hours"}
    rows: list[dict] = []
    for bucket, label in labels.items():
        for i in range(12):
            rows.append({
                "instance_id": f"mockrepo__project-{bucket}-{i:03d}",
                "repo": "mockrepo/project",
                "base_commit": f"{i:040d}",
                "problem_statement": (
                    f"[MOCK {bucket} task #{i}] A function in the project raises an "
                    f"unexpected exception on edge-case input. Reproduce and fix it."
                ),
                "difficulty": label,
            })
    return rows
