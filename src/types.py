"""Shared dataclasses for the self-prediction capabilities project.

These are the canonical data structures threaded through every experiment's
pipeline. Each is JSON-round-trippable (``to_dict`` / ``from_dict``) so the
pipeline stages can cache to ``data/`` and resume (EVAL_SPEC_V1.md sec 12.1).

The generic ``Dimension`` descriptor lets ``src/scoring.py`` stay experiment
-agnostic: the concrete 10-dimension battery is built in the experiment's
``config.py`` (sec 2), but scoring only needs each dimension's ``kind``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Allowed enum values -- a wrong value is a loud failure, never a silent default.
_KINDS = frozenset({"binary", "continuous"})
_CONDITIONS = frozenset({"no_thinking", "thinking"})
#: Self-forecast kinds only.
_KINDS_FORECAST = frozenset({"self", "self_primed"})

#: A (model, task) cell is retained for a dimension only if at least this many
#: of its 8 rollouts are valid for that dimension (EVAL_SPEC_V1.md sec 9.6).
MIN_VALID_ROLLOUTS = 5


@dataclass(frozen=True)
class Dimension:
    """A single forecastable dimension of the 10-dimension battery (sec 2 / §15.1 A1).

    ``kind`` selects the per-cell loss (per the §15.7 amendment): ``binary`` →
    Brier loss; ``continuous`` → MAE on the raw scale. No log transform; no
    per-dim normalisation. Per-dim values are reported in native units.
    """

    name: str
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"Dimension.kind must be one of {sorted(_KINDS)}, got {self.kind!r}")

    @property
    def is_binary(self) -> bool:
        return self.kind == "binary"

    @property
    def is_continuous(self) -> bool:
        return self.kind == "continuous"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Dimension":
        return cls(name=d["name"], kind=d["kind"])


@dataclass
class Rollout:
    """One agentic rollout (one of 8 per (model, task) cell, sec 6).

    ``trajectory`` is the raw mini-swe-agent ``mini-swe-agent-1.1`` trajectory
    dict (keys: ``info``, ``messages``, ``trajectory_format``). ``solved`` is
    filled by the SWE-bench harness in pipeline Stage 3. ``errored`` marks a
    rollout that failed to produce a usable trajectory at all.
    """

    model: str
    task_id: str
    rollout_idx: int
    trajectory: dict | None = None
    solved: bool | None = None
    errored: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Rollout":
        return cls(
            model=d["model"],
            task_id=d["task_id"],
            rollout_idx=d["rollout_idx"],
            trajectory=d.get("trajectory"),
            solved=d.get("solved"),
            errored=d.get("errored", False),
            error=d.get("error"),
        )


@dataclass
class Outcome:
    """The extracted 10-dimension ground truth for a single rollout (sec 8).

    ``values`` maps dimension name -> numeric value (binary dims as 0.0/1.0).
    A value of ``None`` means that dimension could not be extracted for this
    rollout (e.g. ``avg_tokens_step`` when usage metadata is absent) -- the
    rollout is then *not valid* for that dimension (sec 9.6).
    """

    model: str
    task_id: str
    rollout_idx: int
    values: dict[str, float | None] = field(default_factory=dict)
    errored: bool = False
    error: str | None = None

    def valid_for(self, dimension: str) -> bool:
        """True iff this rollout contributes a usable ground truth for ``dimension``."""
        if self.errored:
            return False
        return self.values.get(dimension) is not None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Outcome":
        return cls(
            model=d["model"],
            task_id=d["task_id"],
            rollout_idx=d["rollout_idx"],
            values=dict(d.get("values", {})),
            errored=d.get("errored", False),
            error=d.get("error"),
        )


@dataclass
class Forecast:
    """A single forecast sample, one of N per (model, task, condition).

    ``predictions`` maps dimension name -> forecast value. When ``broken`` is
    True (two-stage extraction failed, sec 10.9) it is empty and the sample is
    treated as missing data (sec 9.6); a cell is missing only when *all* N
    samples for that cell are broken.

    ``sample_idx`` identifies the sample within its (model, task, condition)
    cell -- 0-indexed up to ``N_FORECAST_SAMPLES - 1`` (sec 9.6, multi-sample).

    Two ``kind`` values:

    * ``self``        -- the model forecasting itself (sec 4).
    * ``self_primed`` -- the same, after it answered "which model are you?".

    ``identity_reply`` holds the model's answer to the identity question for
    ``kind="self_primed"``.
    """

    model: str
    task_id: str
    condition: str
    kind: str
    predictions: dict[str, float] = field(default_factory=dict)
    broken: bool = False
    raw_response: str = ""
    sample_idx: int = 0
    identity_reply: str | None = None

    def __post_init__(self) -> None:
        if self.condition not in _CONDITIONS:
            raise ValueError(
                f"Forecast.condition must be one of {sorted(_CONDITIONS)}, got {self.condition!r}"
            )
        if self.kind not in _KINDS_FORECAST:
            raise ValueError(
                f"Forecast.kind must be one of {sorted(_KINDS_FORECAST)}, got {self.kind!r}"
            )
        if (self.kind == "self_primed") != (self.identity_reply is not None):
            raise ValueError(
                "Forecast.identity_reply is required for kind='self_primed' and "
                f"forbidden otherwise; got kind={self.kind!r}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Forecast":
        return cls(
            model=d["model"],
            task_id=d["task_id"],
            condition=d["condition"],
            kind=d["kind"],
            predictions=dict(d.get("predictions", {})),
            broken=d.get("broken", False),
            raw_response=d.get("raw_response", ""),
            sample_idx=d.get("sample_idx", 0),
            identity_reply=d.get("identity_reply"),
        )


@dataclass
class Cell:
    """A (model, task) cell: the 8 rollout Outcomes that a forecast is scored against.

    Per sec 3.1 each forecast is scored against the (up to) 8 rollout outcomes
    of its (model, task) cell.
    """

    model: str
    task_id: str
    outcomes: list[Outcome] = field(default_factory=list)

    def valid_values(self, dimension: str) -> list[float]:
        """Ground-truth values from rollouts valid for ``dimension`` (sec 9.6)."""
        return [
            o.values[dimension]  # type: ignore[index]  -- valid_for guarantees not None
            for o in self.outcomes
            if o.valid_for(dimension)
        ]

    def is_retained(self, dimension: str, min_valid: int = MIN_VALID_ROLLOUTS) -> bool:
        """True iff >= ``min_valid`` rollouts are valid for ``dimension`` (sec 9.6)."""
        return len(self.valid_values(dimension)) >= min_valid

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "task_id": self.task_id,
            "outcomes": [o.to_dict() for o in self.outcomes],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Cell":
        return cls(
            model=d["model"],
            task_id=d["task_id"],
            outcomes=[Outcome.from_dict(o) for o in d.get("outcomes", [])],
        )
