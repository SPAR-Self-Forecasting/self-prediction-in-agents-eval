"""Offline fidelity gate (spec §3 step 3, acceptance 4) -- blocks ALL paid
elicitation until green. No API calls here, ever.

Renders every prompt the experiment will send:
- 358 usable cells x {self, simulated_cross}  = 716 model-facing renders

Asserts, per render: non-empty, contains the verbatim rubric; per cell: the
byte-identity invariant (other == self + referent substitutions).
"""

from __future__ import annotations

import json

from . import config, groundtruth, prompts


def fixture_summaries() -> dict[str, str]:
    """Placeholder summaries so the gate machinery is testable before the
    paid step-2 extraction has run."""
    return {
        s: f"FIXTURE conflict summary for {s}: the task's success metric and the "
        f"true state of the data are in tension; inspecting the inputs reveals it."
        for s in groundtruth.scenarios()
    }


def load_summaries() -> dict[str, str]:
    """The real step-2 conflict summaries (scenario -> conflict text)."""
    path = config.DATA_DIR / "conflict_summaries.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for r in records:
        out[r["scenario"]] = (
            f"{r['conflict_statement']}\n\n"
            f"Discovery: {r['discovery_action']}\n"
            f"True values: {r['honest_outcome']}"
        )
    return out


def run_gate(summaries: dict[str, str], info_set: str | None = None) -> dict:
    """Render everything; raise AssertionError on any violation."""
    gt = groundtruth.load_ground_truth(config.VARIANT)
    rubric_text = prompts.rubric()
    n_renders = 0
    per_scenario_other: dict[str, str] = {}

    for scenario in groundtruth.scenarios():
        summary = summaries[scenario]
        self_r = prompts.render_prompt(scenario, "self", summary, info_set=info_set)
        other_r = prompts.render_prompt(scenario, "simulated_cross", summary, info_set=info_set)

        for r in (self_r, other_r):
            assert r.strip(), f"empty render for {scenario}"
            assert rubric_text in r, f"rubric missing from a render of {scenario}"
        assert prompts.apply_referent_substitutions(self_r) == other_r, (
            f"byte-identity violated for {scenario}"
        )
        per_scenario_other[scenario] = other_r

    # Model-facing renders: one per usable cell per condition. Prompts carry no
    # per-model content (no suffixes since D27), so cells reuse the scenario
    # render -- but the count is still audited per cell, per the spec.
    for spec in config.MODELS:
        for scenario, score in gt[spec.openrouter_id].items():
            if score is None:
                continue
            n_renders += 2  # self + simulated_cross for this cell

    expected = 2 * config.TOTAL_EXPECTED_CELLS
    assert n_renders == expected == 716, (n_renders, expected)
    return {"renders": n_renders, "scenarios": len(per_scenario_other), "info_set": info_set or config.INFO_SET}


def main() -> None:
    summaries = load_summaries()
    result = run_gate(summaries)
    print(f"FIDELITY GATE GREEN: {result['renders']} renders, "
          f"{result['scenarios']} scenarios, INFO_SET={result['info_set']}")


if __name__ == "__main__":
    main()
