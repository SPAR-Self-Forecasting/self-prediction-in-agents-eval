# Self-Prediction in LLM Agents

Code, data, and analysis for **"Open Problems in LLM Self-Prediction"** (Trustworthy
AI for Good workshop, NeurIPS 2026).

We test how well language models can predict what they will do in long-horizon
agentic environments, across two settings: a ten-dimension forecast over their own
SWE-bench Verified rollouts, and a forecast of their own constraint violations on
ODCV-Bench. The short version of the result is that models are overconfident and
their forecasts carry almost no task-level signal — no model beats a constant
forecast of its own base rate on any binary dimension. The apparent advantage of
stronger models is that reality rises to meet a roughly fixed optimism, not that
they know themselves any better.

The suite exists so that self-predictive ability can be tracked in future models: a
sudden step above these baselines would be worth knowing about, since self-knowledge
is plausibly a prerequisite for long-horizon scheming.

## Layout

| Path | What it is |
| --- | --- |
| `src/` | Shared library: LLM client, extraction, scoring, statistics |
| `experiments/2026-05-16_self_prediction/` | Experiment 1 — SWE-bench introspection battery |
| `experiments/2026-07-29_odcv_framing/` | Experiment 2 — ODCV-Bench framing gap |
| `docs/` | Specs and findings notes |
| `run_all_experiments.py` | Root orchestrator |

Each experiment directory has its own `README.md` or `REPORT.md` with the full
methodology, results, and reproduction recipe. Those are the authoritative
documents; this file is the map.

## Quickstart

Requires Python 3.11+.

```bash
git clone https://github.com/SPAR-Self-Forecasting/self-prediction-in-agents-eval
cd self-prediction-in-agents-eval
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Experiment 1 + shared library: 354 tests, all passing, no external data needed
pytest experiments/2026-05-16_self_prediction/tests src tests

# mocked smoke run — seconds, no network, no spend
python run_all_experiments.py --experiments 1 --quick \
  --data-dir /tmp/sp-dry --results-dir /tmp/sp-dry-results
```

The dry run is mocked by default: it swaps in fake models and fake rollouts so the
whole pipeline exercises in seconds with no network, no Docker, and no spend. Use it
to check the plumbing before committing to a real run.

> **Send the dry run somewhere scratch.** Experiment 1 writes to `data/` and
> `results/` by default, and `results/` is where this repo's *published* artifacts
> live — the real run's output, promoted there for release, including the figures used
> in the paper. A default-path dry run will silently overwrite them with mocked
> output. Pass `--data-dir`/`--results-dir` as above. If you forget, restore them with
> `git checkout -- experiments/2026-05-16_self_prediction/results/`.

A bare `pytest` at the repo root will additionally collect Experiment 2's tests, and
13 of those fail with `FileNotFoundError` unless you have the ODCV-Bench checkout in
place — see below. That is the expected state on a fresh clone, not a broken install.

## Reproducing the experiments

**Experiment 1 — SWE-bench introspection battery.** Twelve models each attempt 30
SWE-bench Verified tasks, eight rollouts apiece, and write a self-forecast across ten
dimensions per task. A real run needs Linux + Docker and costs real money (hundreds of
dollars in API spend; the machine itself is cheap). It is resumable — the per-`(model,
task, rollout)` cache means an interrupted run continues rather than re-paying. See
[`experiments/2026-05-16_self_prediction/README.md`](experiments/2026-05-16_self_prediction/README.md).

**Experiment 2 — ODCV-Bench framing gap.** Nine models predict severity-scored
misbehavior in 40 agentic scenarios under two framings that differ only in the
referent: "you are deployed in this environment" versus "another AI agent is." Ground
truth is the models' own previously recorded behavior in those same scenarios. It is
not driven from the root orchestrator — it is a ladder of individually-approved paid
steps, so `run_all_experiments.py --experiments 2` prints the recipe instead of
spending. See the Reproduction section of
[`experiments/2026-07-29_odcv_framing/REPORT.md`](experiments/2026-07-29_odcv_framing/REPORT.md).

This experiment reads scenarios and behavioral ground truth from ODCV-Bench
([Li et al., 2025](https://arxiv.org/abs/2512.20798)), which is **not vendored here**:
it is ~98 MB and belongs to its authors. Place a checkout at `external/ODCV-Bench`
(gitignored, treated as read-only) so that scenario prompts resolve at
`external/ODCV-Bench/mandated_scenarios/<scenario>/prompt.json`. Without it, the
experiment's code and its 13 data-backed tests cannot run; the committed results under
`results/` are still fully inspectable.

API keys go in a repo-root `.env` (`OPENROUTER_API_KEY=...`). It is gitignored and has
never been committed.

## Data availability

You do not need to re-run anything to validate the published numbers.

Committed in this repository:

- `experiments/2026-05-16_self_prediction/data_real/forecasts/` — every self- and
  cross-forecast, per model × task × rollout
- `experiments/2026-05-16_self_prediction/data_real/outcomes/` — the graded outcome of
  every rollout
- `experiments/2026-05-16_self_prediction/data_real/tasks.json` — the exact 30
  SWE-bench Verified instances used
- `experiments/2026-07-29_odcv_framing/data/` and `results/` — raw elicitations,
  per-cell predictions, and the analysis output

Distributed as [GitHub Release](https://github.com/SPAR-Self-Forecasting/self-prediction-in-agents-eval/releases/tag/exp1-rollouts-v1)
assets, too large for git: the full rollout cache of 2,880 agent trajectories
(12 models × 30 tasks × 8 rollouts), as one `rollouts_<model>.tar.gz` per model plus
`SHA256SUMS.txt`. That is 179 MB compressed, roughly 2.5 GB unpacked. Unpack into
place with:

```bash
cd experiments/2026-05-16_self_prediction
tar -xzf rollouts_<model>.tar.gz   # recreates data_real/rollouts/<model>/...
```

### A note on model counts

Experiment 1 was *run* with twelve models, so you will find twelve directories under
`data_real/outcomes/`. The paper analyses and reports **eleven**: `z-ai/glm-5` is
present in the raw data but excluded from the analysis, and the eleven listed in the
paper's appendix are the ones in `results/significance.json`. If you recompute from
the raw data and get twelve, that is why.

## Citation

```bibtex
@inproceedings{nottingham2026selfprediction,
  title     = {Open Problems in {LLM} Self-Prediction},
  author    = {Nottingham, Lydia and Occhipinti, Giovanni Maria and Tran, Andrew},
  booktitle = {NeurIPS 2026 Workshop on Trustworthy AI for Good (AI4GOOD)},
  year      = {2026}
}
```

This work was completed as part of the Supervised Program for Alignment Research
(SPAR). We thank David Atkinson for the preliminary SWE-bench investigation.

## License

MIT — see [LICENSE](LICENSE).
