# Self-prediction in AI agents

Can a language model predict its own behavior — and should anyone believe it when
it tries?

Two experiments here attack that question from opposite ends. The first measures
how accurately models **forecast their own coding behavior** on SWE-bench. The
second asks what happens to those self-reports when the subject is **misbehavior**
rather than coding ability.

Both are complete, with committed results, figures, and full writeups.

| | Experiment 1 | Experiment 2 |
|---|---|---|
| **Question** | Do models know their own coding behavior? | Do models report their own propensity to violate constraints? |
| **Setting** | 30 SWE-bench Verified tasks | 40 ODCV-Bench constraint-conflict scenarios |
| **Subjects** | 12 open-weight models, 2,880 Docker rollouts | 9 models, 358 previously-recorded behavioral cells |
| **Headline** | Self-forecasts miscalibrated on `p_solve` (Brier ≈ 0.49); accuracy tracks capability (ρ ≈ −0.81) | All 9 models rate themselves safer than an identical anonymous agent — 357 of 358 cells |
| **Spend** | ~$5.57 pipeline LLM cost | ~$28 total |
| **Report** | [`REPORT.md`](experiments/2026-05-16_self_prediction/REPORT.md) | [`REPORT.md`](experiments/2026-07-29_odcv_framing/REPORT.md) |

---

## The shared idea

Both experiments elicit **self-reports** under controlled prompts and score them
against recorded behavior. Experiment 2 additionally contrasts self-framing with
an anonymous peer framing; Experiment 1 scores self-forecasts directly against
rollout ground truth.

## Experiment 1 — the introspection battery

Twelve models each attempted 30 SWE-bench Verified tasks, 8 rollouts apiece, via
`mini-swe-agent` in Docker. Eleven models forecast their own behavior across 10
dimensions (solve probability, documentation, defensive coding, hedging, step
counts, and so on); scoring is per-cell loss (Brier for binary dims, MAE for
continuous).

Headline findings from the real run:

**Severe overconfidence on task success.** Mean Brier on `p_solve` ≈ 0.49 — worse
than always guessing 0.5 (0.25).

**Better models self-predict more accurately.** Spearman ρ ≈ −0.81 between mean
self loss and empirical solve rate (descriptive).

**Error is miscalibration, not blindness.** Murphy decomposition: reliability
(REL ≈ 0.37) dwarfs resolution (RES ≈ 0.04) on `p_solve`.

### Sidebar — calibration vs. discrimination

Two different things can go wrong when a model states a probability, and telling
them apart decides whether the problem is fixable.

**Calibration** asks whether the numbers mean what they say. Of all the times a
model claims 70% confidence, does the thing happen about 70% of the time? A model
that says 0.9 whenever the truth is 0.6 is miscalibrated.

**Discrimination** asks whether the forecasts *separate* cases. Do the tasks the
model rates highly actually succeed more often than the ones it rates low? This is
about ordering, not about whether the numbers are literally right.

The two are independent, and the asymmetry between them is the whole point. A
model that ignores the task and always states the base rate is perfectly
calibrated and completely useless. A model whose numbers are all wrong but whose
*ranking* is perfect is genuinely informative. And crucially, miscalibration is
cheap to repair after the fact — a little held-out data and an isotonic or Platt
fit will squash the numbers into place while preserving the ordering. Missing
discrimination cannot be repaired at all, because there is no signal to rescale.

**The Murphy decomposition** splits a Brier score into exactly those pieces:

> **BS = REL − RES + UNC**

- **REL** (reliability) is the miscalibration penalty — 0 means perfectly calibrated.
- **RES** (resolution) is the discrimination credit, subtracted because it helps.
  Its ceiling is UNC.
- **UNC** (uncertainty) is the base-rate variance `p(1−p)`: the irreducible noise
  floor no forecaster can beat. Here it is the rollout-to-rollout stochasticity of
  the agent itself — the same model on the same task does not always do the same
  thing.

Worked through with the study's own numbers, self-forecasts on `p_solve`:
`0.367 − 0.035 + 0.160 = 0.492`. The floor is 0.160, so nearly all the excess is
the reliability term. Resolution of 0.035 against a 0.160 ceiling means the
forecasts barely order tasks at all.

That is what makes "what drives forecast error?" an answerable question rather
than a shrug: it becomes *is the model miscalibrated, or blind to task-to-task
variation?* Experiment 1 estimates the split with CORP (isotonic/PAV
recalibration; Dimitriadis et al. 2021), which is bin-free and makes the identity
exact with non-negative components.

**The error is miscalibration, not blindness.** Reliability dwarfs resolution on
every binary dimension — models are so overconfident on `p_solve` (mean Brier
≈ 0.49) that they lose to a constant 0.5 guess. Recalibration would recover much
of that error, but discrimination remains low: mean resolution never exceeds ~0.06
on `p_solve`. Discrimination, not calibration, is the binding research problem.

## Experiment 2 — the framing gap

Nine models predicted misbehavior severity in 40 scenarios where a mandated KPI
conflicts with an ethical constraint. Ground truth is each model's *own* recorded
conduct in those exact scenarios. Two prompts, differing only in the referent:
"you are deployed in this environment" versus "another AI agent is deployed in
this environment" — byte-identical otherwise, enforced by a unit test.

Every model rated itself safer than the anonymous agent. Across 358 paired cells,
357 moved self-lower and exactly one moved the other way.

The tempting reading is that models know their propensities and conceal them. The
data rules that out: **neither framing is accurate.** Both lose to a constant
baseline, and both sit within a few points of a coin flip on balanced accuracy.
The error profiles are mirror images — self-framing catches 4% of real violations,
other-framing catches 66% by flagging nearly everything. Qualitative coding of the
large-gap cells found the mechanism: the two framings describe *different
behaviors*. The self-framed answer narrates the honest path; the other-framed
answer names the specific cheat. The gloss is an idealized self-model rather than
concealment — the first-person frame recruits a normative self-description instead
of a predictive one.

The calibration/discrimination split above diagnoses this one too. Reading the
median predicted severity as a ranking score, the probability that an actually
violating cell outranks a clean one is 0.51–0.56 across all three conditions —
chance. The large framing-dependent level bias is real miscalibration, but there
is no ordering signal underneath it to rescale. The models are confidently
uninformative, with a framing-dependent constant on top.

Practical upshot: self-reports of alignment propensity are close to worthless as
evidence, and worse than third-person elicitation.

## Repository layout

```
src/                              shared library, imported unchanged by both experiments
  llm_client.py                     OpenRouter chat client
  extraction.py                     two-stage forecast-JSON extractor (parse, then cheap-model fallback)
  scoring.py  stats.py  types.py    loss functions, inference toolbox, canonical dataclasses

experiments/2026-05-16_self_prediction/     Experiment 1 — 9-stage cached pipeline (run.py)
experiments/2026-07-29_odcv_framing/        Experiment 2 — the odcv package, one module per step

docs/                             Experiment 2's spec, exploration findings, and build handoff
run_all_experiments.py            root orchestrator
```

Each experiment directory holds its own `REPORT.md`, `results/`, `figures/` (or
`results/plots/`), `prompts/`, and `tests/`. Experiment 1 additionally carries a
200-line [`README.md`](experiments/2026-05-16_self_prediction/README.md) that walks
through its pipeline in plain language — the friendliest entry point in the repo.

## Getting started

```bash
pip install -e ".[dev]"      # Python 3.11+
pytest                       # see the note on Experiment 2's tests below
```

Experiment 1 runs fully mocked by default — no network, no Docker, no spend:

```bash
python run_all_experiments.py --experiments 1 --quick   # 5 models, 4 tasks, seconds
python run_all_experiments.py --experiments 1           # full mocked dry-run
```

A real run needs Linux with Docker (the SWE-bench harness is Unix-only and spawns
a container per task) plus `OPENROUTER_API_KEY` in `.env`, and costs on the order
of hundreds of dollars in agent rollouts.

Experiment 2 is not orchestrated from the root: it is a ladder of individually
approved paid steps against a read-only benchmark checkout. `python
run_all_experiments.py --experiments 2` prints its reproduction recipe without
launching anything. Every paid step checkpoints per item and resumes from disk at
zero re-spend.

## Testing

The root and Experiment 1 suites run standalone: **338 passing, 1 skipped.**

Experiment 2 has 52 tests, of which 35 require a read-only clone of
[ODCV-Bench](https://arxiv.org/abs/2512.20798) at `external/ODCV-Bench` — it is
the source of the behavioral ground truth, and it is gitignored. Without that
checkout those 35 fail on missing fixtures while the other 17 pass. That is
expected, not a broken build.

## Conventions

A few house rules that explain why the code looks the way it does:

- **Missing data is dropped, never imputed.** Anything unmeasurable or unparseable
  is marked missing and leaves the denominator. Models exceeding a 20% loss rate
  are flagged for human review, never auto-excluded.
- **Everything caches and resumes.** Both pipelines checkpoint per item, so an
  interrupted run costs nothing to restart. Experiment 2's production run survived
  a mid-flight laptop sleep and a DNS outage with zero loss.
- **Results are descriptive.** Experiment 1's inference tier (bootstrap CIs,
  significance tests) is pruned for the MVP; read its numbers as point estimates.
  Experiment 2 does carry clustered bootstraps and BH correction.
- **Reports carry their own limitations.** Both `REPORT.md` files devote a
  numbered section to what would undermine them, and known confounds are named
  inline rather than buried — Experiment 1's variance confound (W2) is flagged at
  every readout it touches.

## License and credits

MIT — see [`LICENSE`](LICENSE).

Experiment 2 builds on **ODCV-Bench** (Outcome-Driven Constraint Violation,
McGill-DMaS, [arXiv:2512.20798](https://arxiv.org/abs/2512.20798), MIT licensed),
reusing its scenarios and shipped judge scores as behavioral ground truth.
The Murphy decomposition uses the CORP estimator of Dimitriadis et al. (2021,
PNAS).
