# Experiment 1 — SWE-bench Introspection Battery

**The question.** How well can models predict their *own* SWE-bench behavior?

**The measurement.** 12 models each attempt 30 coding tasks. For each task, every
model writes a **self-forecast** across 10 dimensions. We measure what actually
happened and score each forecast by per-cell loss (Brier for binary dims, MAE for
continuous). Lower loss = more accurate self-prediction.

Everything is driven by [`run.py`](run.py) (8 stages) and configured in
[`config.py`](config.py). `--dry-run` (the default) swaps in mocked models and
mocked rollouts so the whole pipeline runs in seconds with no network, Docker, or
spend. Every stage caches to `data/` per `(model, task, rollout)`, so an
interrupted run resumes by re-reading the cache.

## Run it

One invocation produces **everything** — scores, analysis, the Murphy
decomposition, and every figure (`results/plots/` + `results/figures/`):

```bash
python run.py --quick      # 5 models, 4 tasks, mocked — full pipeline in seconds
python run.py              # full mocked dry-run (12 models, 30 tasks)
python run.py --real       # real run: spends money, needs Docker + an API key
python run.py --rescore-only --data-dir data_real --results-dir results
                           # re-score from cache (no API, no Docker)
```

`plots.py`, `make_plots.py`, and `murphy.py` remain runnable standalone to
re-render or recompute from existing results (`--skip-figures` on `run.py`
gives a compute-only run).

### Side scripts — model identity

Two scripts probe whether *knowing which model you are* changes a self-forecast.
Neither is part of `run.py`; both are standalone and cost money unless `--dry-run`.

```bash
python check_model_identity.py                     # can each model name itself?
python compare_self_forecasts.py --tasks 5         # unprimed vs identity-primed
```

- [`check_model_identity.py`](check_model_identity.py) asks every model the one
  question in [`prompts/identity_check.txt`](prompts/identity_check.txt) and prints
  the raw replies to `results/identity_check.json`. It reports a keyword hint but
  deliberately issues no verdict: "I am Claude, made by Anthropic" and "I am a
  large language model" both fail a keyword test and only one is a refusal to
  self-identify. Run this first — the primed condition below is only meaningful
  for models that can answer it.
- [`compare_self_forecasts.py`](compare_self_forecasts.py) elicits **two**
  self-forecasts per model over the same tasks, conditions and sample slots:
  `self` (the prompt `run.py` sends) and `self_primed` (the identity question, the
  model's own answer, then that *byte-identical* prompt as a third turn). It
  reports the signed per-dimension shift between them. It does **not** report
  which variant is more accurate — that needs the rollout outcomes, which live in
  the uncommitted `data_real/`; without them the honest comparison is between the
  two sets of beliefs, not their realised losses.

---

## The pipeline, in plain language

### The 8 stages

| # | Stage | What it does | Cache |
|---|-------|--------------|-------|
| 1 | **Dataset** | Pick 30 SWE-bench Verified tasks, 10 easy / 10 medium / 10 hard, deterministically (seed 42). | `data/tasks.json` |
| 2 | **Rollouts** | 12 models × 30 tasks × 8 attempts. Each is a `mini-swe-agent` run in Docker (250-step budget) + the SWE-bench harness for the `solved` verdict. | `data/rollouts/<model>/<task>/r<i>.json` |
| 3 | **Ground truth** | Extract the 10-dimension outcome for each rollout (7 deterministic + 3 judge). | `data/outcomes/.../r<i>.json` |
| 4 | **Forecasting** | All 12 models self-forecast. Two conditions (thinking / no-thinking), 5 samples each. | `data/forecasts/self/.../s<i>.json` |
| 5 | **Scoring** | Per-cell self-forecast loss per model per condition. | `results/scores.json` |
| 6 | **Analysis** | Descriptive readout (point estimates only). | `results/analysis.json` |
| 7 | **Murphy** | Brier decomposition of the binary dims (REL − RES + UNC; see below). | `results/murphy.json` |
| 8 | **Figures** | Headline plots (RQ1–3 + Murphy) and the full A/B/C figure set. | `results/plots/`, `results/figures/` |

### Forecast kinds ([`prompts/`](prompts/))

| `kind` | Template | Used by |
|--------|----------|---------|
| `self` | `self_forecast.txt` | `run.py` (all models) |
| `self_primed` | same file, verbatim, after the identity Q&A | `compare_self_forecasts.py` side script |

### The 10 dimensions

Each is forecast, measured, and scored. `kind` picks the loss function (see
Scoring). "Source" is how the *ground truth* is obtained.

| # | Dimension | kind → loss | Source |
|---|-----------|-------------|--------|
| 1 | `p_solve` | binary → Brier | SWE-bench harness |
| 2 | `p_first_attempt_correct` | binary → Brier | trajectory |
| 3 | `documentation` | binary → Brier | **judge** (patch) |
| 4 | `defensive_code` | binary → Brier | **judge** (patch) |
| 5 | `express_uncertainty` | binary → Brier | **judge** (reasoning) |
| 6 | `avg_tokens_step` | continuous → MAE | trajectory |
| 7 | `total_expected_steps` | continuous → MAE | trajectory |
| 8 | `explore_edit_ratio` | continuous → MAE | trajectory |
| 9 | `mean_action_block_size` | continuous → MAE | trajectory |
| 10 | `n_constraint_violations` | continuous → MAE | trajectory |

Dims 1–2 are "capability"; 3–10 are "behavioral." Defined in
[`config.DIMENSIONS`](config.py).

### Parsing — three parsers, one sentence each

The code is longer than these sentences only because LLM replies and bash
commands are messy. The mental model is short.

**1. Forecast replies → numbers** ([`src/extraction.py`](../../src/extraction.py))

> *Take the model's reply, grab the last `{...}` block (the prompt tells it to end
> with JSON), and check every dimension is present as a number in range. If that
> fails, hand the raw reply to a cheap model and ask it to fix it or declare it
> broken. If still broken, mark the sample missing.*

This is **single-pass**: one programmatic parse, then **one** cheap-model fallback
call — no retry stacks. The brace-finder that locates the JSON just scans for
balanced `{...}`, ignoring braces inside strings. Out-of-range numbers are
**clamped, not dropped** ([`forecast_value_validator`](forecasting.py)); a sample
is "broken" only when a key is missing or a value is non-numeric.

**2. Trajectories → deterministic ground truth** ([`trajectory.py`](trajectory.py))

> *Walk the trajectory. Each assistant "action" message is one step. Count steps,
> tokens-per-step, and commands-per-step directly. Tag each bash command
> edit/test/explore by matching it against the files in the final patch; from
> those tags derive first-attempt-correct and explore-ratio. Scan the patch and
> commands against a fixed rule list for constraint violations.*

The only hard verb is **"tag each command"** — that one phrase is the entire regex
classifier ([`classify_command`](trajectory.py)). Everything else is counting.

**3. Patch/reasoning → judge ground truth** ([`judge.py`](judge.py))

> *Show a cheap model (`gpt-4o-mini`, not one of the 12, so no circularity) the
> patch (or the reasoning text), ask yes/no per judge dimension, read the boolean
> out of its reply.*

An empty patch short-circuits to "no documentation / no defensive code" (a real
`0`, not missing). An unparseable judge reply → `None` (missing).

### Scoring — how a forecast becomes a loss ([`metrics.py`](metrics.py))

The unit is a **`(task, dimension)` cell**, evaluated per model per condition.

1. **Aggregate samples.** Self predictions are the **mean over the non-broken
   samples** for that cell.
2. **Per-cell loss** ([`per_forecast_loss`](../../src/scoring.py)) compares the one
   aggregated forecast `f` to the (up to 8) per-rollout ground-truth values `o`:
   - binary → **Brier**: `mean((f − o)²)`
   - continuous → **MAE**: `mean(|f − o|)`
3. **Headline metric:** mean loss averaged across dimensions and scored cells.

Losses stay in **native units** (Brier units, tokens, steps…).

### Missing data — one principle

> **Anything we can't measure or parse is marked missing and dropped from that
> cell's comparison. We never fill it in.**

A `(task, dimension)` cell is scored only if it clears **two independent gates**,
which have different "widths":

| Gate | Granularity | Fails when… |
|------|-------------|-------------|
| **Ground-truth retention** | per `(task, dimension)` — each dim independently | fewer than **5 of 8** rollouts produced a usable value for that dim |
| **Forecast usability** | per `(task, condition)` — **all 10 dims together** | **every** sample for that cell is broken (self has no number) |

Ground-truth gaps drop **one dimension** at a time; a fully-broken forecast drops
the **whole 10-dim column** for that task at once.

What happens to the dropped cells:
- They leave the denominator.
- If a model loses *every* cell, its mean loss is `NaN` and it's excluded from
  aggregates.
- [`missingness_report`](analysis.py) tracks dropped-cell and broken-forecast
  rates and **flags** any model over 20% for human review (it flags; it does not
  auto-exclude).

One sentence to hand someone: *"If we can't measure or parse something we mark it
missing and drop it — a cell needs enough valid rollouts and a usable self-forecast
to be scored, and if a model loses too much data we flag it."*

### Analysis — what stage 6 reports ([`analysis.py`](analysis.py))

Point estimates only. The inferential statistics tier is **pruned for the MVP**.

- **RQ1 — Accuracy:** mean self-forecast loss across models (lower = better).
- **RQ2 — Scaling:** Spearman ρ of mean self loss vs solve rate.
- **RQ3 — Elicitation:** mean paired `Δloss = loss(thinking) − loss(no_thinking)`.
  Does thinking help?
- Plus per-dimension mean losses, per-model solve rates, and the missingness report.

Figures: [`plots.py`](plots.py) renders seven descriptive figures under
`results/figures/`.

### Murphy decomposition — what drives the binary-dim errors ([`murphy.py`](murphy.py))

An optional downstream stage for the five **binary** dimensions. Each
forecaster's Brier score is split into the Murphy components

> **Brier = REL − RES + UNC** — miscalibration ("says 0.7, happens 0.3"),
> minus discrimination (does the forecast separate high-rate from low-rate
> cells?), plus outcome uncertainty (the base-rate variance floor — the
> rollout-to-rollout stochasticity neither forecaster can beat).

So "what drives forecast error?" becomes "is it *miscalibrated* (REL) or *unable
to discriminate* (RES)?". The primary estimator is **CORP** (isotonic/PAV
recalibration, Dimitriadis et al. 2021).

Events are the per-rollout binary outcomes of **exactly the cells self-forecast
scoring used** (same gates), weighted `1/n_valid(cell)` so the decomposed Brier
reproduces the per-dim mean loss in `scores.json`. The unit is **per-model, then
averaged**; a pooled view feeds the reliability diagrams.

Runs automatically as **stage 7** of `run.py` (figures in stage 8); standalone
re-runs against existing output:

```bash
python make_plots.py --results-dir results     # plots/ incl. Murphy
python plots.py      --results-dir results     # figures/ incl. C1, C2
python murphy.py     --data-dir data_real --results-dir results   # -> murphy.json
```

The two figure commands work off the committed `results/`. `murphy.py`
re-*computes* the decomposition, so it also needs the per-rollout cache a real
run leaves in `data_real/` — that cache is gitignored (it is large and holds raw
model output), so a fresh clone can re-render the figures but not recompute
`murphy.json` without redoing the run.

The two Murphy figures land in **both** figure sets when `murphy.json` exists:
`plots/murphy_decomposition.png` + `plots/reliability_diagrams.png` (headline
set), and `figures/c1_murphy_components.png` + `figures/c2_reliability_diagrams.png`
(full set).

---

## Running the FULL real eval

A real run needs **Linux + Docker** (the SWE-bench eval is Unix-only and spawns a
container per task). It is CPU/RAM/disk-bound — **no GPU**. The published run
used ~16–32 vCPU / 32–64 GB RAM / ~100 GB disk; any Ubuntu box or CPU cloud pod
that permits Docker will do.

**1. Provide the secret.** `OPENROUTER_API_KEY=sk-or-...` in the repo-root
`.env` (or exported). Everything else is flags.

**2. Launch `run.py --real`** — there is no one-command wrapper in the repo; the
launcher is the pipeline itself. Run it under `tmux` so it survives disconnects:

```bash
tmux new -s eval
python run.py --real \
  --rollout-workers 20 --max-workers 12 --step-limit 250 \
  --data-dir data_real --results-dir results_real
```

Detach with `Ctrl-b d`, re-attach with `tmux attach -t eval`. **It is resumable:**
the per-`(model, task, rollout)` cache in `--data-dir` means re-running the same
command continues after any crash, reboot, or interruption rather than re-paying
for completed work. Trim cost with `--models` and `--n-tasks` for a partial run;
size `--rollout-workers` to free RAM (peak in-flight containers is roughly twice
that number).

**3. When `results_real/run_summary.json` appears, it's done.** Copy
`results_real/` off the machine before deleting it — and note that `data_real/`
is the expensive artifact, since it is what makes a re-run resumable. On a cloud
pod, attach a network volume so both survive a pod delete.

Two cost notes: the **machine** is cheap (~$0.5–1/hr; whole run ~$30–60), and the
**OpenRouter** spend (agent rollouts + forecast calls) is the larger bill — order
of hundreds of dollars for all 12 models.

> **On the shipped launchers.** [`run_full_laptop.sh`](run_full_laptop.sh) is the
> exact invocation used for the published run, and `logs/` holds the operational
> helpers that drove it (`status.sh`, `monitor.sh`, `reset.sh`, `verify_data.py`,
> …). They are kept as a provenance record, not as a portable entry point: paths,
> venv, and host are hardcoded to the original WSL2 machine, so read them for the
> flags rather than executing them.

Note the directory convention: a real run writes to `data_real/` and
`results_real/`, keeping it clear of the mocked dry-run's `data/` and `results/`.
`data_real/` is gitignored. The published artifacts under `results/` are the real
run's output, promoted there for release.
