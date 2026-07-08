# Experiment 1 — SWE-bench Introspection Battery

**The question.** Can a model predict its *own* SWE-bench behavior better than a
strong peer model can predict it? If yes, that's an "introspective advantage."

**The measurement.** 12 models each attempt 30 coding tasks. For each task, every
model writes a **self-forecast** (predicting its own behavior across 10
dimensions), and one designated strong model — the **cross-predictor** — writes
an anonymous forecast of an unnamed model on the same task. We then measure what
actually happened and ask, cell by cell: whose forecast was closer, the model
predicting itself or the peer predicting it? The headline number is the
**wins-fraction** — the share of `(task, dimension)` cells where self beat cross.
Above 0.5 means introspective advantage exists.

Everything is driven by [`run.py`](run.py) (the 9 stages) and configured in
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
```

`plots.py`, `make_plots.py`, and `murphy.py` remain runnable standalone to
re-render or recompute from existing results (`--skip-figures` on `run.py`
gives a compute-only run).

---

## The pipeline, in plain language

### The 9 stages

| # | Stage | What it does | Cache |
|---|-------|--------------|-------|
| 1 | **Dataset** | Pick 30 SWE-bench Verified tasks, 10 easy / 10 medium / 10 hard, deterministically (seed 42). | `data/tasks.json` |
| 2 | **Rollouts** | 12 models × 30 tasks × 8 attempts. Each is a `mini-swe-agent` run in Docker (250-step budget) + the SWE-bench harness for the `solved` verdict. | `data/rollouts/<model>/<task>/r<i>.json` |
| 3 | **Ground truth** | Extract the 10-dimension outcome for each rollout (7 deterministic + 3 judge). | `data/outcomes/.../r<i>.json` |
| 4 | **Cross-predictor** | Rank models by solve rate; the top model becomes the peer predictor. | `data/cross_predictor.json` |
| 5 | **Forecasting** | Bottom 11 models self-forecast; the top model cross-forecasts. Two conditions (thinking / no-thinking), 5 samples each. | `data/forecasts/.../s<i>.json` |
| 6 | **Scoring** | Per-cell loss → wins-fraction per model per condition. | `results/scores.json` |
| 7 | **Analysis** | Descriptive readout (point estimates only). | `results/analysis.json` |
| 8 | **Murphy** | Brier decomposition of the binary dims (REL − RES + UNC; see below). | `results/murphy.json` |
| 9 | **Figures** | Every figure: the flat headline set (RQ1–3 + the two Murphy plots) and the full set (A/B/C figures). | `results/plots/`, `results/figures/` |

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

### Scoring — how a forecast becomes a win ([`metrics.py`](metrics.py))

The unit is a **`(task, dimension)` cell**, evaluated per forecaster per condition.

1. **Aggregate samples.** Self and cross predictions are each the **mean over the
   non-broken samples** for that cell.
2. **Per-cell loss** ([`per_forecast_loss`](../../src/scoring.py)) compares the one
   aggregated forecast `f` to the (up to 8) per-rollout ground-truth values `o`:
   - binary → **Brier**: `mean((f − o)²)`
   - continuous → **MAE**: `mean(|f − o|)`
   - *(e.g. `p_solve`: forecast `0.7` against a cell that solved 5/8 → mean of
     `(0.7 − oᵢ)²` over the valid rollouts.)*
3. **Win the cell.** `loss_self < loss_cross` → 1, tie → 0.5, else 0.
4. **wins-fraction** = wins ÷ scored cells, per model. That's the headline; RQ1 is
   its mean across models.

Losses stay in **native units** (Brier units, tokens, steps…) — there's no
cross-dimension normalization (the wins-fraction handles heterogeneity by
binarizing each cell).

### Missing data — one principle

> **Anything we can't measure or parse is marked missing and dropped from that
> cell's comparison. We never fill it in.**

A `(task, dimension)` cell is scored only if it clears **two independent gates**,
which have different "widths":

| Gate | Granularity | Fails when… |
|------|-------------|-------------|
| **Ground-truth retention** | per `(task, dimension)` — each dim independently | fewer than **5 of 8** rollouts produced a usable value for that dim |
| **Forecast usability** | per `(task, condition)` — **all 10 dims together** | **every** sample for that cell is broken (so self or cross has no number) |

The asymmetry worth knowing: ground-truth gaps drop **one dimension** at a time,
but a fully-broken forecast drops the **whole 10-dim column** for that task at
once — because a forecast sample is broken all-or-nothing (all keys or none).
Cross forecasts are shared across targets, so a broken *cross* cell drops that
`(task, dim)` for **every** model; a broken *self* cell drops only that model's.

What happens to the dropped cells:
- They leave the denominator — never counted as a win or a loss.
- If a model loses *every* cell, its wins-fraction is `NaN` and it's excluded from
  the mean (never biased toward 0).
- [`missingness_report`](analysis.py) tracks dropped-cell and broken-forecast
  rates and **flags** any model over 20% for human review (it flags; it does not
  auto-exclude).

One sentence to hand someone: *"If we can't measure or parse something we mark it
missing and drop it — a cell needs enough valid rollouts and a usable forecast on
both sides to be scored, and if a model loses too much data we flag it."*

### Analysis — what stage 7 reports ([`analysis.py`](analysis.py))

Point estimates only. The inferential statistics tier (bootstrap, confidence
intervals, significance tests, robustness battery) is **pruned for the MVP** and
will be reintroduced once the primary signal is validated.

- **RQ1 — Existence:** mean `wins(M, no_thinking)` across models. Is it > 0.5?
- **RQ2 — Scaling:** Spearman ρ of `wins(M)` vs solve rate. Do stronger models
  self-predict-win more? *(A monotonic-association summary and an **upper bound** —
  `wins` is confounded upward because weaker models are also noisier forecasters.
  Read the sign, not a magnitude or significance.)*
- **RQ3 — Elicitation:** mean paired `Δwins = wins(thinking) − wins(no_thinking)`.
  Does thinking help?
- Plus per-dimension raw advantages, per-model solve rates, and the missingness
  report.

Figures: [`plots.py`](plots.py) renders five descriptive figures (no CI bands,
p-values, or fitted curves) under `results/figures/primary__<cross>/`.

### Murphy decomposition — what drives the binary-dim errors ([`murphy.py`](murphy.py))

An optional downstream stage for the five **binary** dimensions. Each
forecaster's Brier score is split into the Murphy components

> **Brier = REL − RES + UNC** — miscalibration ("says 0.7, happens 0.3"),
> minus discrimination (does the forecast separate high-rate from low-rate
> cells?), plus outcome uncertainty (the base-rate variance floor — the
> rollout-to-rollout stochasticity neither forecaster can beat).

So "why does self lose to cross?" becomes "is it *miscalibrated about itself*
(REL) or *unable to discriminate* (RES)?". The primary estimator is **CORP**
(isotonic/PAV recalibration, Dimitriadis et al. 2021): bin-free, exact
identity, non-negative components. A fixed-bin estimator is kept alongside it
because its per-bin table is the reliability diagram.

Two methodology-driven choices: events are the per-rollout binary outcomes of
**exactly the cells the wins-fraction scored** (same gates), weighted
`1/n_valid(cell)` so the decomposed Brier reproduces the per-dim mean loss in
`scores.json` to machine precision; and the unit is **per-model, then averaged**
— pooling would mix per-model base rates, and the shared (anonymous) cross
forecast has zero between-model resolution by construction. A clearly-caveated
pooled view feeds the reliability diagrams.

Runs automatically as **stage 8** of `run.py` (figures in stage 9); standalone
re-runs against an existing cache:

```bash
python murphy.py --data-dir data_real --results-dir results_real   # -> murphy.json
python make_plots.py --results-dir results_real                    # plots/ incl. Murphy
python plots.py  --results-dir results_real                        # figures/ incl. C1, C2
```

The two Murphy figures land in **both** figure sets when `murphy.json` exists:
`plots/murphy_decomposition.png` + `plots/reliability_diagrams.png` (headline
set), and `figures/primary__<cross>/c1_murphy_components.png` + `c2_reliability_diagrams.png`
(full set) — C1 is the REL/RES bars vs the UNC floor, C2 the pooled reliability
diagrams.

---

## Running the FULL real eval on a cloud pod

A real run needs **Linux + Docker** (the SWE-bench eval is Unix-only and spawns a
container per task). It is CPU/RAM/disk-bound — **no GPU**. [`run_on_pod.sh`](run_on_pod.sh)
does the whole thing in one command: installs deps, checks Docker, builds the
venv, pulls the eval images, and launches the full 12-model × 30-task run. It is
**idempotent and resumable** — re-run it to continue after any interruption.

**1. Pick a machine.** A CPU box, ~16–32 vCPU / 32–64 GB RAM / ~100 GB disk, no
GPU. (RunPod CPU pod, or any Ubuntu VM with Docker. Note: RunPod *pods* must
allow Docker — the script tells you immediately if the host can't run it.)

**2. Get the code onto the pod**, either:
- `git clone <your-private-repo-url>` (push this repo to a private remote first), or
- copy it from your laptop: `rsync -az --exclude .venv --exclude data_real ./ root@<pod>:~/spc/`

**3. Provide the secrets:**
- `export OPENROUTER_API_KEY=sk-or-...` (required; or copy your local `.env` into
  the repo root on the pod — the script reads it).
- `export GITHUB_TOKEN=github_pat_...` (recommended). With it set, the script
  **auto-saves the output to `main` every ~10 minutes and once at the end** —
  both `data_real/` (the rollouts/forecasts you paid OpenRouter for, which make a
  re-run resumable) and `results_real/` (the scores). A crash or accidental pod
  delete then can't lose the work. Needs a fine-grained token with **Contents:
  Read and write** on this repo. Without it the output lives only on the pod.

**4. Launch it** (inside `tmux` so it survives disconnects):
```bash
tmux new -s eval
bash experiments/2026-05-16_self_prediction/run_on_pod.sh
```
Detach with `Ctrl-b d`; re-attach with `tmux attach -t eval`.

**5. When `results_real/run_summary.json` appears, it's done.** If you set
`GITHUB_TOKEN`, the results are already on `main` (`git pull` to get them); then
**delete the pod** so it stops billing. (No token? `scp`/`rsync` `results_real/`
back first.) Attach a **network volume** so the data survives a pod delete even
mid-run.

Two cost notes: the **machine** is cheap (~$0.5–1/hr; whole run ~$30–60), and the
**OpenRouter** spend (the agent + forecast calls) is the larger bill — order of
hundreds of dollars for all 12 models. Trim with `MODELS=` / `NTASKS=` /
`STEP_LIMIT=` env vars (see the script header) for a cheaper partial run.
