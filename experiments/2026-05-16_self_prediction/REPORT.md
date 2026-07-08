# Do coding agents know themselves? — Self- vs. peer-prediction on SWE-bench Verified

**Experiment 1 report — final results (recomputed 2026-07-08).**
Artifacts: [`results_real/`](results_real/) (scores, analysis, Murphy decomposition, figures). Pipeline: [`run.py`](run.py) (9 stages), spec constants in [`config.py`](config.py).

## TL;DR

Twelve open-weight models each attempted 30 SWE-bench Verified tasks (8 rollouts each). Every model forecast its **own** behavior on each task across 10 dimensions; the strongest model (GLM-5) forecast the behavior of an **anonymous** agent on the same tasks. Comparing forecast quality cell by cell:

1. **No aggregate introspective advantage** (RQ1): self-forecasts beat the peer's forecasts in 49.2% of cells — chance level. But the aggregate hides a split: the four most capable models are all above chance (up to 61.3%), qwen3-coder sits exactly at 0.500, and the remaining six are all below.
2. **Self-prediction tracks capability** (RQ2): Spearman ρ = 0.83 between a model's wins-fraction and its solve rate (descriptive; confounded upward, read the sign not the size).
3. **"Thinking" doesn't help** (RQ3): mean Δwins = +0.002.
4. **Miscalibration, not poor discrimination, drives forecast error** (Murphy decomposition): on every binary dimension, reliability (REL) dwarfs resolution (RES) for both self and cross forecasters. On `p_solve`, models are so overconfident (mean self Brier 0.492) that they underperform a constant 0.5 guess (0.25). Self-knowledge shows up only as a **calibration** advantage on stylistic dimensions (`defensive_code`, `express_uncertainty`) — exactly the dimensions an anonymous outside observer cannot infer from the task alone.

---

## 1. Research questions

An "introspective advantage" exists if a model can predict its own behavior better than a strong peer can predict it from the outside. Three pre-registered questions:

- **RQ1 (existence):** Do models beat a strong anonymous peer-predictor at forecasting their own SWE-bench behavior? Null: wins-fraction = 0.5.
- **RQ2 (scaling):** Does the advantage grow with model capability?
- **RQ3 (elicitation):** Does explicit reasoning ("thinking") improve self-prediction? Null: Δwins = 0.

## 2. Methodology

### 2.1 Design

| Element | Value |
|---|---|
| Executor models | 12 open-weight models via OpenRouter (5 Qwen, 2 Google, 3 Zhipu, 1 DeepSeek, 1 Moonshot) |
| Tasks | 30 SWE-bench Verified instances, stratified 10 easy / 10 medium / 10 hard by the human difficulty annotation, selected deterministically (seed 42) |
| Rollouts | 8 per (model, task): `mini-swe-agent` in Docker, 250-step budget, SWE-bench harness verdict → 2,880 rollouts |
| Forecast conditions | `no_thinking` / `thinking`, 5 samples each at temperature 0.7 |
| Cross-predictor | Designated **post-hoc** as the highest-solve-rate model → **z-ai/glm-5** (64.7% solve rate) |
| Judge / extraction fallback | `gpt-4o-mini` (not one of the 12 executors — no circularity) |

The pipeline ([`run.py`](run.py)) is 9 cached, resumable stages: dataset → rollouts → ground truth → cross-predictor designation → forecasting → scoring → analysis → Murphy decomposition → figures.

### 2.2 The 10-dimension battery

Each model forecasts, and we measure, 10 dimensions per task. Five are binary (Brier-scored probabilities), five continuous (MAE-scored point estimates):

| Dimension | Kind | Ground truth source |
|---|---|---|
| `p_solve` | binary | SWE-bench harness verdict |
| `p_first_attempt_correct` | binary | trajectory analysis |
| `documentation` (writes docs/comments) | binary | judge on the patch |
| `defensive_code` | binary | judge on the patch |
| `express_uncertainty` (hedging in reasoning) | binary | judge on the reasoning text |
| `avg_tokens_step` | continuous | trajectory |
| `total_expected_steps` | continuous | trajectory |
| `explore_edit_ratio` | continuous | trajectory |
| `mean_action_block_size` | continuous | trajectory |
| `n_constraint_violations` | continuous | trajectory + rule list |

Dims 1–2 are "capability" dimensions, 3–10 "behavioral". An empty patch short-circuits `documentation`/`defensive_code` to a real 0 (not missing).

### 2.3 Forecast elicitation

- **Self:** the 11 non-cross models receive the task, the verbatim agent scaffold (system prompt, rules), and forecast their own 10 outcomes.
- **Cross:** GLM-5 receives the same material but forecasts "an AI coding agent" — **the target is never named**, and one cross-forecast per task is shared across all 11 targets. Any between-model signal in self-forecasts therefore reflects information the outside view cannot have.
- **Thinking control:** hybrids toggle the API `reasoning` parameter; non-reasoning models get a step-by-step prompt suffix in the thinking condition; three RL-trained reasoners with unreliable "off" levers get `reasoning:{max_tokens:1024}` plus an anti-thinking instruction in `no_thinking`.
- **Extraction:** one programmatic JSON parse, then a single `gpt-4o-mini` fallback call; still-broken samples are marked missing. Out-of-range values are clamped, never dropped.

### 2.4 Scoring

The unit is a **(task, dimension) cell**, per forecaster per condition:

1. Predictions = mean over non-broken samples (of 5).
2. Per-cell loss vs. the up-to-8 rollout ground truths: Brier `mean((f−o)²)` for binary, MAE for continuous — native units, no cross-dimension normalization.
3. Cell win: `loss_self < loss_cross` → 1, tie → 0.5, else 0.
4. **wins(M) = wins ÷ scored cells.** RQ1 is its mean across the 11 models.

**Missing data (drop, never impute):** a cell is scored only if (a) ≥ 5 of 8 rollouts yielded a usable ground truth for that dimension and (b) both self and cross have at least one non-broken forecast sample. Dropped cells leave the denominator. Models exceeding 20% dropped cells or broken forecasts are flagged for review (flagging only — no auto-exclusion).

**Known caveat (W2):** loss = bias² + forecast variance; noisier forecasters carry inflated self-loss. Since weaker models are also noisier forecasters, RQ2's correlation is **confounded upward** — reported as a descriptive upper bound.

### 2.5 Murphy decomposition (downstream analysis, stage 8)

For the five binary dimensions, each forecaster's Brier score is decomposed ([`murphy.py`](murphy.py)) as

> **BS = REL − RES + UNC** — reliability (miscalibration penalty; 0 = perfectly calibrated), resolution (discrimination credit; ceiling = UNC), uncertainty (base-rate variance `p(1−p)`, the floor no forecaster can beat — the rollout-to-rollout stochasticity of the agent itself).

Estimator: **CORP** (isotonic/PAV recalibration; Dimitriadis et al. 2021, PNAS) — bin-free, exact identity, non-negative components; a fixed-bin estimator feeds the reliability diagrams. Events are the per-rollout binary outcomes of **exactly the cells the wins-fraction scored**, weighted `1/n_valid(cell)` so the decomposed Brier reproduces the pipeline's per-dim mean loss to machine precision (verified across all 220 slices). Decomposition runs per model, then averages across models; a pooled view exists but is caveated (pooling mixes per-model base rates, and the shared cross-forecast has zero between-model resolution by construction).

### 2.6 Provenance and scope

- **Model set:** two models (`openai/gpt-4.1-mini`, `qwen/qwen3-235b-a22b-2507`) were removed from the study on 2026-07-08 with all their data; every number here is recomputed from the remaining 12 models' cached raw data. (Both removed models had been above-chance self-predictors off the capability trend; the aggregate RQ1 dropped from 0.510 to 0.492 and RQ2's ρ rose from 0.55 to 0.83 as a result.)
- **Inference tier pruned (MVP):** point estimates only — no confidence intervals, significance tests, or bootstrap. Read all results descriptively.
- Judge policy for oversized inputs: truncate to 120k chars and judge.

## 3. Results

### 3.1 Execution

2,880 rollouts (12 × 30 × 8); 111 errored (3.9%) and are treated as missing data; 865 solved (30.0% raw). Per-model solve rates span 5.6% (qwen3-8b) to 64.7% (glm-5). Pipeline LLM accounting (forecasts + judging + extraction; agent rollout spend is tracked separately by OpenRouter): 6,243 calls, $5.57, 14.1M tokens.

### 3.2 RQ1 — existence: no aggregate advantage, a clear capability split

**Mean wins (no_thinking) = 0.492** vs. null 0.5.

| Model | Solve rate | wins (no_thinking) | wins (thinking) |
|---|---|---|---|
| moonshotai/kimi-k2.5 | 0.603 | **0.613** | 0.617 |
| deepseek/deepseek-v3.2 | 0.548 | **0.580** | 0.602 |
| z-ai/glm-4.5 | 0.426 | **0.542** | 0.533 |
| z-ai/glm-4.7 | 0.569 | **0.505** | 0.565 |
| qwen/qwen3-coder | 0.239 | 0.500 | 0.554 |
| qwen/qwen3-next-80b | 0.152 | 0.491 | 0.472 |
| qwen/qwen3-32b | 0.098 | 0.482 | 0.435 |
| google/gemini-2.5-flash-lite | 0.130 | 0.453 | 0.417 |
| qwen/qwen3-14b | 0.060 | 0.438 | 0.417 |
| qwen/qwen3-8b | 0.056 | 0.433 | 0.432 |
| google/gemini-3.1-flash-lite | 0.219 | 0.377 | 0.396 |

![RQ1](results_real/plots/rq1_wins_by_model.png)

### 3.3 RQ2 — scaling: strongly monotone (descriptive)

**Spearman ρ = 0.827** between wins and solve rate (OLS slope on the scatter: +0.25 wins per unit solve rate). The four models above chance are the four strongest self-predictors and among the most capable. Two visible off-trend points: gemini-3.1-flash-lite (0.377, far below trend) and glm-4.7 (0.505, below trend for its 0.569 solve rate). Reminder: upper bound, not a test (§2.4 W2).

![RQ2](results_real/plots/rq2_wins_vs_solverate.png)

### 3.4 RQ3 — elicitation: thinking does not help

**Mean Δwins = +0.002.** Per-model deltas are small and mixed (−0.047 for qwen3-32b to +0.060 for glm-4.7); no exclusions were applied in this readout. The thinking condition also does not change any other conclusion (see the per-condition columns throughout).

![RQ3](results_real/plots/rq3_thinking_delta.png)

### 3.5 Where self wins and loses (per dimension)

Mean raw introspective advantage (`mean cross loss − mean self loss`, native units, no_thinking; positive = self better):

| Dimension | raw advantage | Reading |
|---|---|---|
| `p_solve` | −0.061 Brier | self worse |
| `p_first_attempt_correct` | −0.095 Brier | self worse (largest binary deficit) |
| `documentation` | −0.024 Brier | self slightly worse |
| `defensive_code` | **+0.060 Brier** | self better |
| `express_uncertainty` | **+0.028 Brier** | self better |
| `avg_tokens_step` | −45.1 tokens | self worse |
| `total_expected_steps` | +2.8 steps | self better |
| `explore_edit_ratio` | −0.033 | self slightly worse |
| `mean_action_block_size` | +0.256 cmds/step | self better |
| `n_constraint_violations` | +0.004 | tie |

The per-(model, dimension) wins heatmap ([figures/primary__z-ai__glm-5/b2_wins_heatmap.png](results_real/figures/primary__z-ai__glm-5/b2_wins_heatmap.png)) shows the same split with per-model texture: capability columns are mostly blue (self loses), `defensive_code`/`express_uncertainty` mostly red, and the continuous style dims are strongly model-idiosyncratic (e.g. `avg_tokens_step` wins are 1.00 for qwen3-next-80b but 0.00 for the small Qwens — cell-level wins and mean losses can disagree because means are outlier-dominated).

### 3.6 Murphy decomposition — miscalibration is the error budget

Mean across models, no_thinking (CORP; `BS = REL − RES + UNC`, exact):

| Dimension | BS self / cross | REL self / cross | RES self / cross | UNC | base rate |
|---|---|---|---|---|---|
| `p_solve` | 0.492 / 0.430 | 0.367 / 0.332 | 0.035 / 0.061 | 0.160 | 0.281 |
| `p_first_attempt_correct` | 0.306 / 0.211 | 0.205 / 0.116 | 0.035 / 0.041 | 0.135 | 0.206 |
| `documentation` | 0.260 / 0.236 | 0.091 / 0.078 | 0.015 / 0.026 | 0.184 | 0.325 |
| `defensive_code` | 0.315 / 0.375 | **0.111 / 0.169** | 0.010 / 0.008 | 0.213 | 0.570 |
| `express_uncertainty` | 0.282 / 0.310 | **0.126 / 0.158** | 0.006 / 0.010 | 0.162 | 0.536 |

Three facts jump out:

1. **REL ≫ RES on every dimension, for both forecasters.** Discrimination captures at most ~38% of the available signal (RES/UNC = 0.38 for cross `p_solve`; usually far less, e.g. 4% for self `express_uncertainty`). The excess Brier above the UNC floor is essentially all miscalibration.
2. **`p_solve` overconfidence is extreme.** Mean self Brier 0.492 exceeds the constant-0.5 baseline (0.25) and triples the base-rate forecast (UNC 0.160). The reliability diagrams show the whole cloud far below the diagonal: models state 0.6–0.9 where observed frequencies run 0.0–0.6. Recalibrating the forecasts (keeping their ordering) would cut self Brier to ~0.125 — nearly all avoidable error is calibration.
3. **The self-advantage, where it exists, is a calibration advantage.** On `defensive_code` and `express_uncertainty` — the two dims where self beats cross in wins — self REL is markedly lower while RES is equally negligible for both.

Calibration also tracks capability at the per-model level: on `p_solve`, kimi-k2.5 is the best self-calibrated (REL 0.085 at base rate 0.606) and qwen3-14b the worst (REL 0.616 at base rate 0.059 — stating high confidence while solving ~6%).

![Murphy decomposition](results_real/plots/murphy_decomposition.png)
![Reliability diagrams](results_real/plots/reliability_diagrams.png)

### 3.7 Data quality

Dropped-cell rates (ground-truth retention) are low: 10% for qwen3-coder, 3.3% for qwen3-next-80b, 0% for everyone else — no model near the 20% flag line. Broken-forecast flags: deepseek-v3.2 (23.3% broken samples in the thinking condition — above the 20% flag line; its thinking-condition numbers rest on fewer cells) and gemini-3.1-flash-lite at 10% (below the line). Nothing was excluded; per policy the flags are for human review.

## 4. Discussion

### 4.1 The headline is a split, not a null

The pre-registered aggregate (RQ1 = 0.492) says models as a class have no introspective advantage. But the aggregate averages over a strong monotone trend (RQ2): kimi-k2.5, deepseek-v3.2, glm-4.5 and glm-4.7 all beat the strongest available peer-predictor at forecasting themselves, while every model below ~0.25 solve rate loses to it. Two readings are compatible with this: (a) genuine introspection emerges with capability; (b) the W2 confound — weaker models are noisier forecasters, and forecast variance mechanically inflates their self-loss while the shared cross-forecast's variance is constant across targets. The design cannot separate these; the confound direction means ρ = 0.83 is a ceiling, not an estimate. What the confound *cannot* explain is the dimension pattern (§4.2).

### 4.2 Self-knowledge lives in style, not capability

The cross-predictor sees the task but not the identity of the agent it is predicting. So its forecasts can only encode the *outside view* — task difficulty and generic agent behavior. Self-forecasts could in principle add private information: "I tend to write defensive code", "I hedge a lot", "this kind of repo trips me up". The results show that the private signal is real but narrow: self wins precisely on the stylistic propensities (`defensive_code` +0.060 Brier, `express_uncertainty` +0.028; also step-count and action-granularity among continuous dims), and the Murphy decomposition localizes the advantage in **reliability** — models state better-calibrated probabilities about their own style than an outsider can, while discriminating task-to-task no better than it. On capability (`p_solve`, `p_first_attempt_correct`) the outside view wins outright: knowing the task tells you more about success than the agent's self-model adds — and the agent's self-model is actively biased (overconfidence), making it *worse* than ignorance-with-base-rates.

### 4.3 Everyone is overconfident about agents — including the best model

GLM-5's cross-forecast REL on `p_solve` is 0.332 — barely better than the self-forecasters' 0.367 and still catastrophic against a 0.160 floor. Overconfidence about coding-agent success generalizes beyond introspection failure: even the most capable model in the pool, predicting an anonymous agent, overshoots observed solve rates by a wide margin. Two mechanisms plausibly combine: (i) LLM probability outputs skew high on task-success questions generally; (ii) forecasters underestimate SWE-bench Verified difficulty under a 250-step mini-agent scaffold. The per-model gradient (kimi REL 0.085 vs qwen3-14b 0.616) suggests a Dunning–Kruger-like structure: skill and self-assessment accuracy travel together.

### 4.4 Practical read: recalibrate before trusting self-reports

If an application needs an agent's own probability of success (e.g., for deferral or routing), the raw self-report is worse than useless at these capability levels — a constant base-rate would beat it. But because the error is nearly all *reliability*, a small amount of historical data buys a large fix: isotonic/Platt recalibration of self-forecasts would recover ~75% of `p_solve` Brier (0.492 → ~0.125). What recalibration cannot buy is discrimination — RES ≤ 0.06 everywhere means even recalibrated forecasts barely separate solvable from unsolvable tasks. Discrimination, not calibration, is the binding research problem for agent self-prediction.

### 4.5 Limitations

- **Descriptive only.** The inference tier (bootstrap CIs, tests) is pruned; with 11 models × 30 tasks, per-model wins carry substantial sampling noise (~±0.03–0.05 s.e. from cell counts alone), and none of the RQ readouts should be quoted with significance language.
- **W2 variance confound** on RQ1/RQ2 (§2.4): self- and cross-forecast variances differ and do not cancel.
- **One cross-predictor.** All comparisons are against GLM-5; the alt-cross sensitivity battery was pruned. "Beats the peer" means this peer.
- **Shared anonymous cross-forecast** is a design choice, not a bug: it defines the outside-view baseline. But it means cross has zero between-model resolution by construction, which flatters self on any dimension with large stable between-model differences (arguably the point of the comparison).
- **Judge noise** on dims 3–5 (gpt-4o-mini, yes/no, truncation at 120k chars) adds symmetric outcome noise (inflating UNC) for both sides.
- **RQ3 levers are bundles** for the three RL-trained reasoners (API cap + prompt instruction), so the thinking null is about this elicitation, not about reasoning per se. deepseek-v3.2's thinking condition additionally lost 23% of forecast samples to extraction failures.
- **Scale:** 30 tasks, 8 rollouts; UNC floors of 0.14–0.21 mean per-cell ground truth is itself noisy. Two models were removed post-hoc from the release (§2.6) — the qualitative conclusions (REL ≫ RES, style-vs-capability split, thinking null) were unchanged by the removal, but RQ1's position relative to 0.5 moved from +0.010 to −0.008, underscoring how fragile the aggregate is.

### 4.6 What would strengthen the conclusions

Reinstating the hierarchical bootstrap for CIs on RQ1–RQ3; a variance-matched scoring rule (or sample-mean forecasts with variance correction) to kill W2; multiple cross-predictors including a named (non-anonymous) condition to price the value of identity information; more tasks per difficulty stratum; and a calibration-trained baseline (self-forecast + isotonic map learned on held-out tasks) to test whether the residual discrimination gap closes with capability.

---

## 5. Reproduction

```bash
python run.py --quick        # mocked 5-model smoke run (~20 s, no network)
python run.py                # full mocked dry-run
python run.py --real         # full eval (Linux + Docker; see README §"real eval")
```

One invocation produces every artifact: `results/{scores,analysis,murphy}.json`, headline figures in `results/plots/`, full figure set in `results/figures/primary__<cross>/`. Standalone re-renders: `murphy.py`, `make_plots.py`, `plots.py` (all take `--results-dir`).

## 6. Artifact map

| File | Content |
|---|---|
| [`results_real/run_summary.json`](results_real/run_summary.json) | headline numbers + accounting |
| [`results_real/scores.json`](results_real/scores.json) | per-model wins + per-dim losses, both conditions |
| [`results_real/analysis.json`](results_real/analysis.json) | RQ1/RQ2/RQ3, per-dim advantages, missingness |
| [`results_real/murphy.json`](results_real/murphy.json) | Murphy decomposition (per model / mean / pooled) |
| [`results_real/plots/`](results_real/plots/) | RQ1–3 + Murphy headline figures |
| [`results_real/figures/primary__z-ai__glm-5/`](results_real/figures/primary__z-ai__glm-5/) | full figure set (A1–A3, B1–B2, C1–C2) |
