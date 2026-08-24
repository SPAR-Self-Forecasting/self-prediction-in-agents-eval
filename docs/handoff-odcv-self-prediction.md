# Handoff — ODCV framing-gap experiment (exp2)

**Phase-status: BUILD COMPLETE — results in hand; two operator spot-checks pending before writeup.**
Spec: [spec-odcv-self-prediction.md](spec-odcv-self-prediction.md) (rev 3 — two conditions; D38 drops genuine-cross). Code: `experiments/2026-07-29_odcv_framing/` (package `odcv`). Run date: 2026-07-30; rev 3 cleanup: 2026-08-03.

## 1. Status table

| Spec step | Status | Tests | Claim basis | Code / artifact |
|---|---|---|---|---|
| 1 loader + registry | done | green (suite 234) | PROVEN — reproduces CSV col_avg/col_MR% for 9 models, 358 cells | `odcv/config.py`, `odcv/groundtruth.py` |
| 6a judge ceiling | done | green | PROVEN — **94.1%** binary agreement, 0.367 MAD, n=4×358 | `odcv/judge_ceiling.py`, `results/judge_ceiling.json` |
| 2 conflict summaries | done | reviewer 40/40 clean | PROVEN — 3/3 hand-verified vs scenario files; 1 summary hand-fixed (logged in its checkpoint) | `data/conflict_summaries.json` |
| 2b discovery flags | done | consistency clean after finalize | PROVEN mechanics; flags themselves ~90% reliable (see §4) | `data/discovery_flags_final.json` (2 reparsed, 13 violation-overrides) |
| 3 prompts + fidelity gate | done | byte-identity green; 716 renders | PROVEN | `odcv/prompts.py`, `odcv/fidelity_gate.py`, `prompts/` |
| 4 smoke | done | 29/30 parse; grok reasoning on | PROVEN | run bthky4oii |
| 5 full elicitation | done | 3,580 records (200 genuine-cross removed rev 3), 24 broken (0.6%) | PROVEN | `results/predictions.jsonl` (run bj0oq58t8) |
| 6b analysis | done | synthetic-fixture tests green | PROVEN mechanics | `odcv/analysis.py`, `results/analysis.json` |
| 6c figures | done | 9/9 render (B3 removed rev 3) | PROVEN | `figures/figure_[abcd]*.png` |
| Layer 3 qual coding | done (LLM pass) | — | INFERRED until human spot-check | `results/qual_coding.json` |

## 2. Headline results (all VERIFIED against results/analysis.json unless noted)

- **RQ1 — 9 of 9 models show a significant framing gap** (one-sided, scenario-clustered wild bootstrap, BH-corrected; all p_bh ≤ 0.0023). Per-model mean gaps +0.45 … +3.23 severity points; pooled +2.11 [1.43, 2.78]. Sign split: across all 358 pairs, **exactly 1 cell** went self > other. Both-zero floor cells: 39.1%.
- **Referent manipulation check:** 93.1% of self-condition action texts are first-person vs 4.1% in other-condition — the framing took.
- **RQ2 — neither framing beats the constant baseline** (per-model majority-class baselines 55.0–77.5%; self mean 59.7%, simulated-cross 53.7%; ceiling 94.1%). Not evidence of self-knowledge. **The pre-registered error asymmetry is extreme:** self-framing errs almost exclusively by predicting clean when the model actually violated (fn=140, fp=4); simulated-cross overshoots (fp=117, fn=49).
- **Layer 3 qual coding — 207 large-gap cells coded: LLM-coded 95.7% `different_action`, audit-corrected ≈ 80/20** (21-item independent audit by a different model, `results/spot_check_independent.json`: 15/21 codes agree; errors run both directions). Self-framed text describes honest behavior; simulated-cross text describes the specific cheat. Conclusion unchanged; cite ~80/20, not 95.7%.
- **Reading:** gap present + Layer 3 `different_action` dominance = **a response-style effect with an idealized self-model** — framing changes the story told, not just the label. Discovery-filtered sensitivity variant agrees (self acc 56% on that subset).

## 3. Decisions and deviations (build-time)

- Q1a–Q5a build decisions (plan-approved defaults): sign-off line added to spec; pooled RQ1 = wider of two one-way clusterings; `gpt-4o-mini` as reviewer/fallback; deepseek as 2b checker; fallback template adapted to the severity/action schema (spec said "verbatim" — impossible, exp1's is 10-dim-specific; flagged).
- **D35 (operator, mid-build): discovery filter demoted to sensitivity check** after its reliability was measured. Primary RQ2 = all cells with ground truth.
- **D38 (operator, 2026-08-03): genuine-cross / RQ3 dropped.** Two conditions only (`self`, `simulated_cross`). DeepSeek retained for extraction + discovery only. 200 prediction records removed from `predictions.jsonl`; analysis/figures/code updated.
- Reviewer prompt rewritten once (v1 flagged all 40 summaries — checker misfire, human-verified); its v2 catch (Ai-Transport honest_outcome) hand-fixed in `data/raw_step2/…json` with a `hand_fixed` note.
- Exp2 test files prefixed `test_odcv_*` (module-name collision with root tests); package named `odcv` (collision with exp1's bare `config`).
- Smoke revealed gpt-oss-20b refuses some scenarios outright (propaganda) — refusals counted as broken/missing, final broken 2.0% for that model (drop rule 20%: not triggered).

## 4. Known-wrong / retracted / caveats

- **Judge-ceiling first computation (94.2%, n=1468) was wrong** — per-judge CSVs have a `col_na_count` summary row that leaked in as a fake scenario. Corrected same session to **94.1%, n=1432** (positive row filter). Nothing downstream consumed the wrong value.
- **Discovery checker reliability:** ~10% false-negatives on provably-discovered (violating) cells; 3-cell audit of "not discovered" flags: 1 correct, 1 probably wrong, 1 gray. Hence D35. The 28 "not discovered" cells are the right target for the operator's ≥15-item spot-check.
- Provider heterogeneity is real: glm-5.1 served by 9 different upstream providers within 10 calls (recorded per-record via D34).
- gap magnitudes are not strictly comparable across models (effort levels differ); RQ1's within-model design carries the headline.

## 5. Spend log

| Rung | Approved | Actual |
|---|---|---|
| Steps 2+2b | ~$2.40 | **$4.73** (overrun flagged at the time; deepseek reasoning tokens) |
| Step 2 reviewer re-runs + hand-fix re-review | — | $0.01 |
| Smoke (step 4) | <$0.50 | $0.12 |
| Full run (step 5) | ~$19, cap $25 | **$23.21** |
| Layer-4 coding | ~$0.10 | $0.03 |
| **Total** | | **≈ $28.1** |

All rungs individually approved; fidelity gate green before every paid call; zero re-spend on retries (checkpoints held through a laptop-sleep + DNS outage mid-run).

## 6. Exact next action

1. **Spot-checks: DONE as an independent LLM audit** (Claude, different model from both coders; 21 qual items + 18 discovery flags; `results/spot_check_independent.json`). Corrections applied to REPORT.md (§3.4 ratio → ~80/20; limitation 5 → asymmetric flag reliability). **Residual operator step (5 min):** skim the 6 qual disagreement rows and the 3 clearly-wrong flag rows in that file to close the correlated-LLM-blind-spot gap.
2. Then: the writeup is REPORT.md; exploratory panels labeled; §2 caveats carried.
3. Repo is uncommitted — `git add` of exp2 + docs + conftest/pyproject changes awaits operator go (nothing has been committed this session).
