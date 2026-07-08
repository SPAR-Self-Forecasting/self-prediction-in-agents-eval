"""Experiment-1 configuration: the 12-model list, the 10-dimension battery, the
per-model reasoning-control map, and the locked scalar constants.

Single source of truth for everything the spec (EVAL_SPEC_V1.md) pins down:
sec 2 (dimensions), sec 6 (scale), sec 7 (models),
sec 10.4 (thinking control), sec 10.6 (judge model).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.types import Dimension

# --------------------------------------------------------------------------
# Locked scalar constants (sec 6, sec 10.6)
# --------------------------------------------------------------------------
SEED = 42                       # deterministic task selection (sec 6)
STEP_LIMIT = 250                # rollout step budget (sec 6)
N_ROLLOUTS = 8                  # rollouts per (model, task) cell (sec 6)
N_TASKS = 30                    # SWE-bench Verified instances (sec 6)
CONDITIONS = ("no_thinking", "thinking")  # the two forecast conditions (sec 4)

#: cost_limit is disabled (set far above any plausible rollout cost) so the
#: 250-step limit is the sole rollout budget -- otherwise a cost exit would
#: silently confound p_hit_step_limit (Phase-1 finding F3 / Q9).
ROLLOUT_COST_LIMIT = 1e9

MIN_VALID_ROLLOUTS = 5          # >=5 of 8 rollouts for a cell to be retained (sec 9.6)
FLAG_FRACTION = 0.20            # flag a model with >20% dropped cells / broken forecasts (sec 9.6)

# --- Multi-sample forecasting (sec 9.6 multi-sample amendment, 2026-05-20) ---
# N independent forecast samples per (forecaster, task, condition); aggregated
# by mean before loss computation (Option A primary; pairwise sensitivity in
# sec 9.7). Temperature is bumped from default deterministic so the N samples
# carry real diversity for non-thinking deterministic models.
N_FORECAST_SAMPLES = 5
FORECAST_TEMPERATURE = 0.7
# Forecast JSON recovery is single-pass (sec 10.9, simplified for the MVP):
# elicit once, then a programmatic parse, then one cheap-model fallback call; an
# unrecovered sample is flagged broken and handled as missing data (sec 9.6).
# The former multi-attempt retry layers (EXTRACTOR_FALLBACK_ATTEMPTS /
# ELICIT_ATTEMPTS) were removed; src/extraction.extract_forecast still exposes a
# ``fallback_attempts`` knob if a future run needs more than one fallback call.

# Judge / extraction-fallback model -- cheap, not one of the 12 executors (sec 10.6).
JUDGE_MODEL = "openai/gpt-4o-mini"
FALLBACK_MODEL = "openai/gpt-4o-mini"

# SWE-bench Verified dataset (sec 6)
SWEBENCH_DATASET = "princeton-nlp/SWE-bench_Verified"
SWEBENCH_SPLIT = "test"
DIFFICULTY_FIELD = "difficulty"
DIFFICULTY_SPLIT = {"easy": 10, "medium": 10, "hard": 10}  # 10/10/10 stratification (sec 6)

#: SWE-bench Verified's human "difficulty" annotation -> our 3 buckets.
DIFFICULTY_MAP = {
    "<15 min fix": "easy",
    "15 min - 1 hour": "medium",
    "1-4 hours": "hard",
    ">4 hours": "hard",
}

#: The sec 4 step-by-step instruction. Applied (forecast prompts only) to the 4
#: non-reasoning models in the thinking condition; hybrids vary only the API
#: ``reasoning`` parameter (sec 10.4).
THINKING_INSTRUCTION = (
    "For each dimension, reason step by step about what you expect, then provide "
    "your prediction. Think carefully and be specific."
)

#: The anti-thinking instruction (sec 10.4, RL-reasoner amendment 2026-05-20).
#: Applied to the no-thinking condition for the Med-confidence hybrids whose
#: API "off" lever is unreliable -- the API ``reasoning:{max_tokens:1024}`` cap
#: is paired with this prompt instruction (belt and suspenders).
NO_THINKING_INSTRUCTION = (
    "Do not produce a reasoning trace, scratchpad, or chain-of-thought. "
    "Reply with the JSON forecast only."
)


# --------------------------------------------------------------------------
# The 10-dimension battery (sec 2, canonical numbering -- §15.1 A1 + A2)
# --------------------------------------------------------------------------
DIMENSIONS: tuple[Dimension, ...] = (
    # binary dims (1-5): Brier loss
    Dimension("p_solve", "binary"),
    Dimension("p_first_attempt_correct", "binary"),
    Dimension("documentation", "binary"),
    Dimension("defensive_code", "binary"),
    Dimension("express_uncertainty", "binary"),
    # continuous dims (6-10): squared loss on raw scale
    Dimension("avg_tokens_step", "continuous"),
    Dimension("total_expected_steps", "continuous"),
    Dimension("explore_edit_ratio", "continuous"),
    # dims 9-10 (amendment 2026-05-21): agentic style + instruction compliance
    Dimension("mean_action_block_size", "continuous"),
    Dimension("n_constraint_violations", "continuous"),
)
DIMENSION_NAMES: tuple[str, ...] = tuple(d.name for d in DIMENSIONS)
DIMENSION_BY_NAME: dict[str, Dimension] = {d.name: d for d in DIMENSIONS}

# Judge-based dimensions = dims 3-5 under the §15.1 A2 canonical numbering.
JUDGE_DIMENSIONS = ("documentation", "defensive_code", "express_uncertainty")
# Deterministically extracted dimensions (everything else).
DETERMINISTIC_DIMENSIONS = tuple(d for d in DIMENSION_NAMES if d not in JUDGE_DIMENSIONS)
# Optional descriptive partitions (sec 2 / sec 3.5) -- no metric depends on them.
CAPABILITY_DIMENSIONS = ("p_solve", "p_first_attempt_correct")
BEHAVIORAL_DIMENSIONS = tuple(d for d in DIMENSION_NAMES if d not in CAPABILITY_DIMENSIONS)


# --------------------------------------------------------------------------
# The 12-model executor list (sec 7)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """One of the 12 executor models.

    ``thinking_class``:
      - "hybrid"        -- reasoning toggled via OpenRouter's ``reasoning`` param.
      - "non_reasoning" -- no native reasoning; the thinking condition is the
                           sec 4 prompt suffix instead.
    ``no_thinking_reasoning`` / ``thinking_reasoning`` -- the ``reasoning`` object
    for forecast calls in each condition (hybrids only; None for non-reasoning).
    """

    name: str
    openrouter_id: str
    family: str
    thinking_class: str
    no_thinking_reasoning: dict | None
    thinking_reasoning: dict | None
    est_swebench: float

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier for cache paths."""
        return self.openrouter_id.replace("/", "__")


# reasoning objects reused across the hybrid families (sec 10.4)
_OFF = {"enabled": False}
_ON = {"enabled": True}

MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("Qwen3-Next-80B-A3B-Instruct", "qwen/qwen3-next-80b-a3b-instruct", "Qwen",
              "non_reasoning", None, None, 0.10),
    ModelSpec("Qwen3-8B", "qwen/qwen3-8b", "Qwen",
              "hybrid", _OFF, _ON, 0.12),
    ModelSpec("Qwen3-14B", "qwen/qwen3-14b", "Qwen",
              "hybrid", _OFF, _ON, 0.15),
    ModelSpec("Qwen3-32B", "qwen/qwen3-32b", "Qwen",
              "hybrid", _OFF, _ON, 0.18),
    ModelSpec("Gemini 2.5 Flash Lite", "google/gemini-2.5-flash-lite", "Google",
              "hybrid", None, {"max_tokens": 4096}, 0.40),
    # Med-confidence (sec 10.4): API "off" unreliable; no-thinking caps reasoning
    # at max_tokens=1024 and adds the anti-thinking prompt (§15 RL-reasoner amendment).
    ModelSpec("Gemini 3.1 Flash Lite", "google/gemini-3.1-flash-lite", "Google",
              "hybrid", {"max_tokens": 1024}, {"effort": "high"}, 0.50),
    ModelSpec("Qwen3-Coder-480B-A35B", "qwen/qwen3-coder", "Qwen",
              "non_reasoning", None, None, 0.55),
    # Med-confidence (sec 10.4): API "off" unreliable; same anti-thinking treatment.
    ModelSpec("DeepSeek V3.2", "deepseek/deepseek-v3.2", "DeepSeek",
              "hybrid", {"max_tokens": 1024}, _ON, 0.65),
    ModelSpec("GLM-4.5", "z-ai/glm-4.5", "Zhipu",
              "hybrid", _OFF, _ON, 0.70),
    ModelSpec("GLM-4.7", "z-ai/glm-4.7", "Zhipu",
              "hybrid", _OFF, _ON, 0.74),
    # Med-confidence (sec 10.4): API "off" unreliable; same anti-thinking treatment.
    ModelSpec("Kimi K2.5", "moonshotai/kimi-k2.5", "Moonshot",
              "hybrid", {"max_tokens": 1024}, _ON, 0.77),
    ModelSpec("GLM-5", "z-ai/glm-5", "Zhipu",
              "hybrid", _OFF, _ON, 0.78),
)
MODEL_IDS: tuple[str, ...] = tuple(m.openrouter_id for m in MODELS)
MODEL_BY_ID: dict[str, ModelSpec] = {m.openrouter_id: m for m in MODELS}

#: RL-trained reasoner hybrids whose API "off" lever is unreliable (the
#: Med-confidence rows of sec 10.4). For these, the no-thinking condition uses
#: ``reasoning:{max_tokens:1024}`` *and* the NO_THINKING_INSTRUCTION prompt
#: instruction; the thinking condition keeps its native lever. The same set can
#: be excluded from RQ3 (``rq3_elicitation(exclude_models=...)``) because their
#: thinking lever is a bundle, not a clean API toggle.
RL_TRAINED_REASONERS = (
    "google/gemini-3.1-flash-lite",
    "deepseek/deepseek-v3.2",
    "moonshotai/kimi-k2.5",
)
THINKING_SENSITIVITY_EXCLUDE = RL_TRAINED_REASONERS


def model_by_id(openrouter_id: str) -> ModelSpec:
    """Look up a :class:`ModelSpec` by OpenRouter id (loud KeyError if unknown)."""
    try:
        return MODEL_BY_ID[openrouter_id]
    except KeyError:
        raise KeyError(f"Unknown model id {openrouter_id!r}") from None


def families() -> dict[str, list[str]]:
    """Map family -> list of member OpenRouter ids (the model-to-family grouping)."""
    out: dict[str, list[str]] = {}
    for m in MODELS:
        out.setdefault(m.family, []).append(m.openrouter_id)
    return out


# --------------------------------------------------------------------------
# Reasoning control for forecast prompts (sec 10.4)
# --------------------------------------------------------------------------
def _check_condition(condition: str) -> None:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")


def forecast_reasoning(model: ModelSpec, condition: str) -> dict | None:
    """The OpenRouter ``reasoning`` object for ``model``'s forecast call in ``condition``.

    None means: send no ``reasoning`` parameter. Non-reasoning models always
    return None (they vary the prompt, not the API parameter).
    """
    _check_condition(condition)
    if model.thinking_class == "non_reasoning":
        return None
    return model.no_thinking_reasoning if condition == "no_thinking" else model.thinking_reasoning


def forecast_condition_suffix(model: ModelSpec, condition: str) -> str:
    """The text appended to the forecast prompt for ``model`` in ``condition``.

    Empty for High-confidence hybrids (identical prompt across conditions; API
    parameter does all the work). For non-reasoning models the sec 4
    step-by-step instruction is added in the *thinking* condition. For the
    RL-trained-reasoner hybrids (Med-confidence; §15 RL-reasoner amendment)
    the *no-thinking* condition is reinforced with NO_THINKING_INSTRUCTION
    alongside the ``reasoning:{max_tokens:1024}`` API cap.

    Each suffix has a leading space so it joins onto the preceding sentence.
    """
    _check_condition(condition)
    if model.thinking_class == "non_reasoning" and condition == "thinking":
        return " " + THINKING_INSTRUCTION
    if model.openrouter_id in RL_TRAINED_REASONERS and condition == "no_thinking":
        return " " + NO_THINKING_INSTRUCTION
    return ""


# Backward-compat alias (the old name; new code should use forecast_condition_suffix).
forecast_thinking_suffix = forecast_condition_suffix
