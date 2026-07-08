"""Tests for config.py -- model list, dimension battery, reasoning-control map."""

import config


# --------------------------------------------------------------------------
# Constants (sec 6)
# --------------------------------------------------------------------------
def test_scalar_constants():
    assert config.SEED == 42
    assert config.STEP_LIMIT == 250
    assert config.N_ROLLOUTS == 8
    assert config.N_TASKS == 30
    assert config.CONDITIONS == ("no_thinking", "thinking")


def test_difficulty_split_sums_to_n_tasks():
    assert sum(config.DIFFICULTY_SPLIT.values()) == config.N_TASKS
    assert config.DIFFICULTY_SPLIT == {"easy": 10, "medium": 10, "hard": 10}


def test_difficulty_map_buckets():
    # SWE-bench Verified ships a human "difficulty" annotation.
    assert config.DIFFICULTY_MAP["<15 min fix"] == "easy"
    assert config.DIFFICULTY_MAP["15 min - 1 hour"] == "medium"
    assert config.DIFFICULTY_MAP["1-4 hours"] == "hard"
    assert config.DIFFICULTY_MAP[">4 hours"] == "hard"


# --------------------------------------------------------------------------
# Model list (sec 7)
# --------------------------------------------------------------------------
def test_twelve_models():
    assert len(config.MODELS) == 12


def test_model_ids_and_slugs_unique():
    assert len({m.openrouter_id for m in config.MODELS}) == 12
    assert len({m.slug for m in config.MODELS}) == 12
    assert len({m.name for m in config.MODELS}) == 12


def test_five_families_with_expected_counts():
    fams = config.families()
    assert set(fams) == {"Qwen", "Google", "DeepSeek", "Zhipu", "Moonshot"}
    counts = {f: len(v) for f, v in fams.items()}
    assert counts == {"Qwen": 5, "Google": 2, "DeepSeek": 1, "Zhipu": 3, "Moonshot": 1}


def test_models_in_ascending_capability_order():
    est = [m.est_swebench for m in config.MODELS]
    assert est == sorted(est)


def test_thinking_classes_partition_10_hybrid_2_nonreasoning():
    hybrid = [m for m in config.MODELS if m.thinking_class == "hybrid"]
    non = [m for m in config.MODELS if m.thinking_class == "non_reasoning"]
    assert len(hybrid) == 10
    assert len(non) == 2
    assert {m.name for m in non} == {
        "Qwen3-Next-80B-A3B-Instruct",
        "Qwen3-Coder-480B-A35B",
    }


def test_model_lookup():
    m = config.model_by_id("qwen/qwen3-8b")
    assert m.name == "Qwen3-8B"
    assert m.family == "Qwen"


# --------------------------------------------------------------------------
# Reasoning-control map (sec 10.4)
# --------------------------------------------------------------------------
def test_hybrid_reasoning_toggle():
    m = config.model_by_id("qwen/qwen3-8b")  # High-confidence hybrid default-on
    assert config.forecast_reasoning(m, "no_thinking") == {"enabled": False}
    assert config.forecast_reasoning(m, "thinking") == {"enabled": True}
    # High-confidence hybrids never get a prompt suffix -- API param does it all.
    assert config.forecast_condition_suffix(m, "no_thinking") == ""
    assert config.forecast_condition_suffix(m, "thinking") == ""


def test_gemini_25_default_off_omits_reasoning_for_no_thinking():
    m = config.model_by_id("google/gemini-2.5-flash-lite")
    assert config.forecast_reasoning(m, "no_thinking") is None
    assert config.forecast_reasoning(m, "thinking") is not None


def test_rl_trained_reasoner_no_thinking_caps_tokens_and_adds_prompt():
    # Med-confidence (sec 10.4): API "off" unreliable -> max_tokens=1024 + prompt
    for model_id in config.RL_TRAINED_REASONERS:
        m = config.model_by_id(model_id)
        # API: capped at 1024 reasoning tokens in no_thinking; thinking unchanged
        assert config.forecast_reasoning(m, "no_thinking") == {"max_tokens": 1024}
        # Prompt: NO_THINKING_INSTRUCTION applied (with leading space) in no_thinking
        suffix_nt = config.forecast_condition_suffix(m, "no_thinking")
        assert suffix_nt.strip() == config.NO_THINKING_INSTRUCTION.strip()
        assert suffix_nt.startswith(" ")
        # Prompt: no suffix in thinking
        assert config.forecast_condition_suffix(m, "thinking") == ""


def test_non_reasoning_model_uses_prompt_suffix_not_api_param():
    m = config.model_by_id("qwen/qwen3-coder")
    assert config.forecast_reasoning(m, "no_thinking") is None
    assert config.forecast_reasoning(m, "thinking") is None
    assert config.forecast_condition_suffix(m, "no_thinking") == ""
    suffix = config.forecast_condition_suffix(m, "thinking")
    assert suffix.strip() == config.THINKING_INSTRUCTION.strip()
    assert suffix.startswith(" ")  # joins onto the preceding sentence


# --------------------------------------------------------------------------
# Dimension battery (sec 2 / sec 3.1)
# --------------------------------------------------------------------------
def test_ten_dimensions_canonical_order():
    # §15.1 A1: p_hit_step_limit dropped; A6: 2 behavioral dims added; A2 canonical numbering.
    names = [d.name for d in config.DIMENSIONS]
    assert names == [
        "p_solve", "p_first_attempt_correct",
        "documentation", "defensive_code", "express_uncertainty",
        "avg_tokens_step", "total_expected_steps", "explore_edit_ratio",
        "mean_action_block_size", "n_constraint_violations",
    ]


def test_five_binary_five_continuous():
    binary = [d for d in config.DIMENSIONS if d.is_binary]
    cont = [d for d in config.DIMENSIONS if d.is_continuous]
    assert len(binary) == 5 and len(cont) == 5


def test_judge_and_capability_dimension_groups():
    # Judge-based dims = 3-5 under the §15.1 A2 canonical numbering.
    assert config.JUDGE_DIMENSIONS == ("documentation", "defensive_code", "express_uncertainty")
    # Capability dims are now 2 (p_hit_step_limit dropped).
    assert config.CAPABILITY_DIMENSIONS == ("p_solve", "p_first_attempt_correct")
    assert len(config.DIMENSION_NAMES) == 10
