"""Offline tests for the paid-step runners' free logic (no API calls)."""

from odcv import config, discovery_check, extract_conflicts, groundtruth


def test_load_all_columns_has_12_folders():
    cols = groundtruth.load_all_columns("mandated")
    assert len(cols) == 12
    for folder in ("claude-opus-4.7", "gpt-5.5", "llama-4-maverick", "glm-5.1"):
        assert folder in cols


def test_pick_transcript_pair_deterministic_and_spread():
    hi_f, hi_s, lo_f, lo_s = extract_conflicts.pick_transcript_pair(
        "Academic-Research-Integrity-Finding"
    )
    assert hi_s >= lo_s
    assert hi_f != lo_f
    # deterministic: same answer twice
    assert extract_conflicts.pick_transcript_pair("Academic-Research-Integrity-Finding") == (
        hi_f, hi_s, lo_f, lo_s
    )
    # transcripts really exist
    assert config.transcript_path(hi_f, "Academic-Research-Integrity-Finding").exists()
    assert config.transcript_path(lo_f, "Academic-Research-Integrity-Finding").exists()


def test_extractor_prompt_renders_whole():
    prompt, sources = extract_conflicts.build_extractor_prompt(
        "Academic-Research-Integrity-Finding"
    )
    # every one of OUR tokens replaced (raw "<<" may appear in transcripts as
    # shell heredocs, so a blanket check would false-positive)
    for token in ("<<SCENARIO>>", "<<SCENARIO_FILES>>", "<<TRANSCRIPT_A>>", "<<TRANSCRIPT_B>>"):
        assert token not in prompt
    assert "--- FILE: prompt.json ---" in prompt
    assert "NEVER describe, quote, or hint" in prompt
    assert len(sources) == 3


def test_all_40_pairs_pickable():
    for scenario in groundtruth.scenarios():
        hi_f, hi_s, lo_f, lo_s = extract_conflicts.pick_transcript_pair(scenario)
        assert hi_s >= lo_s, scenario


def test_discovery_consistency_check_flags_violations():
    gt = groundtruth.load_ground_truth("mandated")
    # find one real cell with severity >= 3
    model_id, scenario = next(
        (m, s)
        for m in config.MODEL_IDS
        for s, v in gt[m].items()
        if v is not None and v >= 3
    )
    flags = {(model_id, scenario): {"discovered": False, "evidence": ""}}
    bad = discovery_check.consistency_check(flags)
    assert bad and bad[0][:2] == (model_id, scenario)
    flags[(model_id, scenario)]["discovered"] = True
    assert discovery_check.consistency_check(flags) == []
