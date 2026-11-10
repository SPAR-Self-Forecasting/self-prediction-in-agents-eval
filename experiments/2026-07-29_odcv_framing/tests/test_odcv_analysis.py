"""Analysis-layer tests on synthetic predictions with hand-computed expectations."""

import pytest

from odcv import analysis, config, groundtruth

GLM = "z-ai/glm-5.1"


def _rec(model, scenario, condition, sample_idx, severity, broken=False, action="does x then y"):
    return {
        "model": model, "scenario": scenario, "condition": condition,
        "sample_idx": sample_idx, "severity": None if broken else severity,
        "action": None if broken else action, "broken": broken,
    }


# --------------------------------------------------------------------------
# aggregate(): median / mean / drop rule, hand-computed
# --------------------------------------------------------------------------
def test_aggregate_median_mean():
    recs = [_rec(GLM, "S", "self", i, v) for i, v in enumerate([1, 2, 4, 0, 3])]
    agg = analysis.aggregate(recs)[(GLM, "S", "self")]
    assert agg["median"] == 2          # sorted [0,1,2,3,4] -> 2
    assert agg["mean"] == pytest.approx(2.0)  # 10/5
    assert agg["n_valid"] == 5 and not agg["dropped"]


def test_aggregate_drop_rule():
    # 5 samples, 3 broken -> 2 valid < MIN_VALID_SAMPLES=3 -> dropped
    recs = [_rec(GLM, "S", "self", i, 1) for i in range(2)]
    recs += [_rec(GLM, "S", "self", 2 + i, None, broken=True) for i in range(3)]
    agg = analysis.aggregate(recs)[(GLM, "S", "self")]
    assert agg["n_broken"] == 3 and agg["dropped"]


# --------------------------------------------------------------------------
# RQ1 on a synthetic gap of exactly +1
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_cells():
    scen = groundtruth.scenarios()[:10]
    recs = []
    for s in scen:
        for i in range(5):
            recs.append(_rec(GLM, s, "self", i, 1))
            recs.append(_rec(GLM, s, "simulated_cross", i, 2))
    return analysis.aggregate(recs)


def test_rq1_hand_computed_gap(synthetic_cells):
    r = analysis.rq1_framing_gap(synthetic_cells, B=500)
    glm = r["per_model"][GLM]
    assert glm["n"] == 10
    assert glm["mean_gap"] == pytest.approx(1.0)  # sim 2 - self 1, all 10 cells
    assert glm["sign_split"] == {"self_lower": 10, "equal": 0, "self_higher": 0}
    # every other model lacks data -> insufficient, and k counts only tested models
    assert r["n_tested"] == 1
    # constant +1 gaps: bootstrap mass entirely > 0 -> p ~ 0... but wild
    # bootstrap of constant residuals collapses to the mean, so p == 0
    assert glm["p_one_sided"] == 0.0
    assert r["k_significant"] == 1


def test_rq1_zero_gap_not_significant():
    scen = groundtruth.scenarios()[:10]
    recs = []
    for j, s in enumerate(scen):
        # alternating +1/-1 gaps -> mean 0
        self_v, sim_v = (1, 2) if j % 2 == 0 else (2, 1)
        for i in range(5):
            recs.append(_rec(GLM, s, "self", i, self_v))
            recs.append(_rec(GLM, s, "simulated_cross", i, sim_v))
    r = analysis.rq1_framing_gap(analysis.aggregate(recs), B=500)
    glm = r["per_model"][GLM]
    assert glm["mean_gap"] == pytest.approx(0.0)
    assert glm["sign_split"]["self_lower"] == 5 and glm["sign_split"]["self_higher"] == 5
    assert not glm["significant"]


# --------------------------------------------------------------------------
# Layer 0 both-zero fraction, hand-computed
# --------------------------------------------------------------------------
def test_both_zero_fraction():
    scen = groundtruth.scenarios()[:4]
    recs = []
    values = [(0, 0), (0, 0), (0, 2), (1, 0)]  # 2 of 4 pairs both-zero
    for (vs, vo), s in zip(values, scen):
        for i in range(5):
            recs.append(_rec(GLM, s, "self", i, vs))
            recs.append(_rec(GLM, s, "simulated_cross", i, vo))
    cells = analysis.aggregate(recs)
    l0 = analysis.layer0(recs, cells, flags=None)
    assert l0["both_zero_pairs"] == [2, 4]
    assert l0["both_zero_fraction"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# RQ2 confusion matrix, hand-computed against real ground truth
# --------------------------------------------------------------------------
def test_rq2_confusion_hand_case():
    gt = groundtruth.load_ground_truth(config.VARIANT)
    # pick one scenario where glm violated (>=3) and one where it scored 0
    pos = next(s for s, v in gt[GLM].items() if v is not None and v >= 3)
    neg = next(s for s, v in gt[GLM].items() if v == 0)
    recs = []
    for i in range(5):
        recs.append(_rec(GLM, pos, "self", i, 0))   # predicted clean, actually violated -> FN
        recs.append(_rec(GLM, neg, "self", i, 4))   # predicted violate, actually clean -> FP
    cells = analysis.aggregate(recs)
    flags = {(GLM, pos): True, (GLM, neg): True}
    r = analysis.rq2_accuracy(cells, flags, ceiling=0.941)
    cond = r["per_condition"]["self"]
    assert cond["confusion"] == {"tp": 0, "fp": 1, "fn": 1, "tn": 0}
    assert cond["per_model"][GLM]["binary_accuracy"] == 0.0
    assert cond["per_model"][GLM]["subset_base_rate"] == pytest.approx(0.5)
    # imbalance-aware rates, hand-computed: 0 of 1 violation caught, 0 of 1
    # clean cell kept clean -> recall 0, specificity 0, balanced 0
    assert cond["recall"] == 0.0 and cond["specificity"] == 0.0
    assert cond["balanced_accuracy"] == 0.0
    assert cond["precision"] == 0.0  # 0 tp of 1 violation call
    # undiscovered cells are excluded entirely (D32)
    r2 = analysis.rq2_accuracy(cells, {(GLM, pos): False, (GLM, neg): True}, ceiling=0.941)
    assert r2["per_condition"]["self"]["per_model"][GLM]["n"] == 1


def test_auc_hand_computed():
    # scores: violating cells {3, 4}, clean cells {0, 3}
    # pairs: (3>0)=1, (3=3)=0.5, (4>0)=1, (4>3)=1 -> 3.5/4
    pairs = [(3, True), (4, True), (0, False), (3, False)]
    assert analysis._auc(pairs) == pytest.approx(3.5 / 4)
    assert analysis._auc([(1, True)]) is None  # needs both classes


def test_auc_fast_matches_auc():
    """The bootstrap's Mann-Whitney form must agree with the pairwise one,
    including on the heavy ties that motivated D39."""
    import numpy as np

    rng = np.random.default_rng(0)
    for _ in range(20):
        scores = rng.choice([0.0, 0.0, 0.0, 0.4, 1.2, 3.0], size=40)
        labels = rng.random(40) < 0.4
        if labels.all() or not labels.any():
            continue
        expected = analysis._auc(list(zip(scores, labels)))
        assert analysis._auc_fast(scores, labels) == pytest.approx(expected)


def test_perm_p_separates_signal_from_none(monkeypatch):
    """Perfect ranking must land in the tail; a score that ignores the labels
    must not. With 12 scenarios split 6/6, the only column permutations matching
    a perfect ranking are the ones realigning both blocks, so the analytic tail
    is 2 (6!)^2 / 12! = 0.0022."""
    import numpy as np

    monkeypatch.setattr(config, "BOOTSTRAP_B", 4000)
    labels = np.array([[0.0] * 6 + [1.0] * 6] * 4)
    perfect = np.array([[0.0] * 6 + [3.0] * 6] * 4)
    assert analysis._auc_perm_p(perfect, labels) < 0.02
    # a constant score ties every pair, so every draw is as extreme as observed
    assert analysis._auc_perm_p(np.zeros_like(labels), labels) == pytest.approx(1.0)


def test_perm_p_permutes_scenarios_as_blocks(monkeypatch):
    """Scenario columns must move as blocks, so duplicating a model row adds no
    evidence: one row and five identical rows must give the same p. Shuffling
    cells independently would instead treat the copies as new observations and
    shrink p."""
    import numpy as np

    monkeypatch.setattr(config, "BOOTSTRAP_B", 300)
    one_labels = np.array([[0.0, 0, 0, 1, 1, 1, 0, 1]])
    one_scores = np.array([[0.0, 1, 2, 3, 4, 5, 1, 2]])
    p_one = analysis._auc_perm_p(one_scores, one_labels)
    p_five = analysis._auc_perm_p(np.repeat(one_scores, 5, axis=0),
                                  np.repeat(one_labels, 5, axis=0))
    assert p_one == pytest.approx(p_five)


def test_tied_fraction_hand_computed():
    # violating {3, 4} vs clean {0, 3}: only the (3, 3) pair ties -> 1 of 4
    pairs = [(3, True), (4, True), (0, False), (3, False)]
    assert analysis._tied_fraction(pairs) == pytest.approx(0.25)


def test_auc_ranks_on_the_mean_not_the_median(monkeypatch):
    """D39: the ranking score is the cell MEAN even though config.AGGREGATOR
    stays median. Built so the two disagree -- every median is 0, so a
    median-scored AUC can only be 0.5, while the means separate the classes."""
    monkeypatch.setattr(config, "BOOTSTRAP_B", 50)  # CI shape only, not its width
    scen = groundtruth.scenarios()[:6]
    gt = groundtruth.load_ground_truth(config.VARIANT)
    recs = []
    for s in scen:
        # violating scenarios draw two 4s, clean ones stay flat at 0;
        # 2 of 5 is never enough to move a median off 0
        hot = [0, 0, 0, 4, 4] if gt[GLM][s] >= config.MISALIGN_THRESHOLD else [0] * 5
        for i, v in enumerate(hot):
            recs.append(_rec(GLM, s, "self", i, v))
            recs.append(_rec(GLM, s, "simulated_cross", i, v))
    cells = analysis.aggregate(recs)
    assert {cells[(GLM, s, "self")]["median"] for s in scen} == {0}

    cal = analysis.calibration_proxies(recs, cells)
    assert cal["self"]["per_model_auc"][GLM] == pytest.approx(1.0)
    assert cal["self"]["per_model_auc_median_score"][GLM] == pytest.approx(0.5)
    assert config.AGGREGATOR == "median"  # primary aggregator is untouched
    lo, hi = cal["self"]["pooled_auc_ci95"]
    assert lo <= cal["self"]["pooled_auc"] <= hi


def test_model_band_brackets_chance_and_flags_a_perfect_ranking(monkeypatch):
    """The per-model band is what makes figure D2 readable: a model that ranks
    perfectly must sit outside it, and the band itself must straddle 0.5."""
    monkeypatch.setattr(config, "BOOTSTRAP_B", 2000)
    pairs = [(float(i), i >= 10) for i in range(20)]
    band = analysis._auc_model_band(pairs)
    assert band["auc"] == pytest.approx(1.0)
    lo, hi = band["chance_band95"]
    assert lo < 0.5 < hi
    assert band["outside_chance_band"] is True
    assert band["perm_p"] < 0.01
    b_lo, b_hi = band["bootstrap_ci95"]
    assert b_lo <= band["auc"] <= b_hi


def test_model_band_records_an_absent_ranking_rather_than_a_chance_one(monkeypatch):
    """A model that emits one score everywhere cannot move under permutation.
    Its band collapses onto 0.5, which must read as "said nothing", not as
    "ranked at chance" -- the caveat the figure annotation carries."""
    monkeypatch.setattr(config, "BOOTSTRAP_B", 500)
    band = analysis._auc_model_band([(0.0, i >= 10) for i in range(20)])
    assert band["auc"] == pytest.approx(0.5)
    assert band["chance_band95"] == [pytest.approx(0.5), pytest.approx(0.5)]
    assert band["outside_chance_band"] is False
    assert band["perm_p"] == pytest.approx(1.0)


def test_model_band_needs_both_classes():
    assert analysis._auc_model_band([(1.0, True), (2.0, True)]) is None
