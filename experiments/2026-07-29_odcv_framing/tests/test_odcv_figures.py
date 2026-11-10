"""Figures smoke test: every figure renders from synthetic analysis output."""

from odcv import analysis, config, figures, groundtruth


def _rec(model, scenario, condition, i, sev):
    return {"model": model, "scenario": scenario, "condition": condition,
            "sample_idx": i, "severity": sev, "action": "does x", "broken": False}


def test_all_figures_render(tmp_path):
    gt = groundtruth.load_ground_truth(config.VARIANT)
    recs = []
    for spec in config.MODELS:
        m = spec.openrouter_id
        for j, s in enumerate(groundtruth.scenarios()[:12]):
            if gt[m][s] is None:
                continue
            for i in range(5):
                recs.append(_rec(m, s, "self", i, j % 3))
                recs.append(_rec(m, s, "simulated_cross", i, (j % 3) + 1))

    cells = analysis.aggregate(recs)
    flags = {(m.openrouter_id, s): True for m in config.MODELS for s in groundtruth.scenarios()}
    result = {
        "layer0": analysis.layer0(recs, cells, flags),
        "rq1": analysis.rq1_framing_gap(cells, B=200),
        "rq2": analysis.rq2_accuracy(cells, flags, ceiling=0.941),
        "exploratory": analysis.exploratory(cells),
        "calibration_proxies": analysis.calibration_proxies(recs, cells),
    }
    made = figures.render_all(result, cells, out_dir=tmp_path)
    assert len(made) == 10  # A1-A3, B1-B3, C1-C2 + exploratory D1-D2
    for p in made:
        assert p.exists() and p.stat().st_size > 5_000, p.name
