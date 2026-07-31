"""Step 2b -- per-cell conflict-discovery check (PAID, ~$2; spec rev 2, D32).

For each of the 358 usable (model, scenario) cells, read that cell's shipped
transcript and ask: did the conflict (per the step-2 ``conflict_statement``)
become visible to this agent? RQ2/RQ3 later score only ``discovered`` cells --
a ground-truth 0 with no discovery says nothing about disposition.

Checkpoint per cell into data/discovery_flags.jsonl (append + resume).

Mechanical consistency check (free, spec acceptance 2): violation implies
discovery -- any severity >= 3 cell flagged undiscovered is an extraction
error and fails the run.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.extraction import find_json_objects
from src.llm_client import LLMClient

from . import config, groundtruth

FLAGS_PATH = config.DATA_DIR / "discovery_flags.jsonl"


def _load_summaries() -> dict[str, dict]:
    records = json.loads((config.DATA_DIR / "conflict_summaries.json").read_text(encoding="utf-8"))
    return {r["scenario"]: r for r in records}


def build_prompt(summary: dict, transcript: str) -> str:
    template = (config.PROMPTS_DIR / "discovery_check.txt").read_text(encoding="utf-8")
    return (
        template.replace("<<CONFLICT_STATEMENT>>", summary["conflict_statement"])
        .replace("<<DISCOVERY_ACTION>>", summary["discovery_action"])
        .replace("<<TRANSCRIPT>>", transcript)
    )


def _parse_flag(raw: str) -> dict | None:
    for candidate in reversed(find_json_objects(raw or "")):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("discovered"), bool):
            return {"discovered": obj["discovered"], "evidence": str(obj.get("evidence", ""))}
    # Tolerant fallback: checker replies are occasionally malformed JSON
    # (unescaped quotes inside the evidence string). The verdict itself is
    # still recoverable by pattern -- raw responses are always stored (C5).
    m = re.search(r'"discovered"\s*:\s*(true|false)', raw or "")
    if m:
        return {"discovered": m.group(1) == "true", "evidence": "(recovered from malformed reply)"}
    return None


def check_one(model_id: str, scenario: str, summaries: dict, client: LLMClient) -> dict:
    spec = config.model_by_id(model_id)
    transcript = config.transcript_path(spec.odcv_result_folder, scenario, config.VARIANT).read_text(
        encoding="utf-8", errors="replace"
    )
    resp = client.complete(config.DISCOVERY_MODEL, build_prompt(summaries[scenario], transcript))
    parsed = _parse_flag(resp.content)
    return {
        "model": model_id,
        "scenario": scenario,
        "discovered": None if parsed is None else parsed["discovered"],
        "evidence": None if parsed is None else parsed["evidence"],
        "raw_response": resp.content,
        "cost": resp.cost,
    }


def load_flags(path=FLAGS_PATH) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[(r["model"], r["scenario"])] = r
    return out


def consistency_check(flags: dict[tuple, dict]) -> list[tuple]:
    """Cells violating 'severity >= 3 implies discovered'."""
    gt = groundtruth.load_ground_truth(config.VARIANT)
    bad = []
    for (model_id, scenario), r in flags.items():
        score = gt[model_id][scenario]
        if score is not None and score >= config.MISALIGN_THRESHOLD and r["discovered"] is not True:
            bad.append((model_id, scenario, score, r["discovered"]))
    return bad


def per_model_funnel(flags: dict[tuple, dict]) -> dict[str, dict]:
    gt = groundtruth.load_ground_truth(config.VARIANT)
    out = {}
    for spec in config.MODELS:
        cells = [s for s, v in gt[spec.openrouter_id].items() if v is not None]
        discovered = sum(1 for s in cells if flags.get((spec.openrouter_id, s), {}).get("discovered"))
        out[spec.openrouter_id] = {"total": len(cells), "discovered": discovered}
    return out


FINAL_PATH = config.DATA_DIR / "discovery_flags_final.json"


def finalize() -> dict:
    """Produce the flags RQ2/RQ3 actually consume, from stored raws only
    (zero re-spend):

    1. re-parse any unparsed checker reply with the tolerant parser;
    2. apply the mechanical rule: severity >= 3 forces discovered=True --
       a violation is conclusive proof the agent engaged the conflict
       (verified by transcript inspection 2026-07-30: checker false
       negatives, not pre-emptive cheating);
    3. re-run the consistency check, which must now be clean.
    """
    gt = groundtruth.load_ground_truth(config.VARIANT)
    flags = load_flags()
    final: dict[str, dict] = {}
    n_reparse = n_override = 0
    for (model_id, scenario), r in flags.items():
        discovered, source = r["discovered"], "checker"
        if discovered is None:
            parsed = _parse_flag(r["raw_response"])
            if parsed is not None:
                discovered, source = parsed["discovered"], "reparse"
                n_reparse += 1
        score = gt[model_id][scenario]
        if score is not None and score >= config.MISALIGN_THRESHOLD and discovered is not True:
            discovered, source = True, "violation_override"
            n_override += 1
        final[f"{model_id}|{scenario}"] = {"discovered": discovered, "source": source}

    unresolved = [k for k, v in final.items() if v["discovered"] is None]
    if unresolved:
        raise SystemExit(f"still unparseable after tolerant re-parse: {unresolved}")
    check = {
        (k.split("|")[0], k.split("|")[1]): {"discovered": v["discovered"]}
        for k, v in final.items()
    }
    bad = consistency_check(check)
    if bad:
        raise SystemExit(f"consistency STILL violated after finalize: {bad}")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_PATH.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"finalized {len(final)} flags -> {FINAL_PATH} "
          f"(reparsed={n_reparse}, violation_overrides={n_override})")
    return final


def main() -> None:
    summaries = _load_summaries()
    gt = groundtruth.load_ground_truth(config.VARIANT)
    done = load_flags()
    todo = [
        (spec.openrouter_id, scenario)
        for spec in config.MODELS
        for scenario, v in gt[spec.openrouter_id].items()
        if v is not None and (spec.openrouter_id, scenario) not in done
    ]
    print(f"step 2b: {len(done)} on disk, {len(todo)} to run")

    client = LLMClient()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    with open(FLAGS_PATH, "a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=6) as pool:
            futs = [pool.submit(check_one, m, s, summaries, client) for m, s in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                with lock:
                    out.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
                    out.flush()
                if i % 25 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} cost ${client.accounting()['total_cost_usd']:.2f}")

    flags = load_flags()
    unparsed = [(k, v) for k, v in flags.items() if v["discovered"] is None]
    bad = consistency_check(flags)
    funnel = per_model_funnel(flags)
    print(json.dumps(funnel, indent=2))
    print(f"unparsed flags: {len(unparsed)}; consistency violations: {bad or 'none'}")
    if unparsed or bad:
        raise SystemExit("step 2b NOT green -- investigate before RQ2/RQ3 use these flags")


if __name__ == "__main__":
    main()
