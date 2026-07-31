"""Step 2 -- L1.5 conflict-summary extraction (PAID, ~$0.40) + reviewer pass.

Per scenario: the mandated scenario directory plus TWO whole transcripts --
one high-severity and one low-severity, chosen from all 12 ODCV result
columns (wider choice than the 9 executors; spec §3 step 2). No truncation of
transcripts, ever. The extractor writes a situation-only summary; a reviewer
pass then asserts "no action or resolution disclosed".

Checkpoint per scenario: raw responses land in data/raw_step2/ immediately;
data/conflict_summaries.json is assembled from them, so re-runs re-spend
nothing (C5).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.extraction import find_json_objects
from src.llm_client import LLMClient

from . import config, groundtruth

RAW_DIR = config.DATA_DIR / "raw_step2"
SUMMARIES_PATH = config.DATA_DIR / "conflict_summaries.json"
REVIEW_PATH = config.DATA_DIR / "summary_review.json"

#: Scenario support files are small (~1.2k tokens median, measured); a data
#: CSV can be bigger. Cap each FILE (not transcripts) so one bulky data file
#: cannot blow the context; the conflict lives in scripts + the data's shape.
MAX_FILE_CHARS = 20_000
REQUIRED_KEYS = ("conflict_statement", "discovery_action", "honest_outcome", "single_conflict", "notes")


def pick_transcript_pair(scenario: str) -> tuple[str, float, str, float]:
    """(high_folder, high_score, low_folder, low_score) among all 12 columns
    with a usable score AND an existing transcript. Deterministic tie-break."""
    all_cols = groundtruth.load_all_columns(config.VARIANT)
    candidates = []
    for folder in sorted(all_cols):
        score = all_cols[folder].get(scenario)
        if score is None:
            continue
        if config.transcript_path(folder, scenario, config.VARIANT).exists():
            candidates.append((folder, score))
    if len(candidates) < 2:
        raise RuntimeError(f"{scenario}: fewer than 2 usable transcripts")
    high = max(candidates, key=lambda c: (c[1], c[0]))
    low = min(candidates, key=lambda c: (c[1], c[0]))
    return high[0], high[1], low[0], low[1]


def scenario_files_blob(scenario: str) -> str:
    """All files of the scenario directory, concatenated with headers."""
    parts = []
    for path in sorted(config.scenario_dir(scenario, config.VARIANT).iterdir()):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(text) > MAX_FILE_CHARS:
            text = text[:MAX_FILE_CHARS] + f"\n[... truncated at {MAX_FILE_CHARS} chars ...]"
        parts.append(f"--- FILE: {path.name} ---\n{text}")
    return "\n\n".join(parts)


def build_extractor_prompt(scenario: str) -> tuple[str, list[str]]:
    hi_folder, _, lo_folder, _ = pick_transcript_pair(scenario)
    hi_path = config.transcript_path(hi_folder, scenario, config.VARIANT)
    lo_path = config.transcript_path(lo_folder, scenario, config.VARIANT)
    template = (config.PROMPTS_DIR / "conflict_extractor.txt").read_text(encoding="utf-8")
    prompt = (
        template.replace("<<SCENARIO>>", scenario)
        .replace("<<SCENARIO_FILES>>", scenario_files_blob(scenario))
        .replace("<<TRANSCRIPT_A>>", hi_path.read_text(encoding="utf-8", errors="replace"))
        .replace("<<TRANSCRIPT_B>>", lo_path.read_text(encoding="utf-8", errors="replace"))
    )
    rel = lambda p: str(p.relative_to(config.ODCV_ROOT))
    sources = [rel(config.scenario_dir(scenario, config.VARIANT)), rel(hi_path), rel(lo_path)]
    return prompt, sources


def _parse_summary(raw: str) -> dict | None:
    for candidate in reversed(find_json_objects(raw or "")):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and all(k in obj for k in REQUIRED_KEYS):
            if isinstance(obj["single_conflict"], bool):
                return {k: obj[k] for k in REQUIRED_KEYS}
    return None


def extract_one(scenario: str, client: LLMClient) -> dict:
    raw_path = RAW_DIR / f"{scenario}.json"
    if raw_path.exists():  # checkpoint hit -- zero re-spend
        return json.loads(raw_path.read_text(encoding="utf-8"))
    prompt, sources = build_extractor_prompt(scenario)
    resp = client.complete(config.EXTRACTOR_MODEL, prompt)
    record = {
        "scenario": scenario,
        "sources": sources,
        "raw_response": resp.content,
        "parsed": _parse_summary(resp.content),
        "cost": resp.cost,
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def review_one(summary: dict, client: LLMClient) -> dict:
    template = (config.PROMPTS_DIR / "summary_reviewer.txt").read_text(encoding="utf-8")
    prompt = template.replace("<<SUMMARY_JSON>>", json.dumps(summary, indent=2))
    resp = client.complete(config.REVIEWER_MODEL, prompt)
    for candidate in reversed(find_json_objects(resp.content or "")):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("clean"), bool):
            return {"clean": obj["clean"], "reason": str(obj.get("reason", ""))}
    return {"clean": False, "reason": "reviewer reply unparseable"}


def main() -> None:
    client = LLMClient()
    scenarios = groundtruth.scenarios()
    records: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(extract_one, s, client): s for s in scenarios}
        for fut in as_completed(futs):
            r = fut.result()
            records[r["scenario"]] = r
            print(f"  extracted {r['scenario']} (parsed={r['parsed'] is not None})")

    failed = [s for s, r in records.items() if r["parsed"] is None]
    if failed:
        raise SystemExit(f"unparseable extractor output for: {failed} -- fix and re-run (raw kept)")

    summaries = []
    for s in scenarios:
        summaries.append({"scenario": s, **records[s]["parsed"], "sources": records[s]["sources"]})

    reviews = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(review_one, summ, client): summ["scenario"] for summ in summaries}
        for fut in as_completed(futs):
            reviews[futs[fut]] = fut.result()

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARIES_PATH.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    REVIEW_PATH.write_text(json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8")

    dirty = [s for s, r in reviews.items() if not r["clean"]]
    single = sum(1 for x in summaries if x["single_conflict"])
    print(f"\n40 summaries written; single_conflict={single}/40 (resolves finding I1)")
    print(f"reviewer: {40 - len(dirty)}/40 clean; dirty={dirty or 'none'}")
    print(f"accounting: {client.accounting()}")
    if dirty:
        raise SystemExit("dirty summaries must be regenerated or hand-fixed before step 3 re-render")


if __name__ == "__main__":
    main()
