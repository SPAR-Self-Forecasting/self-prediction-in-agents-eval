"""Layer-4 qualitative coding pass (spec §3 step 6b layer 4, D26).

LLM-assisted coding of every cell with a >= QUAL_GAP_THRESHOLD self/other
severity difference into the fixed 3-bucket schema. A >= 20-item human
spot-check is REQUIRED before the bucket ratio is a load-bearing claim (V9) --
this module prints the sample to check.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.extraction import find_json_objects
from src.llm_client import LLMClient

from . import analysis, config

BUCKETS = ("same_action_different_label", "different_action", "ambiguous")
OUT_PATH = config.RESULTS_DIR / "qual_coding.json"


def build_prompt(item: dict) -> str:
    template = (config.PROMPTS_DIR / "qual_coder.txt").read_text(encoding="utf-8")
    return (
        template.replace("<<SCENARIO>>", item["scenario"])
        .replace("<<SEV_SELF>>", str(item["severity_self"]))
        .replace("<<SEV_OTHER>>", str(item["severity_simulated_cross"]))
        .replace("<<ACTIONS_SELF>>", "\n".join(f"- {a}" for a in item["actions_self"]))
        .replace("<<ACTIONS_OTHER>>", "\n".join(f"- {a}" for a in item["actions_simulated_cross"]))
    )


def code_one(item: dict, client: LLMClient) -> dict:
    resp = client.complete(config.REVIEWER_MODEL, build_prompt(item))
    bucket, rationale = "ambiguous", "coder reply unparseable"
    for candidate in reversed(find_json_objects(resp.content or "")):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and obj.get("bucket") in BUCKETS:
            bucket, rationale = obj["bucket"], str(obj.get("rationale", ""))
            break
    return {**{k: item[k] for k in ("model", "scenario", "severity_self", "severity_simulated_cross")},
            "bucket": bucket, "rationale": rationale}


def main() -> None:
    records = analysis.load_predictions()
    cells = analysis.aggregate(records)
    items = analysis.qualitative_items(cells)
    print(f"layer-4 items (|gap| >= {config.QUAL_GAP_THRESHOLD}): {len(items)}")
    client = LLMClient()
    coded = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(code_one, it, client) for it in items]
        for fut in as_completed(futs):
            coded.append(fut.result())

    counts = {b: sum(1 for c in coded if c["bucket"] == b) for b in BUCKETS}
    unambiguous = counts["same_action_different_label"] + counts["different_action"]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(
        {"counts": counts,
         "label_ratio": (counts["same_action_different_label"] / unambiguous) if unambiguous else None,
         "items": coded},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"counts: {counts} -> {OUT_PATH}")
    print(f"accounting: {client.accounting()}")
    print("\nHUMAN SPOT-CHECK REQUIRED (V9): review >= 20 of the coded items above "
          "in results/qual_coding.json before citing the ratio.")


if __name__ == "__main__":
    main()
