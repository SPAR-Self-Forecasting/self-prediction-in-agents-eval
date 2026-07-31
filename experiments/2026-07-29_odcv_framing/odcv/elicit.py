"""Elicitation engine (spec §3 steps 4 + 5).

Every sample is one independent API call with a fresh context. Checkpoint per
call: each finished record is appended to ``results/predictions.jsonl``
immediately (fsync'd), and a restart skips every (model, scenario, condition,
sample_idx) key already on disk -- parse failures and mid-run fixes are
recoverable with zero new paid calls (C5).

Full plan (rev 2): 358 cells x {self, simulated_cross} x 5 samples = 3,580
calls, plus genuine-cross 40 scenarios x 5 samples = 200 scenario-level calls
(D28 -- the anonymous prompt is byte-identical per scenario, so per-cell calls
would be duplicate spend).

Smoke plan (step 4): 5 severity-spanning scenarios x {glm-5.1, grok-4.20} x
both conditions x 1 sample = 20, + 5 genuine-cross + 5 gpt-oss-20b probe.

An API error after the client's retries is recorded as a broken sample with
``stage: api_error`` -- flagged missing data, never a fake datapoint (C3).
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.llm_client import LLMClient, LLMError

from . import config, extraction, fidelity_gate, groundtruth, prompts

PREDICTIONS_PATH = config.RESULTS_DIR / "predictions.jsonl"
SMOKE_MODELS = ("z-ai/glm-5.1", "x-ai/grok-4.20")
PROBE_MODEL = "openai/gpt-oss-20b"
MAX_WORKERS = 8


# --------------------------------------------------------------------------
# Call plans
# --------------------------------------------------------------------------
def smoke_scenarios(n: int = 5) -> list[str]:
    """Deterministic severity-spanning pick: sort the 40 scenarios by mean
    ground-truth severity across the 9 executors, take evenly spaced picks
    (min ... max). Covers one item per severity case class (C7)."""
    gt = groundtruth.load_ground_truth(config.VARIANT)
    means = {}
    for scenario in groundtruth.scenarios():
        vals = [gt[m][scenario] for m in config.MODEL_IDS if gt[m][scenario] is not None]
        means[scenario] = sum(vals) / len(vals)
    ranked = sorted(means, key=lambda s: (means[s], s))
    idx = [round(i * (len(ranked) - 1) / (n - 1)) for i in range(n)]
    return [ranked[i] for i in idx]


def build_call_plan(mode: str) -> list[dict]:
    """The complete, deterministic list of intended samples for ``mode``."""
    gt = groundtruth.load_ground_truth(config.VARIANT)
    plan: list[dict] = []

    def add(model_id: str, scenario: str, condition: str, sample_idx: int, reasoning):
        plan.append(
            {
                "model": model_id,
                "scenario": scenario,
                "condition": condition,
                "sample_idx": sample_idx,
                "reasoning": reasoning,
            }
        )

    if mode == "full":
        for spec in config.MODELS:
            for scenario in groundtruth.scenarios():
                if gt[spec.openrouter_id][scenario] is None:
                    continue
                for condition in ("self", "simulated_cross"):
                    for k in range(config.N_SAMPLES):
                        add(spec.openrouter_id, scenario, condition, k, spec.reasoning)
        for scenario in groundtruth.scenarios():
            for k in range(config.GENUINE_CROSS_SAMPLES):
                add(config.CROSS_PREDICTOR, scenario, "genuine_cross", k, None)
    elif mode == "smoke":
        chosen = smoke_scenarios()
        for model_id in SMOKE_MODELS:
            spec = config.model_by_id(model_id)
            for scenario in chosen:
                for condition in ("self", "simulated_cross"):
                    add(model_id, scenario, condition, 0, spec.reasoning)
        for scenario in chosen:
            add(config.CROSS_PREDICTOR, scenario, "genuine_cross", 0, None)
        for scenario in chosen:  # JSON-reliability probe (spec §3 step 4 / OQ3)
            add(PROBE_MODEL, scenario, "self", 0, config.model_by_id(PROBE_MODEL).reasoning)
    else:
        raise ValueError(f"mode must be 'full' or 'smoke', got {mode!r}")
    return plan


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------
def _key(record: dict) -> tuple:
    return (record["model"], record["scenario"], record["condition"], record["sample_idx"])


def load_done_keys(path=PREDICTIONS_PATH) -> set[tuple]:
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(_key(json.loads(line)))
    return done


# --------------------------------------------------------------------------
# One call
# --------------------------------------------------------------------------
def run_one(call: dict, summaries: dict[str, str], client: LLMClient) -> dict:
    prompt = prompts.render_prompt(call["scenario"], call["condition"], summaries[call["scenario"]])
    record = dict(call)
    record["ts"] = time.time()
    try:
        resp = client.chat(
            call["model"],
            [{"role": "user", "content": prompt}],
            reasoning=call["reasoning"],
            temperature=config.TEMPERATURE,
        )
    except LLMError as exc:
        record.update(
            raw_response=None, reasoning_content=None, severity=None, action=None,
            broken=True, stage="api_error", error=str(exc), provider=None,
            prompt_tokens=None, completion_tokens=None, cost=0.0,
        )
        return record

    result = extraction.extract_prediction(resp.content, llm_client=client)
    message = (resp.raw.get("choices") or [{}])[0].get("message", {})
    record.update(
        raw_response=resp.content,
        reasoning_content=message.get("reasoning"),
        severity=result.severity,
        action=result.action,
        broken=result.broken,
        stage=result.stage,
        error=None,
        provider=resp.raw.get("provider"),  # D34: serving-provider drift audit
        prompt_tokens=resp.usage.get("prompt_tokens"),
        completion_tokens=resp.usage.get("completion_tokens"),
        cost=resp.cost,
    )
    return record


# --------------------------------------------------------------------------
# The run loop
# --------------------------------------------------------------------------
def run(mode: str, *, client: LLMClient | None = None, summaries: dict | None = None,
        max_workers: int = MAX_WORKERS, path=PREDICTIONS_PATH) -> dict:
    summaries = summaries or fidelity_gate.load_summaries()
    # the gate is free -- re-assert it before ANY paid call (ladder rule)
    fidelity_gate.run_gate(summaries)

    client = client or LLMClient()
    plan = build_call_plan(mode)
    done = load_done_keys(path)
    todo = [c for c in plan if _key(c) not in done]
    print(f"{mode}: {len(plan)} planned, {len(done)} already on disk, {len(todo)} to run")

    path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    n_ok = n_broken = 0

    with open(path, "a", encoding="utf-8") as out:
        def finish(record: dict) -> bool:
            nonlocal n_ok, n_broken
            with write_lock:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                if record["broken"]:
                    n_broken += 1
                else:
                    n_ok += 1
            return not record["broken"]

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(run_one, c, summaries, client) for c in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                finish(fut.result())
                if i % 50 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} done ({n_broken} broken) "
                          f"cost so far ${client.accounting()['total_cost_usd']:.2f}")

    summary = {
        "mode": mode, "planned": len(plan), "ran": len(todo),
        "ok": n_ok, "broken": n_broken, "accounting": client.accounting(),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["smoke", "full"])
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = ap.parse_args()
    run(args.mode, max_workers=args.workers)


if __name__ == "__main__":
    main()
