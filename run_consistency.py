#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai>=2.0.0",
#   "pydantic>=2.8.0",
# ]
# ///

"""Run every case N times and measure run-to-run consistency and label agreement.

`make determinism` replays a single record; this checks all records. Each run calls
the model fresh (there is no response cache), so identical outputs reflect real
determinism of the extraction + policy pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from core import get_openai_client, triage_submission

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "data" / "patients_sample_50.jsonl"))
    parser.add_argument("--report", default=str(ROOT / "data" / "consistency_report.json"))
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def _run_one(task: tuple[int, int, dict, str]) -> dict:
    run_index, record_index, case, model = task
    usage: list[dict] = []
    try:
        output = triage_submission(case["submission"], model=model, client=get_openai_client(),
                                   on_telemetry=lambda t: usage.append(t.as_dict()))
        payload = output.model_dump()
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return {
            "run": run_index,
            "record_index": record_index,
            "decision": output.decision,
            "categories": sorted({issue.category for issue in output.issues}),
            "output_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "error": None,
            "usage": usage,
        }
    except Exception as exc:  # pragma: no cover - network/runtime failure path
        return {"run": run_index, "record_index": record_index, "decision": "ERROR", "categories": [],
                "output_sha256": "ERROR", "error": f"{type(exc).__name__}: {exc}", "usage": usage}


def summarize_usage(results: list[dict], runs: int) -> dict:
    """Token totals and prompt-cache health, overall and per run."""

    def totals(rows: list[dict]) -> dict:
        calls = [u for r in rows for u in r.get("usage", [])]
        input_tokens = sum(u["input_tokens"] for u in calls)
        cached = sum(u["cached_input_tokens"] for u in calls)
        latencies = sorted(u["latency_ms"] for u in calls)
        return {
            "calls": len(calls),
            "input_tokens": input_tokens,
            "cached_input_tokens": cached,
            "uncached_input_tokens": input_tokens - cached,
            "output_tokens": sum(u["output_tokens"] for u in calls),
            "reasoning_tokens": sum(u["reasoning_tokens"] for u in calls),
            "avg_input_tokens": round(input_tokens / len(calls), 1) if calls else 0.0,
            "avg_cached_input_tokens": round(cached / len(calls), 1) if calls else 0.0,
            "avg_output_tokens": round(sum(u["output_tokens"] for u in calls) / len(calls), 1) if calls else 0.0,
            "cache_hit_ratio": round(cached / input_tokens, 4) if input_tokens else 0.0,
            "calls_with_any_cache_hit": sum(u["cached_input_tokens"] > 0 for u in calls),
            "latency_ms_p50": latencies[len(latencies) // 2] if latencies else None,
            "latency_ms_p95": latencies[int(0.95 * (len(latencies) - 1))] if latencies else None,
            "latency_ms_max": latencies[-1] if latencies else None,
        }

    return {"overall": totals(results), "per_run": [totals([r for r in results if r["run"] == run]) for run in range(runs)]}


def main() -> None:
    args = parse_args()
    cases = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    tasks = [(run, index, case, args.model) for run in range(args.runs) for index, case in enumerate(cases)]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_run_one, tasks))

    per_case = []
    for index, case in enumerate(cases):
        rows = [r for r in results if r["record_index"] == index]
        label = case["label"]
        expected = (label["decision"], sorted(set(label["categories"])))
        per_case.append(
            {
                "record_index": index,
                "case_id": case["case_id"],
                "label": {"decision": label["decision"], "categories": expected[1]},
                "decisions": dict(Counter(r["decision"] for r in rows)),
                "category_sets": dict(Counter("|".join(r["categories"]) or "-" for r in rows)),
                "distinct_outputs": len({r["output_sha256"] for r in rows}),
                "label_match_runs": sum((r["decision"], r["categories"]) == expected for r in rows),
                "errors": [r["error"] for r in rows if r["error"]],
            }
        )

    total = len(cases)
    summary = {
        "runs": args.runs,
        "records": total,
        "decision_consistent_cases": sum(len(c["decisions"]) == 1 for c in per_case),
        "category_consistent_cases": sum(len(c["category_sets"]) == 1 for c in per_case),
        "byte_identical_output_cases": sum(c["distinct_outputs"] == 1 for c in per_case),
        "label_match_all_runs_cases": sum(c["label_match_runs"] == args.runs for c in per_case),
        "label_match_rate_pct": round(100 * sum(c["label_match_runs"] for c in per_case) / (total * args.runs), 2),
        "errors": sum(len(c["errors"]) for c in per_case),
    }
    summary["usage"] = summarize_usage(results, args.runs)
    report = {"generated_at": datetime.now(tz=UTC).isoformat(), "model": args.model, "summary": summary, "cases": per_case}
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    for case in per_case:
        if case["distinct_outputs"] != 1 or case["label_match_runs"] != args.runs or case["errors"]:
            print(f"  {case['case_id']}: label={case['label']} decisions={case['decisions']} "
                  f"categories={case['category_sets']} distinct_outputs={case['distinct_outputs']} errors={case['errors'][:1]}")
    print(f"Wrote consistency report -> {args.report}")


if __name__ == "__main__":
    main()
