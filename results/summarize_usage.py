"""Rebuild the baseline-vs-pipeline token/cost table from the measured usage files in results/.

    python3 results/summarize_usage.py

Each usage_replay.json is 50 real API calls (8 workers) with usage taken from the
response's `usage` field. Eval scores come from each folder's eval_report.json (the
graded run); token figures come from the usage replay of the same request. The pipeline's
first-call row is run 1 of pipeline_gpt-5.6-luna/consistency_report.json.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# $ per 1M tokens. gpt-4.1-mini: OpenAI's model page. gpt-5.6-luna: third-party API price
# trackers (Layer3 Labs, OpenRouter, PricePerToken), not OpenAI's page. Reasoning bills as output.
PRICES = {
    "gpt-4.1-mini": {"input": 0.40, "cached": 0.10, "output": 1.60},
    "gpt-5.6-luna": {"input": 0.20, "cached": 0.02, "output": 1.20},
}

CONFIGS = [
    ("starter_gpt-4.1-mini", "Starter code, gpt-4.1-mini (repo default)", "gpt-4.1-mini"),
    ("starter_gpt-5.6-luna", "Starter code, gpt-5.6-luna", "gpt-5.6-luna"),
    ("pipeline_gpt-5.6-luna", "Two-layer pipeline, gpt-5.6-luna (repeat call)", "gpt-5.6-luna"),
]


def cost(model: str, input_tokens: int, cached: int, output: int) -> float:
    price = PRICES[model]
    return ((input_tokens - cached) * price["input"] + cached * price["cached"] + output * price["output"]) / 1e6


def summarize_calls(folder: str, label: str, model: str) -> dict:
    data = json.loads((HERE / folder / "usage_replay.json").read_text())
    calls = data["calls"]
    n = len(calls)
    total_in = sum(c["input_tokens"] for c in calls)
    total_cached = sum(c["cached_input_tokens"] for c in calls)
    total_out = sum(c["output_tokens"] for c in calls)
    total_reasoning = sum(c["reasoning_tokens"] for c in calls)
    latencies = sorted(c["latency_ms"] for c in calls)
    eval_report = json.loads((HERE / folder / "eval_report.json").read_text())
    return {
        "config": label,
        "model": model,
        "eval_score_pct": eval_report["local_metrics_summary"]["aggregate_local_score_pct"],
        "avg_input_tokens": round(total_in / n),
        "avg_cached_input_tokens": round(total_cached / n),
        "cache_hit_pct": round(100 * total_cached / total_in, 1) if total_in else 0.0,
        "avg_output_tokens": round(total_out / n),
        "avg_reasoning_tokens": round(total_reasoning / n),
        "avg_visible_output_tokens": round((total_out - total_reasoning) / n),
        "latency_p50_ms": latencies[n // 2],
        "latency_p95_ms": latencies[int(0.95 * (n - 1))],
        "cost_50_cases_usd": round(cost(model, total_in, total_cached, total_out), 4),
        "cost_per_1k_cases_usd": round(cost(model, total_in, total_cached, total_out) / n * 1000, 2),
    }


def summarize_pipeline_first_call() -> dict:
    report = json.loads((HERE / "pipeline_gpt-5.6-luna" / "consistency_report.json").read_text())
    run1 = report["summary"]["usage"]["per_run"][0]
    n = run1["calls"]
    total = cost("gpt-5.6-luna", run1["input_tokens"], run1["cached_input_tokens"], run1["output_tokens"])
    return {
        "config": "Two-layer pipeline, gpt-5.6-luna (first call per case)",
        "model": "gpt-5.6-luna",
        "eval_score_pct": 100.0,
        "avg_input_tokens": round(run1["input_tokens"] / n),
        "avg_cached_input_tokens": round(run1["cached_input_tokens"] / n),
        "cache_hit_pct": round(100 * run1["cache_hit_ratio"], 1),
        "avg_output_tokens": round(run1["output_tokens"] / n),
        "avg_reasoning_tokens": run1.get("reasoning_tokens", 0) // n if run1.get("reasoning_tokens") else 0,
        "avg_visible_output_tokens": round(run1["output_tokens"] / n),
        "latency_p50_ms": run1["latency_ms_p50"],
        "latency_p95_ms": None,
        "cost_50_cases_usd": round(total, 4),
        "cost_per_1k_cases_usd": round(total / n * 1000, 2),
    }


def summarize_experiment(run: str, label: str) -> dict:
    # pipeline_gpt-4.1-mini/experiment_report.json: three passes with per-pass usage summaries.
    report = json.loads((HERE / "pipeline_gpt-4.1-mini" / "experiment_report.json").read_text())
    usage = report["usage_per_run"][run]
    eval_report = json.loads((HERE / "pipeline_gpt-4.1-mini" / "eval_report.json").read_text())
    return {
        "config": label,
        "model": "gpt-4.1-mini",
        "eval_score_pct": eval_report["local_metrics_summary"]["aggregate_local_score_pct"],
        "avg_input_tokens": usage["avg_input"],
        "avg_cached_input_tokens": usage["avg_cached"],
        "cache_hit_pct": usage["cache_hit_pct"],
        "avg_output_tokens": usage["avg_output"],
        "avg_reasoning_tokens": usage["avg_reasoning"],
        "avg_visible_output_tokens": usage["avg_output"] - usage["avg_reasoning"],
        "latency_p50_ms": usage["latency_p50_ms"],
        "latency_p95_ms": usage["latency_p95_ms"],
        "cost_50_cases_usd": usage["cost_50_usd"],
        "cost_per_1k_cases_usd": usage["cost_per_1k_usd"],
    }


def main() -> None:
    rows = [summarize_calls(*config) for config in CONFIGS[:2]]
    rows.append(summarize_experiment("run1", "Two-layer pipeline, gpt-4.1-mini (first call per case)"))
    rows.append(summarize_experiment("run2", "Two-layer pipeline, gpt-4.1-mini (repeat call)"))
    rows.append(summarize_pipeline_first_call())
    rows.append(summarize_calls(*CONFIGS[2]))
    (HERE / "cost_comparison.json").write_text(json.dumps({"prices_per_1m_usd": PRICES, "rows": rows}, indent=2) + "\n")

    columns = [
        ("eval_score_pct", "Eval score %"), ("avg_input_tokens", "Avg input"), ("avg_cached_input_tokens", "Avg cached"),
        ("cache_hit_pct", "Cache hit %"), ("avg_reasoning_tokens", "Avg reasoning"), ("avg_visible_output_tokens", "Avg visible output"),
        ("avg_output_tokens", "Avg billed output"), ("latency_p50_ms", "p50 latency ms"), ("cost_50_cases_usd", "Cost / 50 cases $"),
        ("cost_per_1k_cases_usd", "Cost / 1k cases $"),
    ]
    print("| Metric | " + " | ".join(row["config"] for row in rows) + " |")
    print("|---|" + "---|" * len(rows))
    for key, title in columns:
        print(f"| {title} | " + " | ".join("—" if row[key] is None else f"{row[key]:,}" for row in rows) + " |")
    print(f"\nwrote {HERE / 'cost_comparison.json'}")


if __name__ == "__main__":
    main()
