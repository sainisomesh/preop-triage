# Results

Measured artifacts behind the numbers in the top-level README. All eval reports were produced by the unmodified `run_evals.py`. Usage figures come from each API response's `usage` field.

| Folder | What ran | When | Pipeline version |
|---|---|---|---|
| `starter_gpt-4.1-mini/` | Original starter `core.py`, unmodified, with the repo's default model: `make baseline/evals/determinism` from a pristine `git archive` of the starter commit | 2026-10-24 | starter |
| `starter_gpt-5.6-luna/` | Original starter `core.py`, unmodified, on `gpt-5.6-luna` | 2026-10-23 | starter |
| `pipeline_gpt-5.6-luna/` | Two-layer pipeline: `make baseline/evals/determinism/consistency` | 2026-10-24 | 2.0.0 |
| `pipeline_gpt-4.1-mini/` | Same pipeline code on `gpt-4.1-mini` (experiment): 3 passes × 50 cases, plus case 0 replayed 10 times | 2026-10-24 | 2.0.0 |

File types:
- **`usage_replay.json`:** the same request replayed on all 50 cases with 8 workers to capture tokens and latency. The archived runs didn't log usage.
- **`experiment_report.json`:** the `gpt-4.1-mini` pipeline experiment's per-pass usage, consistency and determinism.
- **`cost_comparison.json`:** written by `python3 results/summarize_usage.py`, which also prints the comparison table.

Version 2.1.0 (the current code) adds:
- the fail-closed rule for unverified vitals;
- degree-sign normalisation;
- pre-procedure filtering of H&Ps and labs;
- a consent procedure-line fallback.

None of these changes any decision or issue in `pipeline_gpt-5.6-luna/`. Those outputs contain no rejected vitals, every H&P and lab in them is dated before the procedure, and every consent names its procedure. Only summary wording ("most recent verified …") and `pipeline_version` would differ.

For `pipeline_gpt-4.1-mini/`, the three misses (cases 2, 5 and 47) were abnormal vitals whose quotes had a corrupted degree sign. Under 2.1.0 those quotes verify, and any reading that still fails is sent to review instead of being dropped. This hasn't been re-measured with live calls.
