# Results

Measured artifacts behind the numbers in the top-level README. All eval reports were produced by the unmodified `run_evals.py`. Usage figures come from each API response's `usage` field.

| Folder | What ran | When | Pipeline version | Score |
|---|---|---|---|---|
| `starter_gpt-4.1-mini/` | Original starter `core.py`, unmodified, with the repo's default model: `make baseline/evals/determinism` from a pristine `git archive` of the starter commit | 2026-10-24 | starter | 74.67% |
| `starter_gpt-5.6-luna/` | Original starter `core.py`, unmodified, on `gpt-5.6-luna` | 2026-10-23 | starter | 97.33% |
| `pipeline_gpt-4.1-mini/` | Current pipeline on `gpt-4.1-mini` (the `reasoning` parameter dropped, since this model rejects it): 3 passes × 50 cases, plus case 0 replayed 10 times | 2026-10-24 | 2.2.0 | 100.0% |
| `pipeline_gpt-5.6-luna/` | Pipeline via `make baseline/evals/determinism/consistency` | 2026-10-24 | 2.0.0 | 100.0% |

File types:
- **`usage_replay.json`:** the same request replayed on all 50 cases with 8 workers to capture tokens and latency. The archived runs didn't log usage.
- **`experiment_report.json`:** the `gpt-4.1-mini` pipeline's per-pass usage, consistency and determinism.
- **`cost_comparison.json`:** written by `python3 results/summarize_usage.py`, which also prints the comparison table.

The `gpt-5.6-luna` pipeline outputs were produced by 2.0.0 and kept frozen. Every later change was checked against those 50 outputs and changes no decision or issue. Only summary wording and `pipeline_version` would differ, because the outputs contain:
- no rejected vitals;
- no H&P or lab dated after the procedure;
- every consent naming its procedure, with no synonym or laterality cases.

The later changes:
- **2.1.0:** fail-closed unverified vitals, degree-sign normalisation, pre-procedure filtering of H&Ps and labs, a consent `Procedure:` line fallback.
- **2.2.0:** procedure synonyms, laterality conflicts, optional AI procedure suggestions.

The first `gpt-4.1-mini` pipeline run (2.0.0) scored 96.0%. Its three misses (cases 2, 5 and 47) were abnormal vitals whose quotes had a corrupted degree sign, which led to the 2.1.0 fixes. That run has been replaced by the 2.2.0 re-measurement above, where all three are correctly NOT_CLEARED.
