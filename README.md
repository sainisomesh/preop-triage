# Pre-Op Triage

Decides whether a surgical case can be scheduled (READY / NEEDS_FOLLOW_UP / NOT_CLEARED) and lists every missing or blocking item with its evidence.

## Run

```bash
uv sync                                   # Python >= 3.11; runtime + dev deps from uv.lock
echo 'OPENAI_API_KEY=sk-...' > .env        # gitignored; loaded by the Makefile
make test                                 # 173 offline tests (pytest + Hypothesis); no API key needed
make all                                  # baseline -> evals -> determinism -> score on gpt-5.6-luna
make consistency                          # every case x3: byte-identity, label match, token/cache telemetry
make report                               # TUI over data/eval_report.json
```

The provided harness (`run_baseline.py`, `run_evals.py`, `view_report.py`) is unchanged. Raw outputs and how each number was measured are in `results/`.

## Approach and design decisions

### Architecture

I split the system into a constrained extraction layer and a pure Python policy engine so the model does only what it is good at. One strict-schema call reads the chart and returns facts (vitals, medication statements, hold and resume instructions, consent signatures), each with a verbatim quote. Everything that must be exact lives in Python: date math, recency sorting, unit conversion, anticoagulant versus antiplatelet classification, and procedure matching. The model never sees the policy, so it has nothing to improvise, and Python rejects any quote it cannot find in the source. Once the policy moved into code, reasoning tokens were pure cost, so I pair `temperature=0` with `reasoning="none"`. Instructions and schema form a byte-identical 2,100-token prefix, above the 1,024-token caching threshold and routed with a versioned `prompt_cache_key`, with every case-specific detail after it. Verbatim quotes cost more output tokens than a bare decision, but in exchange I get a 54% cache hit on first calls and 99.9% on repeats. And because Python snaps every quote to its full source line, the outputs were byte-identical on all 50 cases across repeated runs.

### Proving it on a smaller model

To prove the architecture, not the model, drives accuracy, I ran the same pipeline on gpt-4.1-mini, the repository's default. The starter code scores 74.67% there; the pipeline scored 96.0% on its first run. The misses mattered more than the score. The smaller model corrupted the degree symbol when serializing quotes, so Python correctly rejected genuinely abnormal vitals, but Rule 4 then fell back to older normal readings and cleared those charts. I made the rule fail closed: if the latest vital is unverified or malformed and no newer verified reading supersedes it, the chart goes to nurse review as `MISSING_REQUIRED_DATA`, and an unverified value can never trigger an exclusion. With that rule and degree-sign normalization, gpt-4.1-mini reached 100%. An offline pytest suite with Hypothesis properties pins these invariants across arbitrary dates, and a mutation check confirmed it catches off-by-one thresholds and fail-open regressions.

### Built for nurse review

The output is built for the pre-op nurse who has to trust it. Each issue carries the failed rule, a plain summary, source IDs and verbatim-flagged quotes. The audit ledger records every rule that passed, how procedure date and risk were resolved, and every excluded record with its reason, so a nurse can confirm a blocked chart or spot-check a ready one in seconds. I also spoke with the owner of an electrophysiology clinic about where cases fall apart, and the answer was day-of-surgery adherence: patients forgetting fasting instructions or anticoagulant hold windows. Hold windows are already in the policy. NPO review is not, so I track it as an informational nurse check in the audit trail, with complete provenance, while the scheduling gate stays strictly bound to the appendix policy.

### Measured comparison

| 50 cases | Starter, gpt-4.1-mini | Pipeline, gpt-4.1-mini | Starter, gpt-5.6-luna | Pipeline, gpt-5.6-luna |
|---|---|---|---|---|
| Eval score | 74.67% | 100% (96.0% first run) | 97.33% | 100% |
| Byte-identical across 3 runs | — | 36 / 50 | — | 50 / 50 |
| Cache hit, first / repeat | 4.0% | 44% / 79% | 0% | 54% / 99.9% |
| Reasoning / output tokens per case | 0 / 234 | 0 / 633 | 145 / 297 | 0 / 511 |
| p50 latency per call | 2.1 s | 4.3 s | 3.3 s | 3.8 s |
| Cost per 1,000 cases, first / repeat | $1.55 | $2.04 / $1.62 | $0.96 | $1.01 / $0.70 |
