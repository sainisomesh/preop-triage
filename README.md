# Pre-Op Triage Take-Home

## Objective

Implement `triage_submission(...)` in `core.py` - it is a pre-op triage function for a single submission package. It is currently a naive LLM-based solution that makes a real model API call. The starter implementation intentionally does not follow some best practices in using the OpenAI API. You may use whatever file structure makes sense for your solution.

Your output must match this schema:

- `decision`: `READY | NEEDS_FOLLOW_UP | NOT_CLEARED`
- `issues[]`: `category` + evidence (how you design this is up to you)
- `explanation`

## What Is Provided

- `data/patients_sample_50.jsonl` includes:
  - `case_id`
  - `submission`
  - `label` (what a human labeled this case as)
- `run_baseline.py` runs your `triage_submission` implementation and writes outputs.
- `run_evals.py` scores outputs against provided `label` and can run determinism checks.

## Completion

Note: this exercise is evaluated on engineering judgment. You may not reach a 100% score, and that is OK! We are looking to understand how you approached the problem and designed a working solution.

---

# Solution

The model reads the documents and returns facts with verbatim quotes. Python checks every quote against the source text, then applies the policy deterministically. The provided harness (`run_baseline.py`, `run_evals.py`, `view_report.py`) is unchanged.

## Setup

```bash
uv --version                                   # uv is the only prerequisite (Python >= 3.11)
uv sync                                        # optional: make test syncs automatically (runtime + dev deps from uv.lock)
echo 'OPENAI_API_KEY=sk-...' > .env            # gitignored; the Makefile loads it (no quotes, one line)
```

`export OPENAI_API_KEY=...` also works if `.env` is absent.

### What changed from the starter setup
- **Dependencies:** `pyproject.toml` and `uv.lock` declare them: `openai` and `pydantic` at runtime; `pytest` and `hypothesis` in the dev group. `make test` now runs `uv run python -m pytest tests` in that environment instead of `uv run --with ...`.
- **Makefile:** loads `.env`, defaults `MODEL` to `gpt-5.6-luna`, and adds `make consistency`.
- **Unchanged:** `make baseline`, `make evals`, `make determinism`, `make score` and `make report` work as before. `run_baseline.py`, `run_evals.py` and `view_report.py` are byte-identical to the starter.

## Commands

| Command | What it does | Needs API key |
|---|---|---|
| `make test` | 159 offline tests, including Hypothesis property tests (`HYPOTHESIS_PROFILE=stress make test` runs 3,000 examples per property) | no |
| `make baseline` | runs all 50 cases → `data/baseline_outputs.jsonl` | yes |
| `make evals` | scores with the provided harness → `data/eval_report.json` | yes |
| `make determinism` | provided 10× replay of one record → `data/determinism_report.json` | yes |
| `make score` | prints the aggregate score | no |
| `make all` | baseline + evals + determinism + score | yes |
| `make consistency` | every case `RUNS` times (default 3), fresh model calls, 8 workers → `data/consistency_report.json` with byte-identity, label match and token/cache/latency telemetry | yes |
| `make report` | interactive TUI over the eval report | no |

Configuration:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL` | `gpt-5.6-luna` | model id (the pipeline sends `temperature=0` with `reasoning.effort="none"`, which this model requires) |
| `RUNS` | `3` | repetitions for `make consistency` |
| `PREOP_FACILITY_TIMEZONE` | unset | facility IANA timezone (e.g. `America/Chicago`) used to order UTC structured vitals against note wall-clock times |
| `INPUT` / `OUTPUT` / `REPORT` / `DETERMINISM_REPORT` / `CONSISTENCY_REPORT` | `data/...` | file locations |

## Results

The task: decide whether a patient can be scheduled, and list what's missing or blocking with evidence for each. Every configuration was scored with the same unmodified `run_evals.py`. The original repo (every commit and branch) defaults to `gpt-4.1-mini` and publishes no reference score, so the starter code on that model is the developers' own baseline. Raw outputs and reports are in `results/`.

| | Starter, `gpt-4.1-mini` (repo default) | Starter, `gpt-5.6-luna` | Pipeline, `gpt-4.1-mini` | **Pipeline, `gpt-5.6-luna`** |
|---|---|---|---|---|
| `make evals` score | 74.67% | 97.33% | 96.0% | **100.0%** |
| Schema / decision / categories | 100 / 68 / 56% | 100 / 98 / 94% | 100 / 94 / 94% | 100 / 100 / 100% |
| Case 0, 10 runs: same decision | 100% (but wrong) | not run | 100% | **100%** |
| Case 0, 10 runs: exact output | 10% | not run | 100% | **100%** |
| Byte-identical full output, 3 runs × 50 cases | — | — | 32 / 50 | **50 / 50** |

Run dates: starter on `gpt-5.6-luna` 2026-10-23, before any changes; everything else 2026-10-24. The `gpt-4.1-mini` pipeline run is an experiment: the same code with the `reasoning` parameter removed, because that model doesn't accept it.

The starter code fails for the same reason on both models: the model applies the policy itself. On `gpt-5.6-luna` it treated aspirin as an anticoagulant, reported window violations when there was no procedure date, and double-counted an unknown anticoagulant status. On `gpt-4.1-mini` it misses 29 of 50 cases. The same two-layer pipeline on `gpt-4.1-mini` misses only 3, so the architecture, not the model, does most of the work. Those 3 misses led to the fail-closed vitals fix described below.

## Design

### 1. Two layers: extraction, then deterministic rules

```
submission ─► Layer 1  extraction.py ─────────────────────────────► Layer 2  policy_engine.py ─► TriageOutput
              preprocess structured fields (no model)               R0 procedure date/risk         decision
              one strict-schema model call over documents          R1 H&P window, signed consent  issues[] (+ evidence)
              verify every fact against source text                R2 CBC / CMP windows           explanation
                                                                   R3 anticoagulation plan        audit
                                                                   R4 BP / temperature exclusion
```

**Layer 1 (`extraction.py`)**
- **Structured fields, no model:**
  - Labs are normalised (`LAB-CBC` → CBC; a BMP never counts as a CMP).
  - Labs must be `final`, `amended` or `corrected`; anything dated after `submission_received_at` is excluded.
  - Document titles are classified by type, tolerating typos ("History and Pyhsical").
  - Medication-list names are matched against 16 anticoagulants and 10 antiplatelets (generic and brand names) in `reference.py`.
- **One model call per case** via `responses.parse` with a strict JSON schema, `temperature=0`, `reasoning.effort="none"`, `store=False` and a versioned `prompt_cache_key`.
  - The model receives documents only: no patient identifiers, no policy, no structured fields.
  - It returns vitals, medication statements, hold/resume instructions, consent signature status, H&P completion and, only when the structured field is unusable, procedure date/risk statements. Each item comes with a verbatim quote.
- **Verification:** each fact must point to a real document, its quote must exist in that document, and its value must appear inside the quote.
  - Temperature units are taken from the text, not the model, and converted with `Decimal`.
  - Documents are shown to the model as `D1`, `D2`, … because it was observed mangling UUIDs when copying them back.
  - Facts that fail verification are logged in `audit.excluded_evidence` and are never used as facts.
  - Vitals that fail verification are also kept as *unverified* readings, so that Rule 4 can fail closed (below).
  - Before comparison, quote matching folds degree-sign variants: `gpt-4.1-mini` emitted a control character plus "6" in place of "°", and `º`, `˚` and `℃` also appear in real text. Evidence still shows the original source line.

**Layer 2 (`policy_engine.py`)** is pure Python: no model, clock reads or randomness. It does all date arithmetic, recency ordering, unit thresholds and category mapping.
- **Procedure fields:** structured `procedure_date`/`procedure_risk` always win. Document statements are used only when the field is null or invalid, and only if they state the value explicitly.
- **Unknown inputs:**
  - An unknown risk evaluates labs under every risk level, and reports a failure only if the lab fails under all of them.
  - An unknown date keeps the window checks INDETERMINATE instead of failing them.
  - An unknown anticoagulant status is `MISSING_REQUIRED_DATA`, not `ANTICOAGULATION_MANAGEMENT`.
- **Anticoagulation plans:**
  - A plan needs both a hold and a resume instruction, each with a concrete time.
  - The plan must be for the drug the patient is taking now, matched across brand and generic names.
  - Antiplatelets never require a plan.
- **H&P and labs:** only items dated on or before the procedure count. "Most recent result" means the most recent pre-procedure result.
- **Vitals:** a later recheck always supersedes an earlier abnormal reading. Same-day readings that can't be ordered (UTC structured timestamps vs local note times with no facility timezone) use the most severe value, and the audit says why.
- **Fail-closed vitals:** suppose a reading failed verification (quote not in the note, value not in the quote, implausible or malformed) and no newer verified reading of the same type supersedes it. Then Rule 4 is INDETERMINATE and the case gets `MISSING_REQUIRED_DATA`, so a nurse confirms the latest value. Before this rule, an unverifiable 157/117 silently fell back to an older normal reading and returned READY. An unverified value is never used to issue NOT_CLEARED either.
- **Safety rule:** a rule that couldn't be evaluated can never produce READY.

### 2. Starter baselines vs this pipeline: measured tokens, latency and cost

Each configuration was run on all 50 cases with 8 workers, using the API's `usage` fields. `python3 results/summarize_usage.py` rebuilds this table from the saved files. Prices per 1M tokens:
- `gpt-4.1-mini`: $0.40 input, $0.10 cached, $1.60 output (OpenAI's model page);
- `gpt-5.6-luna`: $0.20 input, $0.02 cached, $1.20 output (third-party price trackers).

Reasoning tokens are billed as output.

| Per case (mean) | Starter, `gpt-4.1-mini` | Starter, `gpt-5.6-luna` (`reasoning: medium`) | Pipeline, `gpt-4.1-mini`, first / repeat | **Pipeline, `gpt-5.6-luna`, first / repeat** |
|---|---|---|---|---|
| Input tokens | 3,039 | 3,029 | 3,830 | 3,828 |
| Cached input tokens | 120 | 0 | 1,651 / 3,090 | 2,058 / 3,825 |
| Cache hit (by tokens) | 4.0% | 0.0% | 43.1% / 80.7% | **53.8% / 99.9%** |
| Reasoning tokens | 0 | 145 | 0 | **0** |
| Visible = billed output tokens | 234 | 151 (297 billed) | 620 / 612 | 511 / 522 |
| Latency per call, p50 | 2.08 s | 3.30 s | 4.40 / 4.21 s | 3.77 / 3.39 s |
| Cost per 1,000 cases | $1.55 | $0.96 | $2.03 / $1.58 | **$1.01 / $0.70** |

- **Against the repo default (`gpt-4.1-mini`):** the pipeline is 35% cheaper on first calls and 55% cheaper on repeats, and accuracy rises from 74.67% to 100%. It's slower per call (p50 2.1 s → 3.4–3.8 s).
- **Reasoning tokens are eliminated.** On `gpt-5.6-luna` the starter spends 145 reasoning tokens per case applying the policy. The pipeline sends `reasoning.effort="none"`, and Python applies the policy in about 45 ms per case.
- **Same pipeline on `gpt-4.1-mini`:** accuracy goes from 74.67% to 96.0%, but it costs more than its own starter: 31% more on first calls and about the same on repeats. Its quotes are longer (about 615 output tokens at $1.60 per 1M) and it caches less. Model choice drives cost; the architecture drives accuracy and determinism.
- **Visible output is larger, deliberately.** Verbatim quotes for every fact mean 511–522 visible tokens per case, against 151–234. That's the price of having Python check each fact against the chart instead of trusting the model. Caching more than pays for it.
- **The cacheable prefix sits above OpenAI's 1,024-token minimum.** System instructions plus the strict schema come to about 2,100 tokens and are byte-identical on every request. All case-specific content, including the fallback flags, goes in the single user message after them. The starter's static prefix is below the minimum, so it caches little (4.0%) or nothing.

Cache behaviour was confirmed with a controlled experiment on a never-before-seen prefix:

| Request | Input | Cached |
|---|---|---|
| Case 3, first call | 4,343 | 0 |
| Case 3, repeated | 4,343 | 4,340 |
| Case 7, same prefix | 3,867 | 2,123 (exactly the shared static prefix) |
| Case 7, one character changed at the front of the instructions | 3,868 | 0 |

### 3. Audit trail: built for pre-op nurse review

The audit trail is designed for the production workflow, where a pre-op nurse reviews blocked cases and samples READY ones before the schedule is confirmed. Every output carries `audit`, which contains:
- the policy and prompt versions;
- how procedure date and risk were resolved (source, raw value, notes);
- a `rule_checks` entry for every rule: `PASS` / `FAIL` / `INDETERMINATE` / `NOT_APPLICABLE`, each with a summary, source IDs and quotes;
- `excluded_evidence`, listing data that was ignored and why;
- `warnings` and `informational_checks`.

Quotes marked `verbatim: true` are whole lines copied from the source document, so a nurse can search the chart for them directly. Quotes marked `verbatim: false` are renderings of structured rows (lab, vital, medication-list entry). Issues reuse the evidence of the rule check that failed.

- **Blocked case (case 05, NOT_CLEARED):** `R4.BLOOD_PRESSURE` reports the most recent blood pressure, BP 157/117 mmHg at 2026-03-03 09:05, which meets the DBP ≥ 110 exclusion. It quotes the nursing note line `09:05  BP 157/117, Temp 36.7 °C, HR 69, SpO2 99% RA`. The structured vitals looked normal, and the audit shows exactly which newer reading drove the block.
- **READY case worth sampling (case 16):** `R4.BLOOD_PRESSURE` passes on the `08:30` recheck (137/78) and lists the superseded `08:06` reading of 189/112 with both source lines. `R1.SURGICAL_CONSENT` quotes the patient signature line and the procedure line. A reviewer sampling READY cases can check every rule against its quoted line without opening the full chart.
- **Follow-up case (case 18):** `R3.ANTICOAGULATION.APIXABAN` fails with the medication-list entry and the note line "warfarin was stopped last month and replaced with Eliquis". Warfarin is `NOT_APPLICABLE`, and the evidence shows that its plan predates the switch.

### 4. Clinical workflow boundary

This came out of discussions with the owner of an electrophysiology clinic. Items such as NPO/fasting instructions are about adherence on the day of surgery: nursing staff confirm them at intake and on the day itself. They aren't criteria for whether a case can be scheduled. So the scheduling gate covers only the appendix policy (Rules 1–4); the pipeline adds no criteria of its own.

NPO review is still useful to the nurse, so a deterministic scan of nursing intake notes records it in `audit.informational_checks` with one of five statuses:
- `REVIEWED_WITH_PATIENT`
- `INSTRUCTIONS_DOCUMENTED`
- `NOT_REVIEWED`
- `NOT_DOCUMENTED`
- `NO_NURSING_INTAKE_NOTE`

These checks are marked `affects_decision: false`, live outside `rule_checks`, and are covered by tests showing they never change the decision or issues.

## Testing

`make test` runs 159 tests offline in about 2 seconds:
- **Rule-by-rule unit tests**, including window boundaries, signature and procedure matching, brand/generic reconciliation, lab statuses, timezones, mixed °C/°F strings and the NPO informational check.
- **Failure-mode tests:** invented, near-miss, stitched and wrong-document quotes; degree-sign corruption; fail-closed vitals.
- **API-settings tests:** strict schema, `temperature=0`, `store=False`, a static cache prefix, the client's timeout and retries, and telemetry parsing.
- **Replays of recorded real extractions** (`tests/fixtures/`, cases 16 and 25).
- **Hypothesis property tests** (`tests/test_properties.py`), with cases built on arbitrary dates from 2000 to 2099:
  - H&P, CBC and CMP windows hold exactly at 14 and 30 days;
  - a BMP never satisfies a CMP requirement;
  - the latest verified BP ≥ 180/110 or temperature > 100.4 °F always gives NOT_CLEARED with `ACUTE_SAFETY_EXCLUSION`;
  - older abnormal readings never override newer ones;
  - °C converts exactly;
  - a rejected latest vital never resolves to READY;
  - the decision always agrees with the issue categories.

A mutation check confirmed the suite catches eight deliberate bugs: each threshold moved by one, `<=` changed to `<` in the lab window, and a fail-open supersession rule. The check also showed the threshold properties needed boundary-biased inputs, which they now have.

## Code map

| File | Role |
|---|---|
| `core.py` | entry point: shared OpenAI client (30 s timeout, 2 retries), facility timezone, telemetry hook |
| `extraction.py` | Layer 1: preprocessing, prompt and strict schema, model call, usage telemetry, verification |
| `policy_engine.py` | Layer 2: rules, issues, explanation, audit trail, informational checks |
| `reference.py` | drug, lab and title reference data; date/time, quote and procedure-matching helpers |
| `schemas.py` | input, output and audit models |
| `run_consistency.py` | multi-run consistency and telemetry report |
| `results/` | raw outputs, eval reports and usage for all four configurations (see `results/README.md`); `summarize_usage.py` rebuilds the cost table |
| `tests/` | offline tests (`test_properties.py` holds the Hypothesis properties); `tests/fixtures/` holds recorded real extractions plus the script that records them |
| `pyproject.toml`, `uv.lock` | runtime and dev dependencies |

## Known limits

- **Timezone:** same-day ordering of UTC structured vitals against note times needs `PREOP_FACILITY_TIMEZONE`. Without it the most severe reading is used.
- **Procedure names:** consent matching is strict word matching, so synonyms ("hip replacement" vs "arthroplasty") go to follow-up instead of passing.
- **Locale:** dates are parsed as US `M/D/YYYY`, and the text patterns are English.
- **Large charts:** one model call per case. A chart large enough to hit the output limit raises `ExtractionError` for manual review; there's no chunking.
- **Model drift:** `gpt-5.6-luna` has no dated snapshot. The prompt-version hash in every audit, the recorded fixtures and `make consistency` are how drift would be detected.
- **Runtime failures:** model refusals or truncated responses raise `ExtractionError` instead of producing a guessed decision.
- **Not re-measured live:** the fail-closed vitals rule and the degree-sign normalisation (pipeline 2.1.0) are covered offline. They don't change any decision in the frozen `gpt-5.6-luna` outputs. The `gpt-4.1-mini` numbers above were measured with 2.0.0.
