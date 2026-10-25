# Pre-op triage: approach and design decisions

## Problem

A surgery can't be scheduled until documentation, testing and risk checks are complete. For each patient package the system decides whether the case can be scheduled (READY / NEEDS_FOLLOW_UP / NOT_CLEARED) and lists every missing or blocking item, with evidence. The policy has four rules: H&P and signed consent; CBC/CMP windows by procedure risk; an anticoagulation plan; acute BP and temperature exclusions. Any data needed to evaluate a rule that is missing means follow-up.

The starter code asked one model call to read the chart and apply the policy. On the repo's default model (`gpt-4.1-mini`) it scored 74.67%, and most of its errors were in applying the policy:
- it flagged 9 READY cases as NEEDS_FOLLOW_UP;
- it raised 7 spurious anticoagulation issues and 6 spurious testing issues;
- it missed 6 `MISSING_REQUIRED_DATA` cases;
- it got acute-safety exclusions wrong in both directions (2 missed, 3 spurious).

Even on `gpt-5.6-luna` (97.33%), the remaining misses were policy errors: aspirin treated as an anticoagulant, window violations reported when there was no procedure date, and an unknown anticoagulant status counted twice.

## 1. Architecture: two layers

**Layer 1: extraction (the model reads, Python checks).**
- **Structured fields, normalised without a model:**
  - lab codes (`LAB-CBC` → CBC; a BMP is never a CMP);
  - lab statuses (`final`, `amended` and `corrected` count);
  - document titles, tolerating typos ("History and Pyhsical");
  - medication names against 16 anticoagulants and 10 antiplatelets, by generic and brand name.
- **One model call per case**, using a strict JSON schema, `temperature=0`, `reasoning.effort="none"`, `store=False` and a versioned prompt cache key.
  - The model sees documents only: no patient identifiers, no policy, no structured fields.
  - It returns facts (vitals, medication statements, hold/resume instructions, consent signature status, H&P completion), each with a verbatim quote.
- **Python verifies every fact:**
  - the document exists;
  - the quote is in that document;
  - the value is inside the quote;
  - the temperature unit is the one written next to the number.

  Anything that fails is logged and never used as a fact.

**Layer 2: a deterministic policy engine (Python only).**
- It handles all date arithmetic, ordering of readings, unit conversion (`Decimal`, so 38.0 °C is exactly 100.4 °F) and category mapping.
- Every rule writes a PASS / FAIL / INDETERMINATE / NOT_APPLICABLE check, with a summary, source IDs and quotes. Each FAIL becomes an issue.
- Structured procedure fields always win; documents are only a fallback for null or invalid values.
- Unknown inputs become `MISSING_REQUIRED_DATA` rather than guesses. For example, an unknown risk level means labs are evaluated under every level, and a failure is reported only if the lab fails under all of them.
- A rule that couldn't be evaluated can never produce READY.

The split matters: the model does the part models are good at (reading messy notes), and Python does the part that has to be exact and auditable (the policy).

## 2. Results, telemetry and cost

All configurations were scored by the unmodified `run_evals.py`. Token counts come from each API response's `usage` field; latency was measured per call with 8 workers. Prices per 1M tokens:

| Model | Input | Cached input | Output | Source |
|---|---|---|---|---|
| `gpt-4.1-mini` | $0.40 | $0.10 | $1.60 | OpenAI's model page |
| `gpt-5.6-luna` | $0.20 | $0.02 | $1.20 | third-party price trackers |

| Per case (mean) | Starter, gpt-4.1-mini (repo default) | Starter, gpt-5.6-luna | Pipeline, gpt-4.1-mini (first / repeat) | Pipeline, gpt-5.6-luna (first / repeat) |
|---|---|---|---|---|
| Eval score | 74.67% | 97.33% | **100%** | **100%** |
| Label match, 3 runs × 50 cases | — | — | 150/150 | 150/150 |
| Byte-identical outputs, 3 runs × 50 cases | — | — | 36/50 | **50/50** |
| Input tokens | 3,039 | 3,029 | 3,830 | 3,828 |
| Cache hit | 4.0% | 0.0% | 44.1% / 79.0% | **53.8% / 99.9%** |
| Reasoning tokens | 0 | 145 | 0 | **0** |
| Billed output tokens | 234 | 297 | 633 / 624 | 511 / 522 |
| p50 latency | 2.08 s | 3.30 s | 4.31 / 4.15 s | 3.77 / 3.39 s |
| Cost per 1,000 cases | $1.55 | $0.96 | $2.04 / $1.62 | **$1.01 / $0.70** |

What the comparison shows:
- **The architecture does the heavy lifting on accuracy.** On the same model (`gpt-4.1-mini`), moving the policy into Python takes accuracy from 74.67% to 100%, with every label matched in all three runs.
- **The model choice drives cost and stability.** On `gpt-4.1-mini` the pipeline costs more than its starter (32% more on first calls, 5% more on repeats: longer outputs at $1.60 per 1M and weaker caching), and only 36 of 50 cases are byte-identical across runs. On `gpt-5.6-luna` it's 35% cheaper than the repo default on first calls and 55% cheaper on repeats.
- **Latency is the trade-off:** p50 is about 1.3–1.7 s slower per call than the `gpt-4.1-mini` starter.

## 3. Why these API settings

- **`reasoning.effort="none"`:** the policy no longer lives in the model, so reasoning tokens are pure cost. Turning reasoning off removed all 145 reasoning tokens per case on `gpt-5.6-luna`. It's also the only setting under which this model accepts `temperature`.
- **`temperature=0`:** reduces sampling variation in extraction. It's not a determinism guarantee: identical inputs still produced 676–742 output tokens across runs.
- **Byte-identical outputs come from Layer 2, not the sampler:**
  - quotes are snapped to whole source lines;
  - anticoagulation evidence lines are found by Python;
  - summaries are built from deterministic values;
  - the policy is pure Python.

  Model variation that doesn't change a verified fact can't change the output. Result: 50 of 50 cases byte-identical across three runs, and 10 of 10 on the harness's determinism check.
- **A ~2,100-token static prefix, above OpenAI's 1,024-token caching minimum:** system instructions plus the strict schema are byte-identical on every request. Everything case-specific goes in the single user message after them. A controlled test confirmed it:
  - a new prefix caches nothing on its first call;
  - a repeat caches 99.9%;
  - a different case reuses exactly the 2,123-token shared prefix;
  - changing one character at the front drops caching to zero.

  The starter's prefix was below the minimum and never cached.
- **More output tokens, on purpose:** verbatim quotes for every fact mean about 515 output tokens per case, against 151–297 for the starter (and 145 of those 297 were reasoning). The extra tokens are what let Python check every fact against the chart. On `gpt-5.6-luna`, caching more than pays for them: a first call costs about the same as the starter ($1.01 vs $0.96 per 1,000) and a repeat is 27% cheaper ($0.70).
- **Other hygiene:** one pooled client (30 s timeout, 2 retries); `store=False` keeps PHI out of stored responses; refusals and truncated responses raise an error instead of guessing; usage telemetry is kept outside the decision output so it can't break byte-identical results.

## 4. Fail-closed vitals: found by the gpt-4.1-mini experiment

The first run of the pipeline on `gpt-4.1-mini` (version 2.0.0) scored 96.0%, with three misses (cases 2, 5 and 47), all acute-safety exclusions.
- **Cause:** the model corrupted the degree sign in its JSON output (a control character plus "6" in place of "°"). The quotes for 157/117, 161/117 and 38.6 °C therefore failed verbatim matching and were rejected.
- **Consequence:** Rule 4 fell back to older, normal structured vitals and returned READY. Rejecting an unverifiable fact is the right call, but quietly falling back made the decision less safe.

Two fixes, both in Python, with no change to the prompt, schema or frozen outputs:
1. **Fail closed:** if a reading fails verification and no newer verified reading of the same type supersedes it, Rule 4 is INDETERMINATE and the case gets `MISSING_REQUIRED_DATA`, so a nurse confirms the latest value. An unverified value is never used to issue NOT_CLEARED either. The same applies to malformed structured vitals.
2. **Degree-sign normalisation in quote matching:** corrupted forms and look-alikes (`º`, `˚`, `℃`, escaped `°`) are folded before comparison. Evidence still shows the original source line.

Re-measured on the current code, the `gpt-4.1-mini` pipeline scores 100%: the three quotes now verify, so all three cases are correctly NOT_CLEARED. The fail-closed rule didn't need to fire in that run; it remains the safety net for quotes that still can't be verified. Neither change alters any decision in the 50 frozen `gpt-5.6-luna` outputs, which had no rejected vitals.

A review pass also fixed three latent rule issues that the sample data didn't trigger:
- an H&P or lab dated after the procedure could hide a valid earlier one;
- a signed consent could fail if the model omitted the procedure name (Python now reads the form's `Procedure:` line);
- a fractional structured BP could be truncated (179.6 → 179) instead of being flagged;
- `left`, `right` and `bilateral` were ignored when matching consent procedures, so a right-knee consent matched a left-knee schedule. A left/right conflict now always fails as a wrong-site consent.

## 5. Procedure synonyms: deterministic first, AI only as a suggestion

Consent names don't always match the schedule word for word ("hip replacement" vs "total hip arthroplasty", "CABG", "inguinal herniorrhaphy").
- **First line: a small reviewed table** (`PROCEDURE_SYNONYMS`) of synonyms and abbreviations (CABG, TKA/THA, replacement ↔ arthroplasty, herniorrhaphy, RYGB, …). It's applied to both names before the strict word comparison, so "coronary artery bypass grafting" still doesn't match "gastric bypass".
- **Anything still unmatched fails closed** into `REQUIRED_DOCUMENTATION`. A wrong-procedure consent is a patient-safety issue, so an unknown synonym costs a nurse a minute rather than letting a mismatch through.
- **Prepared for AI, without giving it the decision:** an optional `procedure_matcher` (`ai_procedure_matcher.py`, off by default) makes a separate small strict-schema call, but only for consents that fail the deterministic match.
  - Its answer goes into `audit.informational_checks` as a suggestion for the nurse; the consent rule stays failed.
  - Pairs a reviewer confirms are added to the table, so the deterministic path learns them and the result stays reproducible.
  - Because it's a separate request, the extraction prompt, schema and cached prefix don't change.

## 6. Why Hypothesis and stdlib datetime

- **stdlib `datetime` / `zoneinfo`:** the rules are calendar-day windows ("within 30 days") and recency comparisons. Plain `date` objects make day arithmetic exact, with no time-of-day surprises, and need no extra dependency. `zoneinfo` covers the one real timezone question: ordering UTC structured vitals against local note times, via a configurable facility timezone.
- **Hypothesis:** hand-picked examples check the cases you think of; property tests check the invariants across inputs you didn't. Cases are generated on arbitrary dates from 2000 to 2099 (month ends, leap years), and the tests assert:
  - the 14- and 30-day windows hold exactly;
  - a BMP never satisfies a CMP requirement;
  - the latest verified BP ≥ 180/110 or temperature > 100.4 °F always gives NOT_CLEARED;
  - older abnormal readings never override newer ones;
  - a rejected latest vital never resolves to READY;
  - the decision always agrees with the issue categories.
- **Mutation check:** I injected eight deliberate bugs (each threshold moved by one, `<=` changed to `<`, a fail-open supersession rule) and the suite caught all of them. The check also showed that uniformly random inputs rarely hit a threshold exactly, so the strategies now oversample boundary values.

## 7. Evidence packet and audit trail for nurse review

The output is built for a pre-op nurse who reviews every blocked case and samples READY ones. Each issue carries:
- the rule that failed and a plain-language summary;
- the source IDs;
- quotes marked `verbatim` (whole lines from the chart, searchable directly) or not (renderings of structured rows).

The audit records every rule's result, including passes, plus how procedure date and risk were resolved, the data that was ignored and why, and warnings.

- **Blocked chart (case 5):** the BP exclusion cites the nursing-note line `09:05  BP 157/117, Temp 36.7 °C …`. The structured vitals looked normal; the audit shows exactly which newer reading blocked the case.
- **Sampled READY chart (case 16):** the BP check passes on the 08:30 recheck (137/78) and lists the superseded 08:06 reading of 189/112 with both source lines. The consent check quotes the patient signature line and the procedure line. The nurse can confirm each rule against one quoted line instead of re-reading the chart.
- **Follow-up chart (case 18):** the anticoagulation issue quotes the medication list and the note "warfarin was stopped last month and replaced with Eliquis". It shows that the plan on file is for the old drug.

## 8. Workflow boundary: scheduling gate vs day-of-surgery checks

From discussions with the owner of an electrophysiology clinic: items such as NPO/fasting instructions are about adherence on the day of surgery. Nursing staff confirm them at intake and on the day itself; they aren't criteria for whether a case can be scheduled.

So the scheduling decision is bound strictly to the appendix policy (Rules 1–4), and the system adds no criteria of its own. NPO review is still recorded for the nurse: a deterministic scan of nursing intake notes writes one of five statuses (reviewed with patient, instructions documented, not reviewed, not documented, no intake note) to `audit.informational_checks`. It's marked `affects_decision: false` and tested so it can never change a decision.

## 9. Limits and next steps

- **Timezone:** same-day ordering of UTC structured vitals against note times needs the facility timezone configured. Without it, the most severe reading is used.
- **Synonym coverage:** procedure synonyms outside the reviewed table go to follow-up unless the AI suggestion is enabled and a nurse confirms. Dates are parsed as US M/D/YYYY, and the text patterns are English.
- **One call per case:** very large charts fail closed to manual review; there's no chunking.
- **Model drift:** `gpt-5.6-luna` has no dated snapshot. The prompt-version hash in every audit, the recorded extraction fixtures and the multi-run consistency check are how drift would be detected.
- **Result versions:** the `gpt-4.1-mini` pipeline numbers are from the current code (2.2.0). The `gpt-5.6-luna` pipeline outputs were produced by 2.0.0 and kept frozen; every later change was checked against them and alters no decision or issue.
