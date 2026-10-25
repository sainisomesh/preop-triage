"""Entry point: extraction.py reads the documents (one model call), policy_engine.py decides.

The model only extracts quoted facts; every date, unit, ordering and policy decision is Python.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Callable

from extraction import EXTRACTION_SYSTEM_PROMPT, ExtractionError, ExtractionTelemetry, extract_case_facts
from policy_engine import evaluate
from schemas import (
    AuditTrail,
    Decision,
    Evidence,
    EvidenceQuote,
    IssueCategory,
    PatientSubmission,
    ProcedureRisk,
    RuleCheck,
    TriageIssue,
    TriageOutput,
)

OPENAI_TIMEOUT_SECONDS = 30.0
OPENAI_MAX_RETRIES = 2
# Facility IANA timezone (e.g. "America/Chicago"). Needed to order UTC structured vitals
# against note wall-clock times; unset -> the most severe same-day reading is used.
FACILITY_TIMEZONE_ENV = "PREOP_FACILITY_TIMEZONE"

__all__ = [
    "EXTRACTION_SYSTEM_PROMPT",
    "OPENAI_MAX_RETRIES",
    "OPENAI_TIMEOUT_SECONDS",
    "AuditTrail",
    "Decision",
    "Evidence",
    "EvidenceQuote",
    "ExtractionError",
    "ExtractionTelemetry",
    "IssueCategory",
    "PatientSubmission",
    "ProcedureRisk",
    "RuleCheck",
    "TriageIssue",
    "TriageOutput",
    "get_openai_client",
    "triage_output_json_schema",
    "triage_submission",
]


@functools.lru_cache(maxsize=1)
def get_openai_client() -> Any:
    # One pooled client per process; imported lazily so the offline layers and tests
    # run without the openai package.
    from openai import OpenAI

    return OpenAI(timeout=OPENAI_TIMEOUT_SECONDS, max_retries=OPENAI_MAX_RETRIES)


def triage_output_json_schema() -> dict[str, object]:
    return TriageOutput.model_json_schema()


def triage_submission(
    submission: dict[str, object] | PatientSubmission,
    *,
    model: str,
    client: Any | None = None,
    facility_timezone: str | None = None,
    on_telemetry: Callable[[ExtractionTelemetry], None] | None = None,
    procedure_matcher: Callable[[str, str], tuple[bool, str, str]] | None = None,
) -> TriageOutput:
    # Raises ExtractionError (refusal/truncation) or an OpenAI error after 2 retries rather
    # than guessing; callers should send those cases to manual review. on_telemetry gets
    # token/cache usage, which is kept out of TriageOutput so outputs stay byte-identical.
    # procedure_matcher (off by default; see ai_procedure_matcher.py) only adds informational
    # suggestions when a consent's procedure name fails deterministic matching.

    if not isinstance(submission, PatientSubmission):
        submission = PatientSubmission.model_validate(submission)
    timezone = facility_timezone or os.environ.get(FACILITY_TIMEZONE_ENV) or None
    facts = extract_case_facts(
        submission, model=model, client=client or get_openai_client(), facility_timezone=timezone,
        on_telemetry=on_telemetry, procedure_matcher=procedure_matcher,
    )
    return evaluate(facts)
