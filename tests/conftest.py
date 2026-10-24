"""Builders for offline tests: a clean READY submission plus a matching extraction.

Tests mutate copies of these to exercise one rule at a time. No network access.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import pytest
from hypothesis import settings

# Property-test budgets: 150 examples by default; `HYPOTHESIS_PROFILE=stress make test` for 3,000.
settings.register_profile("default", max_examples=150)
settings.register_profile("stress", max_examples=3000)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))

from extraction import (
    DocumentExtraction,
    ExtractedConsent,
    ExtractedHistoryAndPhysical,
    ExtractedVitalReading,
    preprocess,
    verify_extraction,
)
from policy_engine import evaluate
from schemas import PatientSubmission, TriageOutput

HP_TEXT = (
    "PREOPERATIVE HISTORY AND PHYSICAL\n"
    "Planned procedure: inguinal hernia repair (scheduled for 03/31/2026)\n"
    "Vitals 10:05  BP 120/80, HR 70, T 98.1 F\n"
    "Electronically signed by Dana Lee, MD on 03/10/2026."
)
CONSENT_TEXT = (
    "INFORMED CONSENT FOR SURGICAL PROCEDURE\n"
    "Procedure: inguinal hernia repair\n"
    "Patient signature: /s/ Pat Doe (electronically signed 03/20/2026 10:00)\n"
    "Physician attestation: Electronically signed by Dana Lee, MD on 03/20/2026."
)
NURSE_TEXT = "Pre-op nursing intake.\nVITAL SIGNS: Obtained at 08:30: BP 124/76, Temp 98.4 F, HR 72."

BASE_SUBMISSION: dict[str, Any] = {
    "patient": {"id": "p-1"},
    "procedure": {
        "case_id": "case-test",
        "procedure_type": "Elective inguinal hernia repair",
        "procedure_risk": "LOW",
        "procedure_date": "2026-03-31",
    },
    "vitals": [
        {"type": "blood_pressure", "systolic": 118, "diastolic": 72, "date": "2026-03-25T10:00:00Z", "source": "clinic"},
        {"type": "temperature", "value_f": 98.2, "date": "2026-03-25T10:02:00Z", "source": "clinic"},
    ],
    "labs": [
        {"id": "lab-cbc", "code": "CBC", "display": "Complete Blood Count", "effective_at": "2026-03-15T08:00:00Z", "status": "final"},
    ],
    "medications": [],
    "conditions": [],
    "documents": [
        {"doc_id": "doc-hp", "type": "History and Physical", "date": "2026-03-10", "text": HP_TEXT},
        {"doc_id": "doc-consent", "type": "Surgical Consent", "date": "2026-03-20", "text": CONSENT_TEXT},
        {"doc_id": "doc-nurse", "type": "Pre-op Nursing Intake", "date": "2026-03-28", "text": NURSE_TEXT},
    ],
    "metadata": {"submission_received_at": "2026-03-31T12:00:00Z", "source_system": "test"},
}


def base_extraction() -> dict[str, list]:
    return {
        "vital_readings": [
            ExtractedVitalReading(doc_ref="doc-hp", measurement="BLOOD_PRESSURE", value_text="120/80", unit="NOT_STATED",
                                  time_text="10:05", measured_date_text=None, context="MEASURED_AT_THIS_ENCOUNTER",
                                  quote="Vitals 10:05  BP 120/80, HR 70, T 98.1 F"),
            ExtractedVitalReading(doc_ref="doc-nurse", measurement="BLOOD_PRESSURE", value_text="124/76", unit="NOT_STATED",
                                  time_text="08:30", measured_date_text=None, context="MEASURED_AT_THIS_ENCOUNTER",
                                  quote="VITAL SIGNS: Obtained at 08:30: BP 124/76, Temp 98.4 F, HR 72."),
            ExtractedVitalReading(doc_ref="doc-nurse", measurement="TEMPERATURE", value_text="98.4", unit="F",
                                  time_text="08:30", measured_date_text=None, context="MEASURED_AT_THIS_ENCOUNTER",
                                  quote="VITAL SIGNS: Obtained at 08:30: BP 124/76, Temp 98.4 F, HR 72."),
        ],
        "medication_mentions": [],
        "anticoagulation_plans": [],
        "consents": [
            ExtractedConsent(doc_ref="doc-consent", patient_signature_status="SIGNED_BY_PATIENT",
                             signature_quote="Patient signature: /s/ Pat Doe (electronically signed 03/20/2026 10:00)",
                             procedure_quote="inguinal hernia repair"),
        ],
        "history_and_physicals": [
            ExtractedHistoryAndPhysical(doc_ref="doc-hp", completion_status="COMPLETED",
                                        quote="Electronically signed by Dana Lee, MD on 03/10/2026."),
        ],
        "procedure_date_mentions": [],
        "procedure_risk_mentions": [],
    }


def run_pipeline(submission: dict[str, Any], extraction: dict[str, list], facility_timezone: str | None = None) -> TriageOutput:
    facts = preprocess(PatientSubmission.model_validate(submission), model="test-model", facility_timezone=facility_timezone)
    facts = verify_extraction(DocumentExtraction(**extraction), facts)
    return evaluate(facts)


def categories(output: TriageOutput) -> list[str]:
    return sorted({issue.category for issue in output.issues})


def check(output: TriageOutput, rule: str):
    assert output.audit is not None
    matches = [c for c in output.audit.rule_checks if c.rule == rule]
    assert matches, f"rule {rule} not in audit"
    return matches[0]


@pytest.fixture
def submission() -> dict[str, Any]:
    return copy.deepcopy(BASE_SUBMISSION)


@pytest.fixture
def extraction() -> dict[str, list]:
    return base_extraction()
