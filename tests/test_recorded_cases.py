"""Replay recorded real extractions (tests/fixtures) through verification + policy offline.

Fixtures come from tests/fixtures/record_extractions.py. If the prompt or schema
changes, re-record them; the prompt-version check below flags stale fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from extraction import (
    EXTRACTION_PROMPT_VERSION,
    DocumentExtraction,
    ExtractedAnticoagulationPlan,
    ExtractedMedicationMention,
    preprocess,
    verify_extraction,
)
from policy_engine import evaluate
from schemas import PatientSubmission

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
CASES = {json.loads(line)["case_id"]: json.loads(line) for line in (ROOT / "data" / "patients_sample_50.jsonl").read_text().splitlines() if line.strip()}


def _replay(case_id: str, mutate=None):
    fixture = json.loads((FIXTURES / f"extraction_{case_id}.json").read_text())
    extraction = DocumentExtraction.model_validate(fixture["extraction"])
    if mutate:
        extraction = mutate(extraction)
    case = CASES[case_id]
    facts = preprocess(PatientSubmission.model_validate(case["submission"]), model=fixture["model"])
    return case, fixture, evaluate(verify_extraction(extraction, facts))


@pytest.mark.parametrize("case_id", ["case_00016", "case_00025"])
def test_recorded_case_matches_label(case_id):
    case, fixture, output = _replay(case_id)
    assert fixture["prompt_version"] == EXTRACTION_PROMPT_VERSION, "fixture recorded with an older prompt; re-record"
    assert output.decision == case["label"]["decision"]
    assert sorted({i.category for i in output.issues}) == sorted(case["label"]["categories"])


def test_case_16_recheck_in_note_supersedes_earlier_high_reading():
    _, _, output = _replay("case_00016")
    bp = next(c for c in output.audit.rule_checks if c.rule == "R4.BLOOD_PRESSURE")
    assert bp.status == "PASS"
    assert "137/78" in bp.evidence.summary and "08:30" in bp.evidence.summary
    assert "189/112" in bp.evidence.summary and "superseded" in bp.evidence.summary
    # Same result when structured times are converted to a facility timezone.
    facts = preprocess(PatientSubmission.model_validate(CASES["case_00016"]["submission"]), model="m", facility_timezone="America/New_York")
    fixture = json.loads((FIXTURES / "extraction_case_00016.json").read_text())
    tz_output = evaluate(verify_extraction(DocumentExtraction.model_validate(fixture["extraction"]), facts))
    assert tz_output.decision == "READY"


def test_case_25_antiplatelet_over_reporting_does_not_change_output():
    """Observed live: the model sometimes reports 'dual antiplatelet therapy' as aspirin/clopidogrel
    mentions and the aspirin 'continue' line as a plan. None of it may alter the audit."""

    _, _, clean = _replay("case_00025")
    doc_ref = "D3"

    def add_noise(extraction: DocumentExtraction) -> DocumentExtraction:
        noisy = extraction.model_copy(deep=True)
        noisy.medication_mentions += [
            ExtractedMedicationMention(doc_ref=doc_ref, drug_as_written="aspirin", class_hint="ANTIPLATELET", status="CURRENTLY_TAKING",
                                       quote="on dual antiplatelet therapy."),
            ExtractedMedicationMention(doc_ref=doc_ref, drug_as_written="clopidogrel", class_hint="ANTIPLATELET", status="CURRENTLY_TAKING",
                                       quote="on dual antiplatelet therapy."),
        ]
        noisy.anticoagulation_plans += [
            ExtractedAnticoagulationPlan(doc_ref=doc_ref, drug_as_written="aspirin", hold_instruction_quote=None,
                                         resume_instruction_quote="1. Continue aspirin 81 mg daily without interruption through surgery."),
        ]
        return noisy

    _, _, noisy = _replay("case_00025", add_noise)
    assert noisy.model_dump_json() == clean.model_dump_json()
