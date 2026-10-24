"""Offline tests for layer-1 verification and the layer-2 policy engine, one rule at a time."""

from __future__ import annotations

import pytest
from conftest import categories, check, run_pipeline

from extraction import (
    ExtractedAnticoagulationPlan,
    ExtractedConsent,
    ExtractedMedicationMention,
    ExtractedProcedureDateMention,
    ExtractedProcedureRiskMention,
    ExtractedVitalReading,
)


def _add_doc(submission, doc_id, title, date, text):
    submission["documents"].append({"doc_id": doc_id, "type": title, "date": date, "text": text})


def _nurse_vitals(extraction, bp: str | None, temp: str | None, quote: str, doc_id="doc-nurse", time="08:30"):
    readings = [r for r in extraction["vital_readings"] if r.doc_ref != doc_id]
    if bp:
        readings.append(ExtractedVitalReading(doc_ref=doc_id, measurement="BLOOD_PRESSURE", value_text=bp, unit="NOT_STATED",
                                              time_text=time, measured_date_text=None, context="MEASURED_AT_THIS_ENCOUNTER", quote=quote))
    if temp:
        readings.append(ExtractedVitalReading(doc_ref=doc_id, measurement="TEMPERATURE", value_text=temp, unit="NOT_STATED",
                                              time_text=time, measured_date_text=None, context="MEASURED_AT_THIS_ENCOUNTER", quote=quote))
    extraction["vital_readings"] = readings


# ---------------------------------------------------------------- baseline


def test_clean_submission_is_ready_with_full_audit(submission, extraction):
    output = run_pipeline(submission, extraction)
    assert output.decision == "READY"
    assert output.issues == []
    statuses = {c.rule: c.status for c in output.audit.rule_checks}
    assert statuses == {
        "R0.PROCEDURE_DATE": "PASS",
        "R0.PROCEDURE_RISK": "PASS",
        "R1.HISTORY_AND_PHYSICAL": "PASS",
        "R1.SURGICAL_CONSENT": "PASS",
        "R2.CBC": "PASS",
        "R2.CMP": "NOT_APPLICABLE",
        "R3.ANTICOAGULATION": "NOT_APPLICABLE",
        "R4.BLOOD_PRESSURE": "PASS",
        "R4.TEMPERATURE": "PASS",
    }


def test_output_is_byte_identical_for_identical_facts(submission, extraction):
    first = run_pipeline(submission, extraction).model_dump_json()
    second = run_pipeline(submission, extraction).model_dump_json()
    assert first == second


# ---------------------------------------------------------------- Rule 1


def test_hp_exactly_30_days_passes_and_31_fails(submission, extraction):
    submission["documents"][0]["date"] = "2026-03-01"  # 30 days before 03/31
    assert run_pipeline(submission, extraction).decision == "READY"

    submission["documents"][0]["date"] = "2026-02-28"  # 31 days
    output = run_pipeline(submission, extraction)
    assert output.decision == "NEEDS_FOLLOW_UP"
    assert categories(output) == ["REQUIRED_DOCUMENTATION"]
    assert "31 days" in check(output, "R1.HISTORY_AND_PHYSICAL").evidence.summary


def test_hp_title_typo_is_recognised(submission, extraction):
    submission["documents"][0]["type"] = "History and Pyhsical"
    assert run_pipeline(submission, extraction).decision == "READY"


def test_missing_hp_is_documentation_issue(submission, extraction):
    submission["documents"] = [d for d in submission["documents"] if d["doc_id"] != "doc-hp"]
    extraction["vital_readings"] = [r for r in extraction["vital_readings"] if r.doc_ref != "doc-hp"]
    extraction["history_and_physicals"] = []
    assert categories(run_pipeline(submission, extraction)) == ["REQUIRED_DOCUMENTATION"]


def test_consent_signed_for_a_different_procedure_fails(submission, extraction):
    submission["documents"][1]["text"] = submission["documents"][1]["text"].replace(
        "inguinal hernia repair", "coronary artery bypass grafting"
    )
    extraction["consents"] = [
        ExtractedConsent(doc_ref="doc-consent", patient_signature_status="SIGNED_BY_PATIENT",
                         signature_quote="Patient signature: /s/ Pat Doe (electronically signed 03/20/2026 10:00)",
                         procedure_quote="coronary artery bypass grafting")
    ]
    output = run_pipeline(submission, extraction)
    assert categories(output) == ["REQUIRED_DOCUMENTATION"]
    assert "coronary artery bypass grafting" in check(output, "R1.SURGICAL_CONSENT").evidence.summary


def test_extractor_claiming_signed_on_a_pending_line_is_overridden(submission, extraction):
    submission["documents"][1]["text"] = "INFORMED CONSENT\nProcedure: inguinal hernia repair\nPatient signature: pending"
    extraction["consents"] = [
        ExtractedConsent(doc_ref="doc-consent", patient_signature_status="SIGNED_BY_PATIENT",
                         signature_quote="Patient signature: pending", procedure_quote="inguinal hernia repair")
    ]
    assert categories(run_pipeline(submission, extraction)) == ["REQUIRED_DOCUMENTATION"]


def test_newer_signed_consent_supersedes_pending_one(submission, extraction):
    _add_doc(submission, "doc-consent-old", "Consent for Surgery", "2026-03-05",
             "Procedure: inguinal hernia repair\nPatient signature: pending")
    extraction["consents"].append(
        ExtractedConsent(doc_ref="doc-consent-old", patient_signature_status="NOT_SIGNED",
                         signature_quote="Patient signature: pending", procedure_quote="inguinal hernia repair")
    )
    assert run_pipeline(submission, extraction).decision == "READY"


# ---------------------------------------------------------------- Rule 2


def test_lab_codes_are_normalized_and_entered_in_error_is_skipped(submission, extraction):
    submission["labs"] = [
        {"id": "old", "code": "LAB-CBC", "display": "Complete Blood Count w/ Differential", "effective_at": "2026-03-10T08:00:00Z", "status": "final"},
        {"id": "bad", "code": "CBC", "display": "Hemogram/CBC", "effective_at": "2026-03-29T08:00:00Z", "status": "entered-in-error"},
    ]
    output = run_pipeline(submission, extraction)
    assert output.decision == "READY"
    assert "(old)" in check(output, "R2.CBC").evidence.summary
    assert any(e.source_id == "bad" for e in output.audit.excluded_evidence)


def test_only_non_final_cbc_means_missing_cbc(submission, extraction):
    submission["labs"][0]["status"] = "preliminary"
    assert categories(run_pipeline(submission, extraction)) == ["REQUIRED_TESTING"]


def test_high_risk_needs_cbc_and_cmp_within_14_days_and_bmp_is_not_cmp(submission, extraction):
    submission["procedure"]["procedure_risk"] = "HIGH"
    submission["labs"] = [
        {"id": "cbc", "code": "CBC", "display": "Complete Blood Count", "effective_at": "2026-03-17T08:00:00Z", "status": "final"},
        {"id": "bmp", "code": "LAB-BMP", "display": "Basic Metabolic Panel w/ eGFR", "effective_at": "2026-03-20T08:00:00Z", "status": "final"},
    ]
    output = run_pipeline(submission, extraction)  # CBC 14 days: ok; CMP missing
    assert categories(output) == ["REQUIRED_TESTING"]
    assert check(output, "R2.CBC").status == "PASS"
    assert check(output, "R2.CMP").status == "FAIL"

    submission["labs"][0]["effective_at"] = "2026-03-16T08:00:00Z"  # 15 days
    submission["labs"].append({"id": "cmp", "code": "LAB-CMP", "display": "Comprehensive Metabolic Panel w/ Magnesium", "effective_at": "2026-03-20T08:00:00Z", "status": "final"})
    output = run_pipeline(submission, extraction)
    assert check(output, "R2.CBC").status == "FAIL"
    assert check(output, "R2.CMP").status == "PASS"


# ---------------------------------------------------------------- Rule 0: date / risk reconciliation


def test_missing_procedure_date_reports_only_missing_data(submission, extraction):
    submission["procedure"]["procedure_date"] = None
    submission["documents"][0]["text"] = submission["documents"][0]["text"].replace(
        "(scheduled for 03/31/2026)", "(date to be determined pending OR availability)"
    )
    extraction["procedure_date_mentions"] = [
        ExtractedProcedureDateMention(doc_ref="doc-hp", status="NOT_YET_SCHEDULED", date_text=None,
                                      quote="Planned procedure: inguinal hernia repair (date to be determined pending OR availability)")
    ]
    output = run_pipeline(submission, extraction)
    assert output.decision == "NEEDS_FOLLOW_UP"
    assert categories(output) == ["MISSING_REQUIRED_DATA"]
    assert check(output, "R1.HISTORY_AND_PHYSICAL").status == "INDETERMINATE"
    assert check(output, "R2.CBC").status == "INDETERMINATE"


def test_missing_procedure_date_still_reports_tests_absent_under_any_date(submission, extraction):
    submission["procedure"]["procedure_date"] = None
    submission["labs"] = []
    assert categories(run_pipeline(submission, extraction)) == ["MISSING_REQUIRED_DATA", "REQUIRED_TESTING"]


def test_missing_procedure_date_falls_back_to_document_date(submission, extraction):
    submission["procedure"]["procedure_date"] = None
    extraction["procedure_date_mentions"] = [
        ExtractedProcedureDateMention(doc_ref="doc-hp", status="SCHEDULED", date_text="03/31/2026",
                                      quote="Planned procedure: inguinal hernia repair (scheduled for 03/31/2026)")
    ]
    output = run_pipeline(submission, extraction)
    assert output.decision == "READY"
    assert output.audit.procedure_date.source == "DOCUMENT"
    assert output.audit.procedure_date.value == "2026-03-31"


def test_missing_risk_evaluates_labs_against_every_risk_level(submission, extraction):
    submission["procedure"]["procedure_risk"] = None
    output = run_pipeline(submission, extraction)  # CBC 16 days: passes LOW/MOD, fails HIGH; CMP absent
    assert categories(output) == ["MISSING_REQUIRED_DATA"]
    assert check(output, "R2.CBC").status == "INDETERMINATE"

    submission["labs"] = []  # no CBC at all fails under every risk level
    assert categories(run_pipeline(submission, extraction)) == ["MISSING_REQUIRED_DATA", "REQUIRED_TESTING"]


def test_invalid_structured_risk_falls_back_to_explicit_document_risk(submission, extraction):
    submission["procedure"]["procedure_risk"] = "2026-03-31"  # date typed into the risk field
    _add_doc(submission, "doc-sched", "Surgical Scheduling Note", "2026-03-21",
             "Fall risk: low.\nProcedure risk category: MODERATE per surgical team.")
    extraction["procedure_risk_mentions"] = [
        ExtractedProcedureRiskMention(doc_ref="doc-sched", risk_category="LOW", quote="Fall risk: low."),
        ExtractedProcedureRiskMention(doc_ref="doc-sched", risk_category="MODERATE",
                                      quote="Procedure risk category: MODERATE per surgical team."),
    ]
    output = run_pipeline(submission, extraction)
    assert output.audit.procedure_risk.value == "MODERATE"
    assert output.audit.procedure_risk.source == "DOCUMENT"
    assert any("possible field swap" in note for note in output.audit.procedure_risk.notes)
    assert output.decision == "READY"


def test_document_risk_and_date_mentions_are_ignored_when_structured_fields_are_valid(submission, extraction):
    _add_doc(submission, "doc-sched", "Surgical Scheduling Note", "2026-03-21", "Procedure risk category: HIGH.")
    extraction["procedure_risk_mentions"] = [
        ExtractedProcedureRiskMention(doc_ref="doc-sched", risk_category="HIGH", quote="Procedure risk category: HIGH.")
    ]
    extraction["procedure_date_mentions"] = [
        ExtractedProcedureDateMention(doc_ref="doc-hp", status="NOT_YET_SCHEDULED", date_text=None,
                                      quote="Planned procedure: inguinal hernia repair (scheduled for 03/31/2026)")
    ]
    output = run_pipeline(submission, extraction)
    assert output.decision == "READY"
    assert output.audit.procedure_risk.value == "LOW"
    assert output.audit.procedure_risk.source == "STRUCTURED"
    assert output.audit.procedure_date.source == "STRUCTURED"
    assert output.audit.excluded_evidence == []


def test_short_document_refs_resolve_to_real_doc_ids(submission, extraction):
    # Documents sort by (date, doc_id): D1=doc-hp, D2=doc-consent, D3=doc-nurse.
    extraction["vital_readings"] = [r.model_copy(update={"doc_ref": {"doc-hp": "D1", "doc-nurse": "D3"}[r.doc_ref]})
                                    for r in extraction["vital_readings"]]
    extraction["consents"] = [c.model_copy(update={"doc_ref": "D2"}) for c in extraction["consents"]]
    extraction["history_and_physicals"] = [h.model_copy(update={"doc_ref": "D1"}) for h in extraction["history_and_physicals"]]
    output = run_pipeline(submission, extraction)
    assert output.decision == "READY"
    assert check(output, "R4.BLOOD_PRESSURE").evidence.source_ids == ["doc-nurse"]
    assert check(output, "R1.SURGICAL_CONSENT").evidence.source_ids == ["doc-consent"]


def test_unresolvable_doc_ref_is_rejected(submission, extraction):
    extraction["vital_readings"].append(
        ExtractedVitalReading(doc_ref="D9", measurement="BLOOD_PRESSURE", value_text="200/120", unit="mmHg", time_text=None,
                              measured_date_text=None, context="MEASURED_AT_THIS_ENCOUNTER", quote="BP 200/120")
    )
    output = run_pipeline(submission, extraction)
    # Unplaceable in time, so it can never be superseded: the chart goes to review.
    assert categories(output) == ["MISSING_REQUIRED_DATA"]
    assert any(e.source_id.startswith("unresolved-ref:D9") for e in output.audit.excluded_evidence)


# ---------------------------------------------------------------- Rule 3


def test_antiplatelets_never_require_a_plan(submission, extraction):
    submission["medications"] = [{"name": "ASA 81 mg PO daily", "active": True}, {"name": "clopidogrel 75 mg PO daily", "active": True}]
    output = run_pipeline(submission, extraction)
    assert output.decision == "READY"
    assert "not anticoagulants" in check(output, "R3.ANTICOAGULATION").evidence.summary


def _xarelto_plan_doc(submission, hold: bool, resume: bool):
    lines = ["Perioperative plan for Xarelto."]
    if hold:
        lines.append("Hold Xarelto for the 2 days before surgery (last dose 03/28/2026).")
    if resume:
        lines.append("Resume Xarelto 24-48 hours after surgery once hemostasis is secure.")
    _add_doc(submission, "doc-plan", "Cardiology Consultation", "2026-03-18", "\n".join(lines))
    return ExtractedAnticoagulationPlan(
        doc_ref="doc-plan", drug_as_written="Xarelto",
        hold_instruction_quote="Hold Xarelto for the 2 days before surgery (last dose 03/28/2026)." if hold else None,
        resume_instruction_quote="Resume Xarelto 24-48 hours after surgery once hemostasis is secure." if resume else None,
    )


def test_active_anticoagulant_with_hold_and_resume_passes_across_brand_and_generic(submission, extraction):
    submission["medications"] = [{"name": "rivaroxaban 20 mg PO daily", "active": True}]
    extraction["anticoagulation_plans"] = [_xarelto_plan_doc(submission, hold=True, resume=True)]
    output = run_pipeline(submission, extraction)
    assert output.decision == "READY"
    assert check(output, "R3.ANTICOAGULATION.RIVAROXABAN").status == "PASS"


def test_active_anticoagulant_without_resume_instruction_fails(submission, extraction):
    submission["medications"] = [{"name": "Xarelto 20 mg PO daily", "active": True}]
    extraction["anticoagulation_plans"] = [_xarelto_plan_doc(submission, hold=True, resume=False)]
    output = run_pipeline(submission, extraction)
    assert categories(output) == ["ANTICOAGULATION_MANAGEMENT"]
    assert "gives no resume instruction" in check(output, "R3.ANTICOAGULATION.RIVAROXABAN").evidence.summary


def test_plan_for_a_different_anticoagulant_does_not_count(submission, extraction):
    submission["medications"] = [{"name": "apixaban 5 mg PO BID", "active": True}]
    extraction["anticoagulation_plans"] = [_xarelto_plan_doc(submission, hold=True, resume=True)]
    output = run_pipeline(submission, extraction)
    assert categories(output) == ["ANTICOAGULATION_MANAGEMENT"]


def test_inactive_anticoagulant_is_ignored_unless_a_current_note_confirms_use(submission, extraction):
    submission["medications"] = [{"name": "warfarin 5 mg PO daily", "active": False}]
    assert run_pipeline(submission, extraction).decision == "READY"

    _add_doc(submission, "doc-pat", "Pre-Admission Testing Visit", "2026-03-29",
             "- Patient also reports taking Coumadin 5 mg daily from an outside cardiologist.")
    extraction["medication_mentions"] = [
        ExtractedMedicationMention(doc_ref="doc-pat", drug_as_written="Coumadin", class_hint="ANTICOAGULANT",
                                   status="CURRENTLY_TAKING",
                                   quote="- Patient also reports taking Coumadin 5 mg daily from an outside cardiologist.")
    ]
    assert categories(run_pipeline(submission, extraction)) == ["ANTICOAGULATION_MANAGEMENT"]


def test_unknown_anticoagulant_status_is_missing_data_not_anticoagulation_issue(submission, extraction):
    submission["medications"] = [{"name": "Coumadin 5 mg PO daily", "active": None}]
    _add_doc(submission, "doc-pat", "Pre-Admission Testing Visit", "2026-03-29",
             "- Coumadin: patient unable to confirm whether still taking; pharmacy fill history requested")
    extraction["medication_mentions"] = [
        ExtractedMedicationMention(doc_ref="doc-pat", drug_as_written="Coumadin", class_hint="ANTICOAGULANT",
                                   status="STATUS_UNCERTAIN",
                                   quote="- Coumadin: patient unable to confirm whether still taking; pharmacy fill history requested")
    ]
    output = run_pipeline(submission, extraction)
    assert categories(output) == ["MISSING_REQUIRED_DATA"]


def test_instruction_without_concrete_timing_is_not_a_plan(submission, extraction):
    submission["medications"] = [{"name": "dabigatran 150 mg PO BID", "active": True}]
    _add_doc(submission, "doc-plan", "Cardiology Progress Note", "2026-03-18",
             "Hold dabigatran as directed.\nResume dabigatran per surgical team.")
    extraction["anticoagulation_plans"] = [
        ExtractedAnticoagulationPlan(doc_ref="doc-plan", drug_as_written="dabigatran",
                                     hold_instruction_quote="Hold dabigatran as directed.",
                                     resume_instruction_quote="Resume dabigatran per surgical team.")
    ]
    assert categories(run_pipeline(submission, extraction)) == ["ANTICOAGULATION_MANAGEMENT"]


# ---------------------------------------------------------------- Rule 4


def test_diastolic_exactly_110_is_not_cleared(submission, extraction):
    submission["documents"][2]["text"] = "VITAL SIGNS: Obtained at 08:30: BP 148/110, Temp 98.4 F."
    _nurse_vitals(extraction, "148/110", "98.4", "VITAL SIGNS: Obtained at 08:30: BP 148/110, Temp 98.4 F.")
    output = run_pipeline(submission, extraction)
    assert output.decision == "NOT_CLEARED"
    assert categories(output) == ["ACUTE_SAFETY_EXCLUSION"]


def test_celsius_is_converted_and_100_4_is_not_a_fever(submission, extraction):
    submission["documents"][2]["text"] = "VITAL SIGNS: Obtained at 08:30: BP 124/76, Temp 38.0 °C (oral)."
    _nurse_vitals(extraction, "124/76", "38.0", "VITAL SIGNS: Obtained at 08:30: BP 124/76, Temp 38.0 °C (oral).")
    assert run_pipeline(submission, extraction).decision == "READY"  # 38.0 C == 100.4 F, not > 100.4

    submission["documents"][2]["text"] = "VITAL SIGNS: Obtained at 08:30: BP 124/76, Temp 38.1 °C (oral)."
    _nurse_vitals(extraction, "124/76", "38.1", "VITAL SIGNS: Obtained at 08:30: BP 124/76, Temp 38.1 °C (oral).")
    assert run_pipeline(submission, extraction).decision == "NOT_CLEARED"


def test_recheck_later_the_same_visit_supersedes_the_first_reading(submission, extraction):
    text = "08:06  NIBP 189/112, T 98.8 F.\nADDENDUM 08:30: Re-measured 137/78 mmHg after 24 minutes."
    submission["documents"][2]["text"] = text
    extraction["vital_readings"] = [r for r in extraction["vital_readings"] if r.doc_ref != "doc-nurse"] + [
        ExtractedVitalReading(doc_ref="doc-nurse", measurement="BLOOD_PRESSURE", value_text="189/112", unit="NOT_STATED",
                              time_text="08:06", measured_date_text=None, context="MEASURED_AT_THIS_ENCOUNTER", quote="08:06  NIBP 189/112, T 98.8 F."),
        ExtractedVitalReading(doc_ref="doc-nurse", measurement="TEMPERATURE", value_text="98.8", unit="F",
                              time_text="08:06", measured_date_text=None, context="MEASURED_AT_THIS_ENCOUNTER", quote="08:06  NIBP 189/112, T 98.8 F."),
        ExtractedVitalReading(doc_ref="doc-nurse", measurement="BLOOD_PRESSURE", value_text="137/78", unit="mmHg",
                              time_text="08:30", measured_date_text=None, context="MEASURED_AT_THIS_ENCOUNTER",
                              quote="ADDENDUM 08:30: Re-measured 137/78 mmHg after 24 minutes."),
    ]
    output = run_pipeline(submission, extraction)
    assert output.decision == "READY"
    assert "superseded" in check(output, "R4.BLOOD_PRESSURE").evidence.summary


def test_newer_abnormal_note_reading_overrides_older_normal_structured_vitals(submission, extraction):
    submission["documents"][2]["text"] = "09:05  BP 157/117, Temp 36.7 °C"
    _nurse_vitals(extraction, "157/117", "36.7", "09:05  BP 157/117, Temp 36.7 °C", time="09:05")
    assert run_pipeline(submission, extraction).decision == "NOT_CLEARED"


def test_same_day_structured_and_note_readings_use_the_most_severe(submission, extraction):
    submission["vitals"].append({"type": "blood_pressure", "systolic": 185, "diastolic": 90, "date": "2026-03-28T07:00:00Z", "source": "clinic"})
    output = run_pipeline(submission, extraction)
    assert output.decision == "NOT_CLEARED"
    assert "cannot be ordered reliably" in check(output, "R4.BLOOD_PRESSURE").evidence.summary


def test_temperature_absent_everywhere_is_missing_data(submission, extraction):
    submission["vitals"] = [v for v in submission["vitals"] if v["type"] != "temperature"]
    extraction["vital_readings"] = [r for r in extraction["vital_readings"] if r.measurement != "TEMPERATURE"]
    submission["documents"][0]["text"] = submission["documents"][0]["text"].replace(", T 98.1 F", "")
    extraction["vital_readings"] = [
        r if r.doc_ref != "doc-hp" else r.model_copy(update={"quote": "Vitals 10:05  BP 120/80, HR 70"})
        for r in extraction["vital_readings"]
    ]
    assert categories(run_pipeline(submission, extraction)) == ["MISSING_REQUIRED_DATA"]


# ---------------------------------------------------------------- anti-hallucination


def test_fabricated_vital_quote_is_rejected_and_sent_to_review(submission, extraction):
    extraction["vital_readings"].append(
        ExtractedVitalReading(doc_ref="doc-nurse", measurement="BLOOD_PRESSURE", value_text="200/120", unit="mmHg",
                              time_text="09:00", measured_date_text=None, context="MEASURED_AT_THIS_ENCOUNTER",
                              quote="BP 200/120 on recheck")
    )
    output = run_pipeline(submission, extraction)
    # The fabricated value is never used (no NOT_CLEARED) and never silently ignored (no READY).
    assert output.decision == "NEEDS_FOLLOW_UP"
    assert categories(output) == ["MISSING_REQUIRED_DATA"]
    assert check(output, "R4.BLOOD_PRESSURE").status == "INDETERMINATE"
    assert any("not found verbatim" in e.reason for e in output.audit.excluded_evidence)


def test_rejected_reading_superseded_by_a_later_verified_reading_does_not_block(submission, extraction):
    # A reading claimed at 07:55 fails verification, but the note's verified 08:30 reading is later.
    submission["documents"][2]["text"] = "Arrival 07:55.\nVITAL SIGNS: Obtained at 08:30: BP 124/76, Temp 98.4 F, HR 72."
    extraction["vital_readings"].append(
        ExtractedVitalReading(doc_ref="doc-nurse", measurement="BLOOD_PRESSURE", value_text="210/130", unit="mmHg",
                              time_text="07:55", measured_date_text=None, context="MEASURED_AT_THIS_ENCOUNTER",
                              quote="BP 210/130 at arrival")
    )
    output = run_pipeline(submission, extraction)
    assert output.decision == "READY"
    assert check(output, "R4.BLOOD_PRESSURE").status == "PASS"
    assert any("not found verbatim" in e.reason for e in output.audit.excluded_evidence)  # still logged


def test_value_not_in_its_own_quote_is_rejected(submission, extraction):
    extraction["vital_readings"].append(
        ExtractedVitalReading(doc_ref="doc-nurse", measurement="TEMPERATURE", value_text="101.9", unit="F",
                              time_text="08:31", measured_date_text=None, context="MEASURED_AT_THIS_ENCOUNTER",
                              quote="VITAL SIGNS: Obtained at 08:30: BP 124/76, Temp 98.4 F, HR 72.")
    )
    output = run_pipeline(submission, extraction)
    assert categories(output) == ["MISSING_REQUIRED_DATA"]  # 08:31 is after the verified 08:30 reading
    assert any("not present in quoted text" in e.reason for e in output.audit.excluded_evidence)


def test_every_issue_carries_rule_summary_ids_and_verbatim_quotes(submission, extraction):
    submission["documents"][2]["text"] = "VITAL SIGNS: Obtained at 08:30: BP 124/76, Temp 38.6 °C (oral)."
    _nurse_vitals(extraction, "124/76", "38.6", "VITAL SIGNS: Obtained at 08:30: BP 124/76, Temp 38.6 °C (oral).")
    output = run_pipeline(submission, extraction)
    (issue,) = output.issues
    assert issue.evidence.rule == "R4.TEMPERATURE"
    assert issue.evidence.source_ids == ["doc-nurse"]
    assert issue.evidence.quotes[0].verbatim is True
    assert issue.evidence.quotes[0].text in submission["documents"][2]["text"]


def test_anticoagulation_evidence_cites_every_drug_line_even_if_extractor_skipped_it(submission, extraction):
    submission["medications"] = [{"name": "dabigatran (Pradaxa) 150 mg PO BID", "active": True}]
    deferral = "1. Periprocedural management of dabigatran to be determined by the surgical team; happy to assist if needed."
    _add_doc(submission, "doc-cards", "Cardiology Progress Note", "2026-03-14", f"Recommendations:\n{deferral}")
    output = run_pipeline(submission, extraction)  # extractor reported no mentions or plans
    assert categories(output) == ["ANTICOAGULATION_MANAGEMENT"]
    quotes = [q.text for q in check(output, "R3.ANTICOAGULATION.DABIGATRAN").evidence.quotes]
    assert deferral in quotes


# ---------------------------------------------------------------- production edge cases


def test_antiplatelet_plan_is_skipped_quietly_but_unknown_drug_plan_is_logged(submission, extraction):
    submission["medications"] = [{"name": "clopidogrel 75 mg PO daily", "active": True}]
    _add_doc(submission, "doc-cards", "Cardiology Progress Note", "2026-03-16",
             "2. Hold clopidogrel 5 days before surgery; resume the day after surgery if hemostasis is adequate.\n"
             "3. Hold Zorbitran 2 days before surgery; resume on postoperative day 1.")
    extraction["anticoagulation_plans"] = [
        ExtractedAnticoagulationPlan(doc_ref="doc-cards", drug_as_written="clopidogrel",
                                     hold_instruction_quote="Hold clopidogrel 5 days before surgery",
                                     resume_instruction_quote="resume the day after surgery if hemostasis is adequate."),
        ExtractedAnticoagulationPlan(doc_ref="doc-cards", drug_as_written="Zorbitran",
                                     hold_instruction_quote="Hold Zorbitran 2 days before surgery",
                                     resume_instruction_quote="resume on postoperative day 1."),
    ]
    output = run_pipeline(submission, extraction)
    assert output.decision == "READY"
    reasons = [(e.source_id, e.reason) for e in output.audit.excluded_evidence]
    assert reasons == [("doc-cards:Zorbitran", "plan drug 'Zorbitran' is not a recognized anticoagulant")]


@pytest.mark.parametrize(("status", "accepted"), [("final", True), ("amended", True), ("corrected", True), (" Final ", True),
                                                  ("preliminary", False), ("registered", False), ("cancelled", False),
                                                  ("entered-in-error", False), ("unknown", False), (None, False)])
def test_lab_status_acceptance(submission, extraction, status, accepted):
    submission["labs"][0]["status"] = status
    output = run_pipeline(submission, extraction)
    assert check(output, "R2.CBC").status == ("PASS" if accepted else "FAIL")


def _same_day_utc_vs_local(submission, extraction, structured_utc: str, structured_bp=(189, 112)):
    submission["vitals"].append({"type": "blood_pressure", "systolic": structured_bp[0], "diastolic": structured_bp[1],
                                 "date": structured_utc, "source": "clinic"})
    _nurse_vitals(extraction, "137/78", "98.4", "VITAL SIGNS: Obtained at 08:30: BP 137/78, Temp 98.4 F, HR 72.")
    submission["documents"][2]["text"] = "Pre-op nursing intake.\nVITAL SIGNS: Obtained at 08:30: BP 137/78, Temp 98.4 F, HR 72."


def test_later_local_recheck_beats_earlier_utc_high_reading_when_timezone_known(submission, extraction):
    # 13:00Z on 2026-03-28 is 08:00 CDT, i.e. before the 08:30 note recheck.
    _same_day_utc_vs_local(submission, extraction, "2026-03-28T13:00:00Z")
    output = run_pipeline(submission, extraction, facility_timezone="America/Chicago")
    assert output.decision == "READY"
    summary = check(output, "R4.BLOOD_PRESSURE").evidence.summary
    assert "137/78" in summary and "superseded" in summary
    assert output.audit.facility_timezone == "America/Chicago"


def test_later_utc_high_reading_beats_earlier_local_reading_when_timezone_known(submission, extraction):
    # 14:00Z is 09:00 CDT, i.e. after the 08:30 note reading.
    _same_day_utc_vs_local(submission, extraction, "2026-03-28T14:00:00Z")
    assert run_pipeline(submission, extraction, facility_timezone="America/Chicago").decision == "NOT_CLEARED"


def test_utc_vs_local_without_timezone_is_conservative_and_explained(submission, extraction):
    _same_day_utc_vs_local(submission, extraction, "2026-03-28T13:00:00Z")
    output = run_pipeline(submission, extraction)
    assert output.decision == "NOT_CLEARED"
    assert "configure the facility timezone" in check(output, "R4.BLOOD_PRESSURE").evidence.summary


def test_timezone_conversion_moves_a_lab_across_the_window_boundary(submission, extraction):
    # 03:00Z on 03/01 is 21:00 CST on 02/28: 30 days before 03/31 in UTC, 31 days locally.
    submission["labs"][0]["effective_at"] = "2026-03-01T03:00:00Z"
    assert check(run_pipeline(submission, extraction), "R2.CBC").status == "PASS"
    assert check(run_pipeline(submission, extraction, facility_timezone="America/Chicago"), "R2.CBC").status == "FAIL"


def test_unknown_facility_timezone_fails_fast(submission, extraction):
    with pytest.raises(ValueError, match="unknown facility timezone"):
        run_pipeline(submission, extraction, facility_timezone="Mars/Olympus_Mons")


def test_untimed_recheck_later_in_the_same_note_wins_by_text_order(submission, extraction):
    text = "Arrival vitals: BP 186/104, Temp 98.6 F.\nRecheck after rest: BP 132/80."
    submission["documents"][2]["text"] = text
    extraction["vital_readings"] = [r for r in extraction["vital_readings"] if r.doc_ref != "doc-nurse"] + [
        ExtractedVitalReading(doc_ref="doc-nurse", measurement="BLOOD_PRESSURE", value_text="186/104", unit="NOT_STATED", time_text=None,
                              measured_date_text=None, context="MEASURED_AT_THIS_ENCOUNTER", quote="Arrival vitals: BP 186/104, Temp 98.6 F."),
        ExtractedVitalReading(doc_ref="doc-nurse", measurement="BLOOD_PRESSURE", value_text="132/80", unit="NOT_STATED", time_text=None,
                              measured_date_text=None, context="MEASURED_AT_THIS_ENCOUNTER", quote="Recheck after rest: BP 132/80."),
        ExtractedVitalReading(doc_ref="doc-nurse", measurement="TEMPERATURE", value_text="98.6", unit="F", time_text=None,
                              measured_date_text=None, context="MEASURED_AT_THIS_ENCOUNTER", quote="Arrival vitals: BP 186/104, Temp 98.6 F."),
    ]
    assert run_pipeline(submission, extraction).decision == "READY"


@pytest.mark.parametrize(
    ("note_line", "status"),
    [
        ("NPO instructions: nothing to eat after midnight; clear liquids until 2 hours before arrival. Patient verbalized understanding.", "REVIEWED_WITH_PATIENT"),
        ("NPO instructions: nothing to eat after midnight.", "INSTRUCTIONS_DOCUMENTED"),
        ("NPO instructions not reviewed - patient left early.", "NOT_REVIEWED"),
        ("Fasting instructions deferred to anesthesia.", "NOT_REVIEWED"),
        ("Ride home arranged with son.", "NOT_DOCUMENTED"),
    ],
)
def test_npo_review_is_informational_only(submission, extraction, note_line, status):
    baseline = run_pipeline(submission, extraction)
    submission["documents"][2]["text"] += f"\n{note_line}"
    output = run_pipeline(submission, extraction)
    (info,) = output.audit.informational_checks
    assert info.check == "INFO.NPO_FASTING_REVIEW"
    assert info.status == status
    assert info.affects_decision is False
    assert (output.decision, output.issues) == (baseline.decision, baseline.issues)
    assert all(not c.rule.startswith("INFO") for c in output.audit.rule_checks)


def test_npo_review_without_nursing_note(submission, extraction):
    submission["documents"][2]["type"] = "Clinic Follow-up Note"
    (info,) = run_pipeline(submission, extraction).audit.informational_checks
    assert info.status == "NO_NURSING_INTAKE_NOTE"
