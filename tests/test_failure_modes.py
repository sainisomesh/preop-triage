"""Subtle failure modes: hallucinated quotes, mixed C/F strings, amended labs, brand vs generic."""

from __future__ import annotations

import pytest
from conftest import categories, check, run_pipeline

from extraction import (
    ExtractedAnticoagulationPlan,
    ExtractedConsent,
    ExtractedMedicationMention,
    ExtractedVitalReading,
)


def _add_doc(submission, doc_id, title, date, text):
    submission["documents"].append({"doc_id": doc_id, "type": title, "date": date, "text": text})


def _set_nurse_temperature(submission, extraction, line: str, value: str, unit: str = "NOT_STATED"):
    # Replace the whole nursing note, so drop every reading extracted from the old text
    # (a stale reading would now fail verification and correctly trigger review).
    submission["documents"][2]["text"] = f"Pre-op nursing intake.\n{line}"
    extraction["vital_readings"] = [r for r in extraction["vital_readings"] if r.doc_ref != "doc-nurse"]
    extraction["vital_readings"].append(
        ExtractedVitalReading(doc_ref="doc-nurse", measurement="TEMPERATURE", value_text=value, unit=unit, time_text="08:30",
                              measured_date_text=None, context="MEASURED_AT_THIS_ENCOUNTER", quote=line)
    )


# ---------------------------------------------------------------- hallucinated quotes


def test_hallucinated_signature_line_does_not_make_a_consent_valid(submission, extraction):
    submission["documents"][1]["text"] = "INFORMED CONSENT\nProcedure: inguinal hernia repair\n[Form printed for patient review; not yet signed]"
    extraction["consents"] = [
        ExtractedConsent(doc_ref="doc-consent", patient_signature_status="SIGNED_BY_PATIENT",
                         signature_quote="Patient signature: /s/ Pat Doe (electronically signed 03/20/2026 10:00)",
                         procedure_quote="inguinal hernia repair")
    ]
    output = run_pipeline(submission, extraction)
    assert categories(output) == ["REQUIRED_DOCUMENTATION"]
    assert "signature quote not found verbatim" in check(output, "R1.SURGICAL_CONSENT").evidence.summary


def test_near_miss_plan_quote_with_altered_timing_is_rejected(submission, extraction):
    submission["medications"] = [{"name": "Xarelto 20 mg PO daily", "active": True}]
    _add_doc(submission, "doc-plan", "Cardiology Consultation", "2026-03-18",
             "Hold Xarelto for the 2 days before surgery.\nResume Xarelto per surgical team.")
    extraction["anticoagulation_plans"] = [
        ExtractedAnticoagulationPlan(doc_ref="doc-plan", drug_as_written="Xarelto",
                                     hold_instruction_quote="Hold Xarelto for the 2 days before surgery.",
                                     resume_instruction_quote="Resume Xarelto 24 hours after surgery."),  # not in the note
    ]
    output = run_pipeline(submission, extraction)
    assert categories(output) == ["ANTICOAGULATION_MANAGEMENT"]
    assert any("resume instruction quote not found verbatim" in e.reason for e in output.audit.excluded_evidence)


def test_quote_stitched_with_ellipsis_is_rejected_and_sent_to_review(submission, extraction):
    extraction["vital_readings"].append(
        ExtractedVitalReading(doc_ref="doc-nurse", measurement="BLOOD_PRESSURE", value_text="190/115", unit="mmHg", time_text=None,
                              measured_date_text=None, context="MEASURED_AT_THIS_ENCOUNTER", quote="VITAL SIGNS: ... BP 190/115")
    )
    output = run_pipeline(submission, extraction)
    # Never used as a verified reading (no NOT_CLEARED), never silently dropped (no READY).
    assert output.decision == "NEEDS_FOLLOW_UP"
    assert categories(output) == ["MISSING_REQUIRED_DATA"]
    assert any("not found verbatim" in e.reason for e in output.audit.excluded_evidence)


def test_hallucinated_stopped_statement_cannot_clear_an_active_anticoagulant(submission, extraction):
    submission["medications"] = [{"name": "apixaban 5 mg PO BID", "active": True}]
    extraction["medication_mentions"] = [
        ExtractedMedicationMention(doc_ref="doc-nurse", drug_as_written="apixaban", class_hint="ANTICOAGULANT", status="STOPPED",
                                   quote="Patient stopped apixaban last week."),
    ]
    output = run_pipeline(submission, extraction)
    assert categories(output) == ["ANTICOAGULATION_MANAGEMENT"]  # still treated as taking; no plan on file


def test_quote_from_the_wrong_document_is_rejected(submission, extraction):
    # The line exists in the nursing note, but the extractor attributes it to the H&P.
    extraction["vital_readings"].append(
        ExtractedVitalReading(doc_ref="doc-hp", measurement="BLOOD_PRESSURE", value_text="124/76", unit="NOT_STATED", time_text="08:30",
                              measured_date_text=None, context="MEASURED_AT_THIS_ENCOUNTER",
                              quote="VITAL SIGNS: Obtained at 08:30: BP 124/76, Temp 98.4 F, HR 72.")
    )
    output = run_pipeline(submission, extraction)
    assert any(e.source_id.startswith("doc-hp:") for e in output.audit.excluded_evidence)


# ---------------------------------------------------------------- mixed C / F strings


@pytest.mark.parametrize(
    ("line", "value", "extractor_unit", "expected_decision"),
    [
        ("Temp 38.6 °C (oral)", "38.6", "NOT_STATED", "NOT_CLEARED"),  # 101.48 F
        ("T 37.1C", "37.1", "NOT_STATED", "READY"),
        ("Temp: 38.6° C", "38.6", "NOT_STATED", "NOT_CLEARED"),
        ("Temperature 38.5 Celsius", "38.5", "NOT_STATED", "NOT_CLEARED"),
        ("Temp 38.5 ℃", "38.5", "NOT_STATED", "NOT_CLEARED"),  # unicode degree-Celsius sign
        ("Temp 37.9 deg C", "37.9", "NOT_STATED", "READY"),  # 100.22 F
        ("Temp 100.9°F", "100.9", "NOT_STATED", "NOT_CLEARED"),
        ("Temp 101.2 F", "101.2", "C", "NOT_CLEARED"),  # extractor says C; written F wins
        ("Temp 38.2 C", "38.2", "F", "NOT_CLEARED"),  # extractor says F; written C wins (100.76 F)
        ("Temp 38.0 °C", "38.0", "NOT_STATED", "READY"),  # exactly 100.4 F is not > 100.4
        ("Temp 37.9", "37.9", "NOT_STATED", "READY"),  # no unit: Celsius by range
        ("Temp 100.6", "100.6", "NOT_STATED", "NOT_CLEARED"),  # no unit: Fahrenheit by range
    ],
)
def test_mixed_temperature_units(submission, extraction, line, value, extractor_unit, expected_decision):
    _set_nurse_temperature(submission, extraction, line, value, extractor_unit)
    assert run_pipeline(submission, extraction).decision == expected_decision


def test_temperature_with_no_unit_and_implausible_value_is_rejected(submission, extraction):
    _set_nurse_temperature(submission, extraction, "Temp 58.1", "58.1")
    output = run_pipeline(submission, extraction)
    assert any("no unit and is implausible" in e.reason for e in output.audit.excluded_evidence)
    # The older structured temperature is cited but not trusted as the latest: review, not READY.
    assert check(output, "R4.TEMPERATURE").status == "INDETERMINATE"
    assert check(output, "R4.TEMPERATURE.UNVERIFIED_READING").evidence.quotes[0].text == "Temp 58.1"
    assert categories(output) == ["MISSING_REQUIRED_DATA"]


def test_unit_override_is_recorded_as_a_warning(submission, extraction):
    _set_nurse_temperature(submission, extraction, "Temp 101.2 F", "101.2", "C")
    output = run_pipeline(submission, extraction)
    assert any("overridden by unit written in text (F)" in w for w in output.audit.warnings)


# ---------------------------------------------------------------- amended / corrected labs


def test_newer_amended_cbc_supersedes_an_older_out_of_window_final(submission, extraction):
    submission["labs"] = [
        {"id": "old-final", "code": "CBC", "display": "Complete Blood Count", "effective_at": "2026-02-01T08:00:00Z", "status": "final"},
        {"id": "amended", "code": "CBC", "display": "Complete Blood Count", "effective_at": "2026-03-20T08:00:00Z", "status": "amended"},
    ]
    output = run_pipeline(submission, extraction)
    assert check(output, "R2.CBC").status == "PASS"
    assert "(amended)" in check(output, "R2.CBC").evidence.summary


def test_corrected_result_is_used_but_only_the_most_recent_result_counts(submission, extraction):
    submission["labs"] = [
        {"id": "recent-but-preliminary", "code": "CBC", "display": "CBC", "effective_at": "2026-03-25T08:00:00Z", "status": "preliminary"},
        {"id": "corrected-old", "code": "LAB-CBC", "display": "Complete Blood Count", "effective_at": "2026-02-15T08:00:00Z", "status": "corrected"},
    ]
    output = run_pipeline(submission, extraction)  # corrected result is 44 days out
    assert categories(output) == ["REQUIRED_TESTING"]
    assert any(e.source_id == "recent-but-preliminary" for e in output.audit.excluded_evidence)


def test_amended_lab_dated_after_review_time_is_excluded(submission, extraction):
    submission["labs"].append(
        {"id": "future", "code": "CBC", "display": "CBC", "effective_at": "2026-04-02T08:00:00Z", "status": "amended"}
    )
    output = run_pipeline(submission, extraction)
    assert "(lab-cbc)" in check(output, "R2.CBC").evidence.summary
    assert any(e.source_id == "future" and "after review time" in e.reason for e in output.audit.excluded_evidence)


# ---------------------------------------------------------------- brand vs generic


def _plan_doc(submission, drug_in_note: str):
    text = (f"Hold {drug_in_note} for the 2 days before surgery (last dose 03/28/2026).\n"
            f"Resume {drug_in_note} 48 hours after surgery if hemostasis is adequate.")
    _add_doc(submission, "doc-consult", "Hematology Consultation", "2026-03-18", text)
    return ExtractedAnticoagulationPlan(
        doc_ref="doc-consult", drug_as_written=drug_in_note,
        hold_instruction_quote=f"Hold {drug_in_note} for the 2 days before surgery (last dose 03/28/2026).",
        resume_instruction_quote=f"Resume {drug_in_note} 48 hours after surgery if hemostasis is adequate.",
    )


@pytest.mark.parametrize(
    ("med_list_name", "consult_name", "expected"),
    [
        ("Eliquis 5 mg PO BID", "apixaban", "READY"),
        ("apixaban 5 mg PO BID", "Eliquis", "READY"),
        ("warfarin 5 mg PO daily", "Coumadin", "READY"),
        ("Jantoven 5 mg PO daily", "warfarin", "READY"),
        ("Lovenox 40 mg SC daily", "enoxaparin", "READY"),
        ("Savaysa 60 mg PO daily", "edoxaban", "READY"),
        ("Pradaxa 150 mg PO BID", "dabigatran", "READY"),
        ("Xarelto 20 mg PO daily", "Eliquis", "NEEDS_FOLLOW_UP"),  # plan is for a different drug
        ("Coumadin 5 mg PO daily", "Pradaxa", "NEEDS_FOLLOW_UP"),
    ],
)
def test_brand_and_generic_names_are_reconciled_between_med_list_and_consult(submission, extraction, med_list_name, consult_name, expected):
    submission["medications"] = [{"name": med_list_name, "active": True}]
    extraction["anticoagulation_plans"] = [_plan_doc(submission, consult_name)]
    output = run_pipeline(submission, extraction)
    assert output.decision == expected
    if expected == "NEEDS_FOLLOW_UP":
        assert categories(output) == ["ANTICOAGULATION_MANAGEMENT"]


def test_brand_in_note_confirms_inactive_generic_on_med_list(submission, extraction):
    submission["medications"] = [{"name": "warfarin 5 mg PO daily", "active": False}]
    _add_doc(submission, "doc-pat", "Pre-Admission Testing Visit", "2026-03-29", "- Patient reports taking Coumadin 5 mg daily.")
    extraction["medication_mentions"] = [
        ExtractedMedicationMention(doc_ref="doc-pat", drug_as_written="Coumadin", class_hint="ANTICOAGULANT",
                                   status="CURRENTLY_TAKING", quote="- Patient reports taking Coumadin 5 mg daily.")
    ]
    output = run_pipeline(submission, extraction)
    assert categories(output) == ["ANTICOAGULATION_MANAGEMENT"]
    warfarin = check(output, "R3.ANTICOAGULATION.WARFARIN")
    assert {q.text for q in warfarin.evidence.quotes} >= {"- Patient reports taking Coumadin 5 mg daily."}


def test_switch_between_agents_requires_a_plan_for_the_new_one(submission, extraction):
    submission["medications"] = [
        {"name": "warfarin 5 mg PO daily", "active": False},
        {"name": "Eliquis 5 mg PO BID", "active": True},
    ]
    extraction["anticoagulation_plans"] = [_plan_doc(submission, "Coumadin")]  # plan predates the switch
    output = run_pipeline(submission, extraction)
    assert categories(output) == ["ANTICOAGULATION_MANAGEMENT"]
    summary = check(output, "R3.ANTICOAGULATION.APIXABAN").evidence.summary
    assert "plans on file only address ['warfarin']" in summary
    assert check(output, "R3.ANTICOAGULATION.WARFARIN").status == "NOT_APPLICABLE"


# ---------------------------------------------------------------- rule-alignment regressions


def test_post_procedure_hp_and_lab_do_not_hide_valid_pre_op_ones(submission, extraction):
    # Review time after the procedure (late upload) lets post-op documents through preprocessing.
    submission["metadata"]["submission_received_at"] = "2026-04-05T12:00:00Z"
    _add_doc(submission, "doc-hp-postop", "History and Physical", "2026-04-02", "Post-op H&P.\nElectronically signed by Dana Lee, MD.")
    submission["labs"].append({"id": "lab-postop", "code": "CBC", "display": "CBC", "effective_at": "2026-04-02T08:00:00Z", "status": "final"})
    output = run_pipeline(submission, extraction)
    assert check(output, "R1.HISTORY_AND_PHYSICAL").status == "PASS"
    assert "doc-hp" in check(output, "R1.HISTORY_AND_PHYSICAL").evidence.summary
    assert "(lab-cbc)" in check(output, "R2.CBC").evidence.summary
    assert any("is before the review time" in w for w in output.audit.warnings)


def test_only_post_procedure_hp_fails(submission, extraction):
    submission["metadata"]["submission_received_at"] = "2026-04-05T12:00:00Z"
    submission["documents"][0]["date"] = "2026-04-02"
    output = run_pipeline(submission, extraction)
    assert check(output, "R1.HISTORY_AND_PHYSICAL").status == "FAIL"
    assert "dated after the 2026-03-31 procedure" in check(output, "R1.HISTORY_AND_PHYSICAL").evidence.summary


def test_consent_procedure_falls_back_to_the_forms_procedure_line(submission, extraction):
    extraction["consents"] = [c.model_copy(update={"procedure_quote": None}) for c in extraction["consents"]]
    output = run_pipeline(submission, extraction)
    assert check(output, "R1.SURGICAL_CONSENT").status == "PASS"
    assert "Procedure: inguinal hernia repair" in [q.text for q in check(output, "R1.SURGICAL_CONSENT").evidence.quotes]


def test_fractional_structured_bp_is_not_truncated_into_a_pass(submission, extraction):
    # 179.6 would truncate to 179; BP is recorded in whole mmHg, so the value goes to review.
    submission["vitals"].append({"type": "blood_pressure", "systolic": 179.6, "diastolic": 80, "date": "2026-03-30T10:00:00Z", "source": "clinic"})
    output = run_pipeline(submission, extraction)
    assert categories(output) == ["MISSING_REQUIRED_DATA"]


@pytest.mark.parametrize("corrupted", ["36.7 \x176C", "36.7 ºC", "36.7˚C", "36.7 \\u00b0C", "36.7 C"])
def test_degree_sign_corruption_in_quotes_still_verifies(submission, extraction, corrupted):
    # gpt-4.1-mini emitted U+0017 + "6" for the degree sign; the source line itself is clean.
    line = "09:05  BP 157/117, Temp 36.7 °C, HR 69, SpO2 99% RA"
    submission["documents"][2]["text"] = f"Nursing pre-procedure assessment.\n{line}"
    quote = line.replace("36.7 °C", corrupted)
    extraction["vital_readings"] = [r for r in extraction["vital_readings"] if r.doc_ref != "doc-nurse"] + [
        ExtractedVitalReading(doc_ref="doc-nurse", measurement="BLOOD_PRESSURE", value_text="157/117", unit="mmHg", time_text="09:05",
                              measured_date_text=None, context="MEASURED_AT_THIS_ENCOUNTER", quote=quote),
        ExtractedVitalReading(doc_ref="doc-nurse", measurement="TEMPERATURE", value_text="36.7", unit="C", time_text="09:05",
                              measured_date_text=None, context="MEASURED_AT_THIS_ENCOUNTER", quote=quote),
    ]
    output = run_pipeline(submission, extraction)
    assert output.decision == "NOT_CLEARED"
    assert categories(output) == ["ACUTE_SAFETY_EXCLUSION"]
    assert check(output, "R4.BLOOD_PRESSURE").evidence.quotes[0].text == line  # evidence is the clean source line


# ---------------------------------------------------------------- procedure synonyms, laterality, AI suggestions


@pytest.mark.parametrize(
    ("scheduled", "consented", "matches"),
    [
        ("Elective total hip arthroplasty", "right hip replacement", True),
        ("Elective coronary artery bypass grafting", "CABG x3", True),
        ("Elective inguinal hernia repair", "left inguinal herniorrhaphy", True),
        ("Elective Roux-en-Y gastric bypass", "RYGB", True),
        ("Elective total knee arthroplasty", "TKA", True),
        ("Elective left knee arthroscopy", "right knee arthroscopy", False),  # wrong side
        ("Elective right total knee arthroplasty", "bilateral total knee arthroplasty", False),
        ("Elective thyroidectomy", "coronary artery bypass grafting", False),
        ("Elective coronary artery bypass grafting", "Roux-en-Y gastric bypass", False),
        ("Elective total hip arthroplasty", "total knee arthroplasty", False),
    ],
)
def test_procedure_synonyms_and_laterality(scheduled, consented, matches):
    from reference import procedure_match_detail

    assert procedure_match_detail(scheduled, consented)[0] is matches


def test_wrong_side_consent_is_a_documentation_issue_with_a_clear_reason(submission, extraction):
    submission["procedure"]["procedure_type"] = "Elective left inguinal hernia repair"
    submission["documents"][1]["text"] = submission["documents"][1]["text"].replace("Procedure: inguinal hernia repair", "Procedure: right inguinal hernia repair")
    extraction["consents"] = [c.model_copy(update={"procedure_quote": "right inguinal hernia repair"}) for c in extraction["consents"]]
    output = run_pipeline(submission, extraction)
    assert categories(output) == ["REQUIRED_DOCUMENTATION"]
    assert "laterality conflict" in check(output, "R1.SURGICAL_CONSENT").evidence.summary


def _unknown_synonym_case(submission, extraction):
    submission["procedure"]["procedure_type"] = "Elective anterior cruciate ligament repair"
    submission["documents"][1]["text"] = submission["documents"][1]["text"].replace("Procedure: inguinal hernia repair", "Procedure: ACL reconstruction")
    extraction["consents"] = [c.model_copy(update={"procedure_quote": "ACL reconstruction"}) for c in extraction["consents"]]


def test_ai_suggestion_is_informational_and_never_passes_the_consent(submission, extraction):
    from extraction import preprocess, suggest_procedure_matches, verify_extraction, DocumentExtraction
    from policy_engine import evaluate
    from schemas import PatientSubmission

    _unknown_synonym_case(submission, extraction)
    calls = []

    def fake_matcher(scheduled, consented):
        calls.append((scheduled, consented))
        return True, "ACL reconstruction is the standard repair of the anterior cruciate ligament.", "fake matcher"

    facts = verify_extraction(DocumentExtraction(**extraction), preprocess(PatientSubmission.model_validate(submission), model="m"))
    suggest_procedure_matches(facts, fake_matcher)
    output = evaluate(facts)

    assert calls == [("Elective anterior cruciate ligament repair", "ACL reconstruction")]
    assert categories(output) == ["REQUIRED_DOCUMENTATION"]  # still fails closed
    info = [i for i in output.audit.informational_checks if i.check == "INFO.PROCEDURE_EQUIVALENCE"]
    assert len(info) == 1 and info[0].status == "AI_SUGGESTS_SAME_PROCEDURE" and info[0].affects_decision is False


def test_ai_matcher_is_not_called_when_names_already_match_and_errors_are_contained(submission, extraction):
    from extraction import preprocess, suggest_procedure_matches, verify_extraction, DocumentExtraction
    from schemas import PatientSubmission

    def exploding_matcher(scheduled, consented):
        raise TimeoutError("model down")

    facts = verify_extraction(DocumentExtraction(**extraction), preprocess(PatientSubmission.model_validate(submission), model="m"))
    suggest_procedure_matches(facts, exploding_matcher)  # names match: never called
    assert facts.procedure_suggestions == [] and facts.warnings == []

    _unknown_synonym_case(submission, extraction)
    facts = verify_extraction(DocumentExtraction(**extraction), preprocess(PatientSubmission.model_validate(submission), model="m"))
    suggest_procedure_matches(facts, exploding_matcher)
    assert facts.procedure_suggestions == []
    assert any("procedure matcher failed" in w for w in facts.warnings)


def test_openai_procedure_matcher_request_shape():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from ai_procedure_matcher import ProcedureEquivalence, openai_procedure_matcher

    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(output_parsed=ProcedureEquivalence(same_procedure=False, rationale="Different joint."))
    same, rationale, source = openai_procedure_matcher("gpt-5.6-luna", client=client)("total hip arthroplasty", "knee replacement")
    call = client.responses.parse.call_args.kwargs
    assert (same, rationale, source) == (False, "Different joint.", "gpt-5.6-luna procedure matcher")
    assert call["temperature"] == 0 and call["store"] is False and call["reasoning"] == {"effort": "none"}
    assert call["text_format"] is ProcedureEquivalence
    openai_procedure_matcher("gpt-4.1-mini", client=client)("a", "b")
    assert "reasoning" not in client.responses.parse.call_args.kwargs
