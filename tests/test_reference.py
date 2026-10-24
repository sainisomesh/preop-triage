"""Deterministic normalizers: document titles, lab codes, drug classes, quotes."""

from __future__ import annotations

import pytest

from reference import classify_document_title, classify_drug, locate_quote, normalize_lab, procedures_match


@pytest.mark.parametrize(
    "title",
    ["History and Physical", "History and Pyhsical", "Pre-op H and P", "H&P [PDF]", "Clinic H&P (external)",
     "History & Physical (scanned)", "Preoperative History & Physical (H&P)", "Histroy and Physcial"],
)
def test_hp_titles_including_typos(title: str) -> None:
    assert classify_document_title(title)[0] == "HISTORY_AND_PHYSICAL"


@pytest.mark.parametrize("title", ["Surgical Consent", "Consent - Elective Procedure", "Surgery Consent (scanned)", "Pre-op Consnet Discussion"])
def test_consent_titles(title: str) -> None:
    assert classify_document_title(title)[0] == "SURGICAL_CONSENT"


@pytest.mark.parametrize("title", ["Physical Therapy Evaluation", "Pre-anesthesia Evaluation", "Social Work Note", "Hematology Consultation"])
def test_other_titles(title: str) -> None:
    assert classify_document_title(title)[0] == "OTHER"


@pytest.mark.parametrize(
    ("code", "display", "expected"),
    [
        ("CBC", "Hemogram/CBC", "CBC"),
        ("LAB-CBC", "Complete Blood Count w/ Differential", "CBC"),
        ("LAB-CMP", "Comprehensive Metabolic Panel w/ Magnesium", "CMP"),
        ("BMP", "Basic Metabolic Panel", "BMP"),
        ("LAB-BMP", "Basic Metabolic Panel w/ eGFR", "BMP"),
        ("TSH", "Thyroid Stimulating Hormone", None),
        ("CMP", "Basic Metabolic Panel", None),  # contradictory -> left for a human
    ],
)
def test_lab_normalization_never_promotes_bmp_to_cmp(code, display, expected) -> None:
    assert normalize_lab(code, display)[0] == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Eliquis 5 mg PO BID", [("apixaban", "ANTICOAGULANT")]),
        ("Coumadin 5 mg PO daily", [("warfarin", "ANTICOAGULANT")]),
        ("edoxaban (Savaysa) 60 mg PO daily", [("edoxaban", "ANTICOAGULANT")]),
        ("Lovenox 40 mg SC daily", [("enoxaparin", "ANTICOAGULANT")]),
        ("ASA 81 mg PO daily", [("aspirin", "ANTIPLATELET")]),
        ("clopidogrel 75 mg PO daily", [("clopidogrel", "ANTIPLATELET")]),
        ("atorvastatin 40 mg PO nightly", []),
        ("Basaltic supplement", []),  # 'asa' must not match inside words
    ],
)
def test_drug_classification(name, expected) -> None:
    assert classify_drug(name) == expected


def test_procedure_matching_is_strict() -> None:
    assert procedures_match("Elective total hip arthroplasty", "total hip arthroplasty")
    assert not procedures_match("Elective thyroidectomy", "coronary artery bypass grafting")
    assert not procedures_match("Elective coronary artery bypass grafting", "Roux-en-Y gastric bypass")


def test_locate_quote_snaps_to_full_source_line() -> None:
    text = "Header\nVITAL SIGNS: Obtained at 08:30: BP 127/75 mmHg, Temperature 38.6 °C (oral).\nFooter"
    assert locate_quote(text, "temperature 38.6 °C") == (1, "VITAL SIGNS: Obtained at 08:30: BP 127/75 mmHg, Temperature 38.6 °C (oral).")
    assert locate_quote(text, "Temperature 39.6 °C") is None
