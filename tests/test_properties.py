"""Property-based tests (Hypothesis) for policy-engine invariants.

Cases are built relative to an arbitrary procedure date (2000-2099), so the date
arithmetic is exercised across month ends, leap days and year boundaries with the stdlib
`datetime` types the engine uses. Everything runs offline.
"""

from __future__ import annotations

import copy
from datetime import date, timedelta
from decimal import Decimal

from conftest import BASE_SUBMISSION, base_extraction, categories, check, run_pipeline
from hypothesis import example, given, settings
from hypothesis import strategies as st

from extraction import ExtractedVitalReading

SETTINGS = settings(deadline=None)  # example count comes from the profile in conftest.py
procedure_dates = st.dates(min_value=date(2000, 1, 1), max_value=date(2099, 12, 31))
risks = st.sampled_from(["LOW", "MODERATE", "HIGH"])
# Thresholds are oversampled: uniform draws alone rarely land exactly on 180 / 110 / 100.4,
# and a mutation check showed a 180 -> 181 change slipping through without this.
systolic = st.one_of(st.sampled_from([179, 180, 181]), st.integers(min_value=70, max_value=260))
diastolic = st.one_of(st.sampled_from([109, 110, 111]), st.integers(min_value=40, max_value=160))
temperature_tenths_f = st.one_of(st.sampled_from([1003, 1004, 1005]), st.integers(min_value=950, max_value=1069))  # x0.1 F


def make_case(proc: date, *, risk: str = "LOW", hp_offset: int = 10, labs=(("CBC", 5),), vitals=None, keep_note_readings=False):
    """A clean case whose every date is `proc - offset days`; review time is the procedure date."""

    def on(offset: int) -> str:
        return (proc - timedelta(days=offset)).isoformat()

    submission = copy.deepcopy(BASE_SUBMISSION)
    submission["procedure"].update(procedure_date=proc.isoformat(), procedure_risk=risk)
    submission["metadata"]["submission_received_at"] = f"{proc.isoformat()}T12:00:00Z"
    submission["documents"][0]["date"] = on(hp_offset)
    submission["documents"][1]["date"] = on(5)
    submission["documents"][2]["date"] = on(2)
    submission["labs"] = [
        {"id": f"lab-{i}", "code": code, "display": code, "effective_at": f"{on(offset)}T08:00:00Z", "status": "final"}
        for i, (code, offset) in enumerate(labs)
    ]
    if vitals is not None:
        submission["vitals"] = [
            ({"type": "blood_pressure", "systolic": value[0], "diastolic": value[1]} if kind == "bp"
             else {"type": "temperature", "value_f": value}) | {"date": f"{on(offset)}T{hhmm}:00Z", "source": "clinic"}
            for kind, value, offset, hhmm in vitals
        ]
    else:
        for vital in submission["vitals"]:
            vital["date"] = f"{on(3)}T10:00:00Z"
    extraction = base_extraction()
    if not keep_note_readings:
        extraction["vital_readings"] = []
    return submission, extraction


def _bp_or_temp_excluded(sbp: int, dbp: int, temp_f: Decimal) -> bool:
    return sbp >= 180 or dbp >= 110 or temp_f > Decimal("100.4")


# ---------------------------------------------------------------- Rule 1 / Rule 2 windows


@SETTINGS
@given(proc=procedure_dates, offset=st.integers(min_value=-5, max_value=60))
def test_hp_passes_exactly_within_0_to_30_days(proc, offset):
    submission, extraction = make_case(proc, hp_offset=offset)
    status = check(run_pipeline(submission, extraction), "R1.HISTORY_AND_PHYSICAL").status
    assert status == ("PASS" if 0 <= offset <= 30 else "FAIL")


@SETTINGS
@given(proc=procedure_dates, risk=risks, offset=st.integers(min_value=-5, max_value=60))
def test_cbc_window_is_30_days_or_14_for_high_risk(proc, risk, offset):
    submission, extraction = make_case(proc, risk=risk, labs=(("CBC", offset), ("CMP", 1)))
    output = run_pipeline(submission, extraction)
    window = 14 if risk == "HIGH" else 30
    assert check(output, "R2.CBC").status == ("PASS" if 0 <= offset <= window else "FAIL")
    assert ("REQUIRED_TESTING" in categories(output)) == (not 0 <= offset <= window)


@SETTINGS
@given(proc=procedure_dates, risk=risks, offset=st.integers(min_value=-5, max_value=60))
def test_cmp_is_required_within_14_days_only_for_high_risk(proc, risk, offset):
    submission, extraction = make_case(proc, risk=risk, labs=(("CBC", 1), ("CMP", offset)))
    status = check(run_pipeline(submission, extraction), "R2.CMP").status
    if risk == "HIGH":
        assert status == ("PASS" if 0 <= offset <= 14 else "FAIL")
    else:
        assert status == "NOT_APPLICABLE"


@SETTINGS
@given(proc=procedure_dates, offset=st.integers(min_value=0, max_value=60))
def test_bmp_never_satisfies_a_cmp_requirement(proc, offset):
    submission, extraction = make_case(proc, risk="HIGH", labs=(("CBC", 1), ("BMP", offset)))
    assert check(run_pipeline(submission, extraction), "R2.CMP").status == "FAIL"


# ---------------------------------------------------------------- Rule 4 exclusions


@SETTINGS
@given(proc=procedure_dates, sbp=systolic, dbp=diastolic, tenths=temperature_tenths_f)
@example(proc=date(2026, 3, 31), sbp=180, dbp=80, tenths=984)   # SBP exactly 180 -> excluded
@example(proc=date(2026, 3, 31), sbp=179, dbp=80, tenths=984)   # just below -> not excluded
@example(proc=date(2026, 3, 31), sbp=148, dbp=110, tenths=984)  # DBP exactly 110 -> excluded
@example(proc=date(2026, 3, 31), sbp=148, dbp=109, tenths=984)
@example(proc=date(2026, 3, 31), sbp=128, dbp=76, tenths=1004)  # 100.4 F is not > 100.4
@example(proc=date(2026, 3, 31), sbp=128, dbp=76, tenths=1005)
def test_latest_verified_vitals_decide_not_cleared_exactly_at_the_thresholds(proc, sbp, dbp, tenths):
    if dbp >= sbp:
        dbp = sbp - 10
    temp_f = Decimal(tenths) / 10
    submission, extraction = make_case(
        proc, vitals=[("bp", (122, 78), 10, "09:00"), ("temp", 98.4, 10, "09:00"),
                      ("bp", (sbp, dbp), 1, "09:00"), ("temp", float(temp_f), 1, "09:05")])
    output = run_pipeline(submission, extraction)
    excluded = _bp_or_temp_excluded(sbp, dbp, temp_f)
    assert ("ACUTE_SAFETY_EXCLUSION" in categories(output)) == excluded
    assert (output.decision == "NOT_CLEARED") == excluded
    # Same facts, same bytes.
    assert run_pipeline(submission, extraction).model_dump_json() == output.model_dump_json()


@SETTINGS
@given(proc=procedure_dates, sbp=st.integers(min_value=180, max_value=260), dbp=st.integers(min_value=110, max_value=160))
def test_older_abnormal_reading_never_overrides_a_newer_normal_one(proc, sbp, dbp):
    submission, extraction = make_case(
        proc, vitals=[("bp", (sbp, min(dbp, sbp - 1)), 4, "09:00"), ("temp", 101.9, 4, "09:00"),
                      ("bp", (128, 76), 1, "09:00"), ("temp", 98.2, 1, "09:05")])
    output = run_pipeline(submission, extraction)
    assert "ACUTE_SAFETY_EXCLUSION" not in categories(output)
    assert output.decision == "READY"


@SETTINGS
@given(proc=procedure_dates, tenths_c=st.integers(min_value=300, max_value=450))
@example(proc=date(2026, 3, 31), tenths_c=380)  # 38.0 C == 100.4 F exactly
@example(proc=date(2026, 3, 31), tenths_c=381)
def test_celsius_note_readings_convert_exactly_before_the_threshold(proc, tenths_c):
    celsius = Decimal(tenths_c) / 10
    line = f"VITAL SIGNS: Obtained at 08:30: BP 124/76, Temp {celsius} °C (oral)."
    submission, extraction = make_case(proc)
    submission["documents"][2]["text"] = f"Pre-op nursing intake.\n{line}"
    extraction["vital_readings"] = [
        ExtractedVitalReading(doc_ref="doc-nurse", measurement="TEMPERATURE", value_text=str(celsius), unit="C", time_text="08:30",
                              measured_date_text=None, context="MEASURED_AT_THIS_ENCOUNTER", quote=line),
    ]
    output = run_pipeline(submission, extraction)
    assert (output.decision == "NOT_CLEARED") == (celsius * 9 / 5 + 32 > Decimal("100.4"))


# ---------------------------------------------------------------- fail-closed verification


@SETTINGS
@given(
    proc=procedure_dates,
    measurement=st.sampled_from(["BLOOD_PRESSURE", "TEMPERATURE"]),
    value=st.sampled_from(["210/130", "126/80", "101.9", "98.6", "38.9"]),
    junk=st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=0, max_size=20),
    time=st.sampled_from([None, "23:59", "08:31"]),
)
def test_rejected_latest_vital_never_resolves_to_ready(proc, measurement, value, junk, time):
    submission, extraction = make_case(proc)  # verified structured vitals are 3 days old
    # A reading in the newest note whose quote is not in the note: it must not be used and must not be ignored.
    extraction["vital_readings"] = [
        ExtractedVitalReading(doc_ref="doc-nurse", measurement=measurement, value_text=value, unit="NOT_STATED", time_text=time,
                              measured_date_text=None, context="MEASURED_AT_THIS_ENCOUNTER", quote=f"QZX {junk} {value}")
    ]
    output = run_pipeline(submission, extraction)
    assert output.decision != "READY"
    assert "MISSING_REQUIRED_DATA" in categories(output)
    assert "ACUTE_SAFETY_EXCLUSION" not in categories(output)  # an unverified value is never acted on


@SETTINGS
@given(proc=procedure_dates, value=st.sampled_from(["210/130", "96/60"]))
def test_rejected_reading_superseded_by_a_newer_verified_one_is_not_blocking(proc, value):
    submission, extraction = make_case(proc, hp_offset=10)
    submission["vitals"] = [
        {"type": "blood_pressure", "systolic": 124, "diastolic": 76, "date": f"{(proc - timedelta(days=1)).isoformat()}T09:00:00Z", "source": "clinic"},
        {"type": "temperature", "value_f": 98.4, "date": f"{(proc - timedelta(days=1)).isoformat()}T09:05:00Z", "source": "clinic"},
    ]
    # The rejected reading sits in the H&P, 10 days before the procedure: older than the verified vitals.
    extraction["vital_readings"] = [
        ExtractedVitalReading(doc_ref="doc-hp", measurement="BLOOD_PRESSURE", value_text=value, unit="mmHg", time_text=None,
                              measured_date_text=None, context="MEASURED_AT_THIS_ENCOUNTER", quote=f"QZX {value}")
    ]
    output = run_pipeline(submission, extraction)
    assert output.decision == "READY"
    assert all(not c.rule.endswith("UNVERIFIED_READING") for c in output.audit.rule_checks)


@SETTINGS
@given(proc=procedure_dates, hp_offset=st.integers(-5, 60), cbc_offset=st.integers(-5, 60), risk=risks,
       sbp=systolic, dbp=diastolic, tenths=temperature_tenths_f)
def test_decision_is_always_consistent_with_issue_categories(proc, hp_offset, cbc_offset, risk, sbp, dbp, tenths):
    if dbp >= sbp:
        dbp = sbp - 10
    submission, extraction = make_case(proc, risk=risk, hp_offset=hp_offset, labs=(("CBC", cbc_offset), ("CMP", 1)),
                                       vitals=[("bp", (sbp, dbp), 1, "09:00"), ("temp", tenths / 10, 1, "09:05")])
    output = run_pipeline(submission, extraction)
    cats = set(categories(output))
    if "ACUTE_SAFETY_EXCLUSION" in cats:
        assert output.decision == "NOT_CLEARED"
    elif cats:
        assert output.decision == "NEEDS_FOLLOW_UP"
    else:
        assert output.decision == "READY"
    assert all(c.status != "INDETERMINATE" for c in output.audit.rule_checks) or "MISSING_REQUIRED_DATA" in cats
