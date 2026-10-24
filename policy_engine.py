"""Layer 2: applies the pre-op policy to verified CaseFacts from extraction.py.

No model calls, randomness or clock reads, so identical facts give byte-identical output.
Every rule writes a RuleCheck to the audit; each FAIL with a category becomes an issue.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Callable, Iterable

from extraction import (
    EXTRACTION_PROMPT_VERSION,
    CaseFacts,
    MedicationMention,
    UnverifiedVital,
    VitalRecord,
)
from reference import (
    CBC_WINDOW_DAYS_STANDARD,
    DIASTOLIC_EXCLUSION_MMHG,
    HIGH_RISK_LAB_WINDOW_DAYS,
    HP_WINDOW_DAYS,
    POLICY_VERSION,
    PROCEDURE_RISKS,
    SYSTOLIC_EXCLUSION_MMHG,
    NPO_ACKNOWLEDGED_PATTERN,
    NPO_NOT_REVIEWED_PATTERN,
    NPO_PATTERN,
    TEMPERATURE_EXCLUSION_F,
    ProcedureRisk,
    classify_drug,
    is_nursing_intake_title,
    normalize_for_match,
    parse_iso_date,
    procedures_match,
    structured_procedure_date,
    structured_procedure_risk,
)
from schemas import (
    AuditTrail,
    Decision,
    Evidence,
    EvidenceQuote,
    InformationalCheck,
    IssueCategory,
    ResolvedProcedureField,
    RuleCheck,
    RuleStatus,
    TriageIssue,
    TriageOutput,
)

PIPELINE_VERSION = "2.1.0"  # 2.1.0: fail-closed unverified vitals, degree-sign normalization

# =====================================================================
# Helpers
# =====================================================================


def _sorted_quotes(quotes: Iterable[EvidenceQuote | None]) -> list[EvidenceQuote]:
    unique: dict[tuple[str, str, str], EvidenceQuote] = {}
    for quote in quotes:
        if quote is not None:
            unique.setdefault((quote.source_type, quote.source_id, quote.text), quote)
    return sorted(unique.values(), key=lambda q: (q.date or "", q.source_type, q.source_id, q.text))


def _check(
    rule: str,
    status: RuleStatus,
    summary: str,
    quotes: Iterable[EvidenceQuote | None] = (),
    category: IssueCategory | None = None,
) -> RuleCheck:
    ordered = _sorted_quotes(quotes)
    return RuleCheck(
        rule=rule,
        status=status,
        issue_category=category if status == "FAIL" else None,
        evidence=Evidence(
            rule=rule,
            summary=summary,
            source_ids=sorted({quote.source_id for quote in ordered}),
            quotes=ordered,
        ),
    )


def _doc_ref(doc_id: str, doc_date: date, title: str | None) -> EvidenceQuote:
    return EvidenceQuote(
        source_type="document", source_id=doc_id, date=doc_date.isoformat(),
        text=f"document {title!r} dated {doc_date.isoformat()}", verbatim=False,
    )


def _related_lines(facts: CaseFacts, pattern: re.Pattern[str], limit: int = 3) -> list[EvidenceQuote]:
    # Evidence for absences ("no consent on file") cites nearby lines such as
    # "consent to be obtained by surgical team" so reviewers see why.

    found: list[EvidenceQuote] = []
    for doc in reversed(facts.documents):  # most recent first
        for line in doc.text.splitlines():
            if pattern.search(line):
                found.append(
                    EvidenceQuote(source_type="document", source_id=doc.doc_id, date=doc.date.isoformat(),
                                  text=line.strip(), verbatim=True)
                )
                if len(found) >= limit:
                    return found
    return found


def _days(earlier: date, later: date) -> int:
    return (later - earlier).days


# =====================================================================
# Procedure date / risk reconciliation
# =====================================================================


@dataclass(frozen=True)
class ResolvedProcedure:
    procedure_date: date | None
    procedure_risk: ProcedureRisk | None
    date_field: ResolvedProcedureField
    risk_field: ResolvedProcedureField


def _structured_quote(field_name: str, raw: str | None) -> EvidenceQuote:
    return EvidenceQuote(source_type="procedure", source_id=f"procedure.{field_name}", date=None,
                         text=f"procedure.{field_name} = {raw!r}", verbatim=False)


def _resolve_from_documents(mentions: list, value_of: Callable) -> tuple[object | None, list[EvidenceQuote], str]:
    # Only reached when the structured field is unusable. Latest mentioning document wins
    # (handles rescheduling); "not yet assigned" or conflicting values stay unresolved.

    if not mentions:
        return None, [], "no document states a value"
    latest_date = max(mention.doc_date for mention in mentions)
    latest = [mention for mention in mentions if mention.doc_date == latest_date]
    values = {value_of(mention) for mention in latest}
    quotes = [mention.evidence for mention in latest]
    if len(values) == 1 and None not in values:
        return values.pop(), quotes, f"most recent mention ({latest_date.isoformat()}) gives a single value"
    if None in values and len(values) == 1:
        return None, quotes, f"most recent mention ({latest_date.isoformat()}) says it is not yet assigned/scheduled"
    return None, quotes, f"most recent mentions ({latest_date.isoformat()}) are conflicting"


def resolve_procedure(facts: CaseFacts) -> ResolvedProcedure:
    # ---- date
    raw_date = facts.structured_procedure_date
    date_notes: list[str] = []
    parsed_date = structured_procedure_date(raw_date)
    if parsed_date is not None:
        others = sorted({m.scheduled_for for m in facts.procedure_date_mentions if m.scheduled_for and m.scheduled_for != parsed_date})
        if others:
            message = f"documents also mention scheduled date(s) {[d.isoformat() for d in others]}; structured value kept"
            date_notes.append(message)
            facts.warnings.append(f"procedure_date discrepancy: {message}")
        date_field = ResolvedProcedureField(
            field="procedure_date", value=parsed_date.isoformat(), source="STRUCTURED", structured_raw=raw_date,
            evidence=[_structured_quote("procedure_date", raw_date)], notes=date_notes,
        )
    else:
        if raw_date is None or not str(raw_date).strip():
            date_notes.append("structured procedure_date is missing")
        else:
            date_notes.append(f"structured procedure_date {raw_date!r} is not a valid date")
            if str(raw_date).strip().upper() in PROCEDURE_RISKS:
                date_notes.append("value looks like a procedure risk category (possible field swap)")
        value, quotes, note = _resolve_from_documents(facts.procedure_date_mentions, lambda m: m.scheduled_for)
        date_notes.append(f"document fallback: {note}")
        parsed_date = value  # type: ignore[assignment]
        date_field = ResolvedProcedureField(
            field="procedure_date", value=parsed_date.isoformat() if parsed_date else None,
            source="DOCUMENT" if parsed_date else "UNRESOLVED", structured_raw=raw_date,
            evidence=_sorted_quotes([_structured_quote("procedure_date", raw_date), *quotes]), notes=date_notes,
        )

    # ---- risk
    raw_risk = facts.structured_procedure_risk
    risk_notes: list[str] = []
    risk = structured_procedure_risk(raw_risk)
    if risk is not None:
        if raw_risk != risk:
            risk_notes.append(f"structured value {raw_risk!r} normalized to {risk}")
        others = sorted({m.risk for m in facts.procedure_risk_mentions if m.risk and m.risk != risk})
        if others:
            message = f"documents also mention risk category {others}; structured value kept"
            risk_notes.append(message)
            facts.warnings.append(f"procedure_risk discrepancy: {message}")
        risk_field = ResolvedProcedureField(
            field="procedure_risk", value=risk, source="STRUCTURED", structured_raw=raw_risk,
            evidence=[_structured_quote("procedure_risk", raw_risk)], notes=risk_notes,
        )
    else:
        if raw_risk is None or not str(raw_risk).strip():
            risk_notes.append("structured procedure_risk is missing")
        else:
            risk_notes.append(f"structured procedure_risk {raw_risk!r} is not one of {list(PROCEDURE_RISKS)}")
            if parse_iso_date(str(raw_risk)) is not None:
                risk_notes.append("value looks like a date (possible field swap)")
        value, quotes, note = _resolve_from_documents(facts.procedure_risk_mentions, lambda m: m.risk)
        risk_notes.append(f"document fallback: {note}")
        risk = value  # type: ignore[assignment]
        risk_field = ResolvedProcedureField(
            field="procedure_risk", value=risk, source="DOCUMENT" if risk else "UNRESOLVED", structured_raw=raw_risk,
            evidence=_sorted_quotes([_structured_quote("procedure_risk", raw_risk), *quotes]), notes=risk_notes,
        )

    if parsed_date is not None and facts.review_date is not None and parsed_date < facts.review_date:
        facts.warnings.append(
            f"procedure date {parsed_date.isoformat()} is before the review time {facts.review_time}; check for a data-entry error"
        )
    return ResolvedProcedure(parsed_date, risk, date_field, risk_field)


# =====================================================================
# Rule 0: data needed to evaluate other rules
# =====================================================================


def check_procedure_fields(procedure: ResolvedProcedure) -> list[RuleCheck]:
    checks: list[RuleCheck] = []
    for field_info, label, consequence in (
        (procedure.date_field, "Procedure date", "H&P and lab windows cannot be evaluated"),
        (procedure.risk_field, "Procedure risk", "lab requirements are evaluated against every risk level and only results failing under all of them are reported"),
    ):
        rule = f"R0.{field_info.field.upper()}"
        if field_info.value is not None:
            checks.append(_check(rule, "PASS", f"{label} {field_info.value} (source: {field_info.source.lower()}).", field_info.evidence))
        else:
            checks.append(
                _check(rule, "FAIL", f"{label} unknown ({'; '.join(field_info.notes)}); {consequence}.",
                       field_info.evidence, "MISSING_REQUIRED_DATA")
            )
    return checks


# =====================================================================
# Rule 1: documentation
# =====================================================================

_HP_MENTION = re.compile(r"\bH&P\b|history and physical|history & physical", re.IGNORECASE)
_CONSENT_MENTION = re.compile(r"consent", re.IGNORECASE)


def check_history_and_physical(facts: CaseFacts, procedure_date: date | None) -> RuleCheck:
    rule = "R1.HISTORY_AND_PHYSICAL"
    records = facts.history_and_physicals
    eligible = [hp for hp in records if hp.status != "INCOMPLETE"]
    incomplete = [hp for hp in records if hp.status == "INCOMPLETE"]

    if not eligible:
        quotes = [hp.evidence for hp in incomplete] or _related_lines(facts, _HP_MENTION)
        detail = f"{len(incomplete)} H&P document(s) marked incomplete" if incomplete else "no document is titled as an H&P"
        return _check(rule, "FAIL", f"No completed History and Physical on file ({detail}).", quotes, "REQUIRED_DOCUMENTATION")

    # Only H&Ps on or before the procedure count; a later one cannot satisfy a pre-op rule
    # and must not hide an earlier valid one.
    pre_op = [hp for hp in eligible if procedure_date is None or hp.date <= procedure_date]
    if not pre_op:
        after = max(eligible, key=lambda hp: (hp.date, hp.doc_id))
        return _check(rule, "FAIL", f"Every completed H&P is dated after the {procedure_date.isoformat()} procedure "
                      f"(most recent {after.title!r}, {after.doc_id}, {after.date.isoformat()}).",
                      [after.evidence or _doc_ref(after.doc_id, after.date, after.title)], "REQUIRED_DOCUMENTATION")
    latest = max(pre_op, key=lambda hp: (hp.date, hp.doc_id))
    quotes = [latest.evidence or _doc_ref(latest.doc_id, latest.date, latest.title)]
    described = f"Most recent H&P {latest.title!r} ({latest.doc_id}) dated {latest.date.isoformat()}"
    if procedure_date is None:
        return _check(rule, "INDETERMINATE", f"{described}; the {HP_WINDOW_DAYS}-day window cannot be checked without a procedure date.", quotes)

    days = _days(latest.date, procedure_date)
    if days <= HP_WINDOW_DAYS:
        return _check(rule, "PASS", f"{described} is {days} days before the {procedure_date.isoformat()} procedure (limit {HP_WINDOW_DAYS}).", quotes)
    return _check(rule, "FAIL", f"{described} is {days} days before the {procedure_date.isoformat()} procedure, "
                  f"outside the {HP_WINDOW_DAYS}-day window.", quotes, "REQUIRED_DOCUMENTATION")


def check_surgical_consent(facts: CaseFacts) -> list[RuleCheck]:
    rule = "R1.SURGICAL_CONSENT"
    if not facts.consents:
        return [
            _check(rule, "FAIL", "No surgical consent document on file.", _related_lines(facts, _CONSENT_MENTION), "REQUIRED_DOCUMENTATION")
        ]

    if not facts.procedure_type:
        return [
            _check(rule, "FAIL", "Scheduled procedure type is missing, so no consent can be matched to it.",
                   [c.signature for c in facts.consents], "MISSING_REQUIRED_DATA")
        ]

    valid = []
    problems: list[str] = []
    problem_quotes: list[EvidenceQuote | None] = []
    for consent in sorted(facts.consents, key=lambda c: (c.date, c.doc_id)):
        label = f"{consent.title!r} ({consent.doc_id}, {consent.date.isoformat()})"
        if consent.status != "SIGNED_BY_PATIENT":
            state = "not signed by the patient" if consent.status == "NOT_SIGNED" else "patient signature cannot be confirmed"
            extra = f" [{'; '.join(consent.notes)}]" if consent.notes else ""
            problems.append(f"{label} {state}{extra}")
            problem_quotes += [consent.signature or _doc_ref(consent.doc_id, consent.date, consent.title)]
        elif not consent.procedure_name:
            problems.append(f"{label} does not name a procedure")
            problem_quotes += [consent.signature]
        elif not procedures_match(facts.procedure_type, consent.procedure_name):
            named = consent.procedure.text if consent.procedure else consent.procedure_name
            problems.append(f"{label} names {named!r}, which does not match the scheduled procedure {facts.procedure_type!r}")
            problem_quotes += [consent.procedure, consent.signature]
        else:
            valid.append(consent)

    if valid:
        best = max(valid, key=lambda c: (c.date, c.doc_id))
        summary = (f"Patient-signed consent {best.title!r} ({best.doc_id}, {best.date.isoformat()}) names a procedure "
                   f"matching the scheduled {facts.procedure_type!r}.")
        if problems:
            summary += f" Superseded/other consent(s): {'; '.join(problems)}."
        return [_check(rule, "PASS", summary, [best.signature, best.procedure])]

    return [
        _check(rule, "FAIL", f"No valid patient-signed consent for {facts.procedure_type!r}: {'; '.join(problems)}.",
               problem_quotes, "REQUIRED_DOCUMENTATION")
    ]


# =====================================================================
# Rule 2: testing
# =====================================================================


def lab_window_days(test: str, risk: ProcedureRisk) -> int | None:
    if test == "CBC":
        return HIGH_RISK_LAB_WINDOW_DAYS if risk == "HIGH" else CBC_WINDOW_DAYS_STANDARD
    if test == "CMP":
        return HIGH_RISK_LAB_WINDOW_DAYS if risk == "HIGH" else None
    raise ValueError(test)


def check_lab(facts: CaseFacts, test: str, procedure: ResolvedProcedure) -> RuleCheck:
    rule = f"R2.{test}"
    # Unknown risk: evaluate under every level and FAIL only if the test fails under all
    # of them (e.g. no CBC at all); otherwise INDETERMINATE, covered by the R0 missing-risk issue.
    candidates: tuple[ProcedureRisk, ...] = (procedure.procedure_risk,) if procedure.procedure_risk else PROCEDURE_RISKS
    # "Most recent result" means most recent pre-procedure result; a post-procedure lab is not
    # a pre-op test and must not hide an earlier valid one.
    results = [lab for lab in facts.labs if lab.lab_type == test
               and (procedure.procedure_date is None or lab.date <= procedure.procedure_date)]
    latest = max(results, key=lambda lab: (lab.date, lab.lab_id)) if results else None
    excluded = sorted(e.source_id for e in facts.excluded if e.source_type == "lab" and e.reason.startswith(test))
    excluded_note = f" ({len(excluded)} {test} result(s) excluded as not final/invalid: {excluded})" if excluded else ""
    quotes = [latest.evidence] if latest else []

    outcomes: dict[str, str] = {}
    for risk in candidates:
        window = lab_window_days(test, risk)
        if window is None:
            outcomes[risk] = "NOT_REQUIRED"
        elif latest is None:
            outcomes[risk] = "FAIL_MISSING"
        elif procedure.procedure_date is None:
            outcomes[risk] = "UNKNOWN"
        else:
            days = _days(latest.date, procedure.procedure_date)
            outcomes[risk] = "PASS" if days <= window else "FAIL_WINDOW"

    values = set(outcomes.values())
    if values == {"NOT_REQUIRED"}:
        return _check(rule, "NOT_APPLICABLE", f"{test} is not required for {procedure.procedure_risk} risk procedures.", quotes)

    if latest is None:
        found = f"no final {test} result on file{excluded_note}"
    elif procedure.procedure_date is None:
        found = f"most recent final {test} ({latest.lab_id}) dated {latest.date.isoformat()}{excluded_note}"
    else:
        days = _days(latest.date, procedure.procedure_date)
        found = f"most recent final {test} ({latest.lab_id}) dated {latest.date.isoformat()} is {days} days before the {procedure.procedure_date.isoformat()} procedure{excluded_note}"

    if procedure.procedure_risk:
        window = lab_window_days(test, procedure.procedure_risk)
        requirement = f"{procedure.procedure_risk} risk requires {test} within {window} days"
    else:
        requirement = "procedure risk unknown; per-risk outcome " + ", ".join(f"{r}={outcomes[r]}" for r in candidates)

    if values <= {"PASS", "NOT_REQUIRED"}:
        status: RuleStatus = "PASS"
    elif all(value.startswith("FAIL") for value in values):
        status = "FAIL"
    else:
        status = "INDETERMINATE"
    return _check(rule, status, f"{requirement}; {found}.", quotes, "REQUIRED_TESTING")


# =====================================================================
# Rule 3: anticoagulation
# =====================================================================

AnticoagStatus = str  # "TAKING" | "NOT_TAKING" | "UNKNOWN"
_STATUS_MENTIONS = {"CURRENTLY_TAKING", "STOPPED", "STATUS_UNCERTAIN"}


def anticoagulant_status(facts: CaseFacts, generic: str) -> tuple[AnticoagStatus, str, list[EvidenceQuote]]:
    # Med list x most recent note statement:
    #   active=True  -> TAKING, unless the latest note says stopped (-> UNKNOWN)
    #   active=False -> NOT_TAKING, unless the latest note says currently taking
    #   active=None  -> latest note decides; no note -> UNKNOWN (MISSING_REQUIRED_DATA)
    #   not listed   -> TAKING only if a note says so (outside-prescriber case)
    # "Unsure" or same-day conflicting statements -> UNKNOWN.

    entries = [entry for entry in facts.medication_list if any(g == generic for g, _ in entry.drugs)]
    mentions = [m for m in facts.medication_mentions if m.generic == generic and m.status in _STATUS_MENTIONS]
    quotes: list[EvidenceQuote] = [entry.evidence for entry in entries]

    if any(entry.active is True for entry in entries):
        listed = "ACTIVE"
    elif any(entry.active is None for entry in entries):
        listed = "UNKNOWN"
    elif entries:
        listed = "INACTIVE"
    else:
        listed = "NOT_LISTED"

    latest_note: str | None = None
    latest_date: date | None = None
    if mentions:
        latest_date = max(m.date for m in mentions)
        latest = [m for m in mentions if m.date == latest_date]
        quotes += [m.evidence for m in latest]
        statuses = {m.status for m in latest}
        latest_note = statuses.pop() if len(statuses) == 1 else "CONFLICTING"

    note_desc = f"most recent note statement ({latest_date.isoformat()}): {latest_note}" if latest_note else "no note states current use"
    basis = f"medication list: {listed.lower().replace('_', ' ')}; {note_desc}"

    if latest_note in {"STATUS_UNCERTAIN", "CONFLICTING"}:
        return "UNKNOWN", basis, quotes
    if listed == "ACTIVE":
        return ("UNKNOWN" if latest_note == "STOPPED" else "TAKING"), basis, quotes
    if listed == "INACTIVE":
        return ("TAKING" if latest_note == "CURRENTLY_TAKING" else "NOT_TAKING"), basis, quotes
    if listed == "UNKNOWN":
        if latest_note == "CURRENTLY_TAKING":
            return "TAKING", basis, quotes
        if latest_note == "STOPPED":
            return "NOT_TAKING", basis, quotes
        return "UNKNOWN", basis, quotes
    # NOT_LISTED
    return ("TAKING" if latest_note == "CURRENTLY_TAKING" else "NOT_TAKING"), basis, quotes


def drug_mention_lines(facts: CaseFacts, generic: str) -> list[EvidenceQuote]:
    # Rule 3 evidence is every line naming the drug (generic or brand), found here rather
    # than taken from the extractor, so the cited set is identical across runs.

    lines: list[EvidenceQuote] = []
    for doc in facts.documents:
        for line in doc.text.splitlines():
            if any(found == generic for found, _ in classify_drug(line)):
                lines.append(EvidenceQuote(source_type="document", source_id=doc.doc_id, date=doc.date.isoformat(),
                                           text=line.strip(), verbatim=True))
    return lines


def check_anticoagulation(facts: CaseFacts) -> list[RuleCheck]:
    # Evaluate anticoagulants on the medication list, plus any a note says the patient takes
    # now (or may take). Passing mentions such as a bridging instruction cannot change the
    # outcome and are not evaluated as separate medications.
    generics = sorted(
        {g for entry in facts.medication_list for g, cls in entry.drugs if cls == "ANTICOAGULANT"}
        | {m.generic for m in facts.medication_mentions
           if m.generic and m.drug_class == "ANTICOAGULANT" and m.status in {"CURRENTLY_TAKING", "STATUS_UNCERTAIN"}}
    )
    checks: list[RuleCheck] = []

    for generic in generics:
        rule = f"R3.ANTICOAGULATION.{generic.upper()}"
        status, basis, _ = anticoagulant_status(facts, generic)
        quotes = [entry.evidence for entry in facts.medication_list if any(g == generic for g, _ in entry.drugs)]
        quotes += drug_mention_lines(facts, generic)
        if status == "NOT_TAKING":
            checks.append(_check(rule, "NOT_APPLICABLE", f"{generic} is not a current medication ({basis}).", quotes))
            continue
        if status == "UNKNOWN":
            checks.append(
                _check(rule, "FAIL", f"Cannot determine whether the patient is currently taking {generic} ({basis}); "
                       "current use must be confirmed before the anticoagulation rule can be applied.",
                       quotes, "MISSING_REQUIRED_DATA")
            )
            continue

        plans = [plan for plan in facts.plans if plan.generic == generic]
        complete = [plan for plan in plans if plan.hold and plan.resume]
        if complete:
            best = max(complete, key=lambda plan: (plan.date, plan.doc_id))
            checks.append(
                _check(rule, "PASS", f"Patient is taking {generic} ({basis}); perioperative plan in {best.doc_id} "
                       f"({best.date.isoformat()}) gives both hold and resume instructions.",
                       [*quotes, best.hold, best.resume])
            )
            continue

        partial = []
        for plan in plans:
            missing = "hold" if plan.hold is None else "resume"
            partial.append(f"{plan.doc_id} ({plan.date.isoformat()}) gives no {missing} instruction")
        other_plans = sorted({plan.generic for plan in facts.plans if plan.generic != generic and plan.generic in generics})
        detail = "; ".join(partial) if partial else f"no perioperative plan addresses {generic}"
        if other_plans:
            detail += f"; plans on file only address {other_plans}"
        checks.append(
            _check(rule, "FAIL", f"Patient is taking {generic} ({basis}) but there is no clear perioperative plan "
                   f"with both hold and resume instructions: {detail}.",
                   [*quotes, *(p.hold for p in plans), *(p.resume for p in plans)], "ANTICOAGULATION_MANAGEMENT")
        )

    unrecognized: dict[str, list[MedicationMention]] = {}
    for mention in facts.medication_mentions:
        if mention.generic is None and mention.class_hint == "ANTICOAGULANT" and mention.status in {"CURRENTLY_TAKING", "STATUS_UNCERTAIN"}:
            unrecognized.setdefault(mention.drug_as_written.casefold(), []).append(mention)
    for name, mentions in sorted(unrecognized.items()):
        checks.append(
            _check(f"R3.ANTICOAGULATION.UNRECOGNIZED.{name.upper()}", "FAIL",
                   f"Document describes {mentions[0].drug_as_written!r} as a current anticoagulant, but it is not in the "
                   "reference list; pharmacist verification required.", [m.evidence for m in mentions], "MISSING_REQUIRED_DATA")
        )

    if not checks:
        antiplatelets = sorted({g for entry in facts.medication_list for g, cls in entry.drugs if cls == "ANTIPLATELET"})
        note = f" Antiplatelet(s) {antiplatelets} are not anticoagulants and do not require a plan under this rule." if antiplatelets else ""
        quotes = [entry.evidence for entry in facts.medication_list if any(cls == "ANTIPLATELET" for _, cls in entry.drugs)]
        checks.append(_check("R3.ANTICOAGULATION", "NOT_APPLICABLE", f"No current anticoagulant identified.{note}", quotes))
    return checks


# =====================================================================
# Rule 4: acute safety exclusions
# =====================================================================


def _bp_excluded(reading: VitalRecord) -> bool:
    return (reading.systolic or 0) >= SYSTOLIC_EXCLUSION_MMHG or (reading.diastolic or 0) >= DIASTOLIC_EXCLUSION_MMHG


def _temp_excluded(reading: VitalRecord) -> bool:
    return (reading.temperature_f or Decimal(0)) > TEMPERATURE_EXCLUSION_F


def most_recent_reading(readings: list[VitalRecord], severity: Callable[[VitalRecord], tuple]) -> tuple[VitalRecord, str | None]:
    # 1) latest date; 2) clock time, only if all candidates share a time basis (so a
    # 08:30 recheck beats 08:06); 3) text order within one note (untimed addenda);
    # 4) otherwise unknowable (UTC structured vs local note, no facility tz): use the most
    # severe reading and say so. Older abnormal readings never override newer ones.

    latest_date = max(r.date for r in readings)
    candidates = [r for r in readings if r.date == latest_date]
    if len(candidates) == 1:
        return candidates[0], None

    same_source = len({(r.source_type, r.source_id) for r in candidates}) == 1
    bases = {r.time_basis for r in candidates}
    if all(r.minutes is not None for r in candidates) and len(bases) == 1:
        key = lambda r: (r.minutes, r.position)  # noqa: E731
    elif same_source:
        key = lambda r: r.position  # noqa: E731
    else:
        chosen = max(candidates, key=lambda r: (severity(r), r.source_id, r.position))
        reason = (
            "their timestamps use different time bases (structured UTC vs. local note time); "
            "configure the facility timezone to order them"
            if len(bases) > 1
            else "not every reading has a time"
        )
        return chosen, (f"{len(candidates)} readings on {latest_date.isoformat()} cannot be ordered reliably "
                        f"({reason}); the most severe one was used")

    top = max(key(r) for r in candidates)
    tied = [r for r in candidates if key(r) == top]
    chosen = max(tied, key=lambda r: (severity(r), r.source_id))
    return chosen, (f"{len(tied)} readings share the latest timestamp; the most severe was used" if len(tied) > 1 else None)


def _when(reading: VitalRecord) -> str:
    time = f" {reading.minutes // 60:02d}:{reading.minutes % 60:02d}" if reading.minutes is not None else ""
    return f"{reading.date.isoformat()}{time}"


_VITALS_CONTEXT = re.compile(r"vital|flowsheet|temp|blood pressure|\bBP\b", re.IGNORECASE)


def _superseded(unverified: UnverifiedVital, readings: list[VitalRecord]) -> bool:
    # A verified reading supersedes an unverified one only if it is provably later: a later
    # date, or the same date with comparable clock times. Undatable readings never are.
    if unverified.date is None:
        return False
    for reading in readings:
        if reading.date > unverified.date:
            return True
        if (reading.date == unverified.date and reading.minutes is not None and unverified.minutes is not None
                and reading.time_basis == unverified.time_basis and reading.minutes > unverified.minutes):
            return True
    return False


def _unverified_quotes(facts: CaseFacts, unverified: UnverifiedVital) -> list[EvidenceQuote]:
    # Point the reviewer at the source: every line of the cited note containing the claimed
    # value (found by Python), else a non-verbatim reference saying what failed.
    doc = next((d for d in facts.documents if d.doc_id == unverified.doc_id), None)
    needle = normalize_for_match(unverified.claimed)
    if doc is not None and needle:
        lines = [line.strip() for line in doc.text.splitlines() if needle in normalize_for_match(line)]
        if lines:
            return [EvidenceQuote(source_type="document", source_id=doc.doc_id, date=doc.date.isoformat(), text=line, verbatim=True)
                    for line in lines]
    return [EvidenceQuote(source_type="document" if unverified.doc_id else "vital", source_id=unverified.doc_id or unverified.source_id,
                          date=unverified.date.isoformat() if unverified.date else None,
                          text=f"unverified reading {unverified.claimed!r}: {unverified.reason}", verbatim=False)]


def _check_vital(facts: CaseFacts, kind: str) -> list[RuleCheck]:
    if kind == "BLOOD_PRESSURE":
        rule, label = "R4.BLOOD_PRESSURE", "blood pressure"
        excluded, severity = _bp_excluded, lambda r: (_bp_excluded(r), r.systolic or 0, r.diastolic or 0)
        threshold = f"SBP >= {SYSTOLIC_EXCLUSION_MMHG} or DBP >= {DIASTOLIC_EXCLUSION_MMHG} mmHg"
    else:
        rule, label = "R4.TEMPERATURE", "temperature"
        excluded, severity = _temp_excluded, lambda r: (_temp_excluded(r), r.temperature_f or Decimal(0))
        threshold = f"temperature > {TEMPERATURE_EXCLUSION_F} F"

    readings = [r for r in facts.vitals if r.kind == kind]
    # Fail closed: a reading that failed verification and is not superseded by a newer verified
    # reading could be the true latest value, so the chart goes to review instead of silently
    # falling back to an older normal reading.
    pending = sorted((u for u in facts.unverified_vitals if u.kind == kind and not _superseded(u, readings)),
                     key=lambda u: (u.date or date.min, u.minutes or -1, u.source_id))
    pending_check: list[RuleCheck] = []
    if pending:
        listed = "; ".join(f"{u.claimed!r} ({u.date.isoformat() if u.date else 'undated'}, {u.reason})" for u in pending)
        pending_check = [_check(f"{rule}.UNVERIFIED_READING", "FAIL",
                                f"{len(pending)} {label} reading(s) could not be verified against the source and are not "
                                f"superseded by a newer verified reading: {listed}. Confirm the latest {label} before scheduling.",
                                [q for u in pending for q in _unverified_quotes(facts, u)], "MISSING_REQUIRED_DATA")]

    if not readings:
        return [_check(rule, "FAIL", f"No verified {label} in structured vitals or in any document.",
                       _related_lines(facts, _VITALS_CONTEXT), "MISSING_REQUIRED_DATA"), *pending_check]

    latest, ambiguity = most_recent_reading(readings, severity)
    superseded = sorted((r for r in readings if r is not latest and excluded(r)), key=lambda r: (r.date, r.minutes or -1))
    described = f"most recent verified {label} {latest.display} at {_when(latest)} ({latest.source_id})"
    notes = []
    if ambiguity:
        notes.append(ambiguity)
    if superseded:
        notes.append("older abnormal reading(s) superseded: " + ", ".join(f"{r.display} at {_when(r)}" for r in superseded))
    suffix = f" Note: {'; '.join(notes)}." if notes else ""
    quotes = [latest.evidence, *(r.evidence for r in superseded)]
    if excluded(latest):
        # Still an exclusion even if an unverified later reading exists: the conservative outcome stands.
        return [_check(rule, "FAIL", f"Acute safety exclusion: {described} meets {threshold}.{suffix}", quotes,
                       "ACUTE_SAFETY_EXCLUSION"), *pending_check]
    if pending:
        return [_check(rule, "INDETERMINATE", f"{described[0].upper()}{described[1:]} does not meet {threshold}, but a "
                       f"possibly newer reading could not be verified.{suffix}", quotes), *pending_check]
    return [_check(rule, "PASS", f"{described[0].upper()}{described[1:]} does not meet {threshold}.{suffix}", quotes)]


def check_vitals(facts: CaseFacts) -> list[RuleCheck]:
    return [*_check_vital(facts, "BLOOD_PRESSURE"), *_check_vital(facts, "TEMPERATURE")]


# =====================================================================
# Informational (never affects the decision)
# =====================================================================


def review_npo_instructions(facts: CaseFacts) -> InformationalCheck:
    # Informational only: returned into audit.informational_checks, never rule_checks,
    # so decide() cannot see it. Negations ("not reviewed", "deferred") are checked first.

    check = "INFO.NPO_FASTING_REVIEW"
    notes = [doc for doc in facts.documents if is_nursing_intake_title(doc.title)]
    if not notes:
        return InformationalCheck(check=check, status="NO_NURSING_INTAKE_NOTE", quotes=[],
                                  summary="No nursing intake / pre-admission testing note on file.")

    for doc in sorted(notes, key=lambda d: (d.date, d.doc_id), reverse=True):
        lines = doc.text.splitlines()
        for index, line in enumerate(lines):
            if not NPO_PATTERN.search(line):
                continue
            context = " ".join(lines[index : index + 2])
            quote = EvidenceQuote(source_type="document", source_id=doc.doc_id, date=doc.date.isoformat(),
                                  text=line.strip(), verbatim=True)
            where = f"{doc.title!r} ({doc.doc_id}, {doc.date.isoformat()})"
            if NPO_NOT_REVIEWED_PATTERN.search(context):
                status, summary = "NOT_REVIEWED", f"{where} indicates NPO/fasting instructions were not reviewed."
            elif NPO_ACKNOWLEDGED_PATTERN.search(context):
                status, summary = "REVIEWED_WITH_PATIENT", f"{where} documents NPO/fasting instructions reviewed with the patient."
            else:
                status, summary = "INSTRUCTIONS_DOCUMENTED", f"{where} documents NPO/fasting instructions; patient acknowledgement not recorded."
            if doc is not max(notes, key=lambda d: (d.date, d.doc_id)):
                summary += " The most recent nursing intake note does not mention NPO/fasting."
            return InformationalCheck(check=check, status=status, summary=summary, quotes=[quote])

    return InformationalCheck(check=check, status="NOT_DOCUMENTED", quotes=[],
                              summary=f"{len(notes)} nursing intake note(s) on file; none mention NPO/fasting instructions.")


# =====================================================================
# Decision
# =====================================================================

_CATEGORY_ORDER: dict[str, int] = {
    "ACUTE_SAFETY_EXCLUSION": 0,
    "MISSING_REQUIRED_DATA": 1,
    "REQUIRED_DOCUMENTATION": 2,
    "REQUIRED_TESTING": 3,
    "ANTICOAGULATION_MANAGEMENT": 4,
}


def decide(checks: list[RuleCheck]) -> tuple[Decision, list[TriageIssue]]:
    issues = [
        TriageIssue(category=check.issue_category, evidence=check.evidence)
        for check in checks
        if check.status == "FAIL" and check.issue_category is not None
    ]
    issues.sort(key=lambda issue: (_CATEGORY_ORDER[issue.category], issue.evidence.rule))
    categories = {issue.category for issue in issues}
    if "ACUTE_SAFETY_EXCLUSION" in categories:
        return "NOT_CLEARED", issues
    if issues:
        return "NEEDS_FOLLOW_UP", issues
    return "READY", issues


def build_explanation(decision: Decision, issues: list[TriageIssue], checks: list[RuleCheck]) -> str:
    if issues:
        listed = " ".join(f"({index}) {issue.category}: {issue.evidence.summary}" for index, issue in enumerate(issues, 1))
        passed = [check.rule for check in checks if check.status == "PASS"]
        tail = f" Passed: {', '.join(passed)}." if passed else ""
        return f"{decision}: {len(issues)} issue(s). {listed}{tail}"
    passed = " ".join(check.evidence.summary for check in checks if check.status == "PASS")
    return f"READY: every required policy check passed. {passed}"


def evaluate(facts: CaseFacts) -> TriageOutput:
    procedure = resolve_procedure(facts)
    checks: list[RuleCheck] = [
        *check_procedure_fields(procedure),
        check_history_and_physical(facts, procedure.procedure_date),
        *check_surgical_consent(facts),
        check_lab(facts, "CBC", procedure),
        check_lab(facts, "CMP", procedure),
        *check_anticoagulation(facts),
        *check_vitals(facts),
    ]

    # Safety invariant: a rule that could not be evaluated must never yield READY.
    if any(check.status == "INDETERMINATE" for check in checks) and not any(
        check.status == "FAIL" and check.issue_category == "MISSING_REQUIRED_DATA" for check in checks
    ):
        pending = [check.rule for check in checks if check.status == "INDETERMINATE"]
        checks.append(_check("R0.UNEVALUATED_RULES", "FAIL", f"Rules could not be evaluated: {pending}.", [], "MISSING_REQUIRED_DATA"))

    decision, issues = decide(checks)
    audit = AuditTrail(
        policy_version=POLICY_VERSION,
        pipeline_version=f"{PIPELINE_VERSION}+prompt.{EXTRACTION_PROMPT_VERSION}",
        extraction_model=facts.extraction_model,
        facility_timezone=facts.facility_timezone,
        review_time=facts.review_time,
        procedure_type=facts.procedure_type,
        procedure_date=procedure.date_field,
        procedure_risk=procedure.risk_field,
        rule_checks=checks,
        excluded_evidence=sorted(facts.excluded, key=lambda e: (e.source_type, e.source_id, e.reason)),
        informational_checks=[review_npo_instructions(facts)],
        warnings=sorted(set(facts.warnings)),
    )
    return TriageOutput(
        decision=decision,
        issues=issues,
        explanation=build_explanation(decision, issues, checks),
        audit=audit,
    )
