"""Layer 1: normalize structured fields, extract quoted facts from documents (one model
call), and verify every fact against the source text before policy_engine.py sees it."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict

from reference import (
    ACCEPTED_LAB_STATUSES,
    PROCEDURE_RISK_EXCLUDE_PATTERN,
    TIMING_PATTERN,
    UNSIGNED_CONSENT_PATTERN,
    DocumentClass,
    DrugClass,
    LabType,
    ProcedureRisk,
    anticoagulant_generics,
    classify_document_title,
    classify_drug,
    find_dates,
    fold_degree_signs,
    locate_quote,
    normalize_for_match,
    normalize_lab,
    parse_clock_time,
    parse_iso_date,
    load_timezone,
    to_facility_time,
    structured_procedure_date,
    structured_procedure_risk,
)
from schemas import EvidenceQuote, ExcludedEvidence, PatientSubmission

# =====================================================================
# Extraction contract. Strict structured outputs need every field present, so "unknown"
# is an explicit null/enum value, never an omitted key. Numbers stay strings
# ("157/117", "38.6") so verification can find them verbatim in the quote.
# =====================================================================


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractedVitalReading(_Strict):
    doc_ref: str
    measurement: Literal["BLOOD_PRESSURE", "TEMPERATURE"]
    value_text: str
    unit: Literal["mmHg", "C", "F", "NOT_STATED"]
    time_text: str | None
    measured_date_text: str | None
    context: Literal["MEASURED_AT_THIS_ENCOUNTER", "PATIENT_REPORTED_OR_HOME", "HISTORICAL_REFERENCE"]
    quote: str


class ExtractedMedicationMention(_Strict):
    doc_ref: str
    drug_as_written: str
    class_hint: Literal["ANTICOAGULANT", "ANTIPLATELET", "OTHER_OR_UNSURE"]
    status: Literal["CURRENTLY_TAKING", "STOPPED", "STATUS_UNCERTAIN", "PERIOPERATIVE_INSTRUCTION"]
    quote: str


class ExtractedAnticoagulationPlan(_Strict):
    doc_ref: str
    drug_as_written: str
    hold_instruction_quote: str | None
    resume_instruction_quote: str | None


class ExtractedConsent(_Strict):
    doc_ref: str
    patient_signature_status: Literal["SIGNED_BY_PATIENT", "NOT_SIGNED", "CANNOT_DETERMINE"]
    signature_quote: str | None
    procedure_quote: str | None


class ExtractedHistoryAndPhysical(_Strict):
    doc_ref: str
    completion_status: Literal["COMPLETED", "INCOMPLETE", "CANNOT_DETERMINE"]
    quote: str | None


class ExtractedProcedureDateMention(_Strict):
    doc_ref: str
    status: Literal["SCHEDULED", "NOT_YET_SCHEDULED"]
    date_text: str | None
    quote: str


class ExtractedProcedureRiskMention(_Strict):
    doc_ref: str
    risk_category: Literal["LOW", "MODERATE", "HIGH", "NOT_ASSIGNED"]
    quote: str


class DocumentExtraction(_Strict):
    vital_readings: list[ExtractedVitalReading]
    medication_mentions: list[ExtractedMedicationMention]
    anticoagulation_plans: list[ExtractedAnticoagulationPlan]
    consents: list[ExtractedConsent]
    history_and_physicals: list[ExtractedHistoryAndPhysical]
    procedure_date_mentions: list[ExtractedProcedureDateMention]
    procedure_risk_mentions: list[ExtractedProcedureRiskMention]


EXTRACTION_SYSTEM_PROMPT = """
You extract facts from clinical documents for a pre-operative scheduling review.
You do NOT decide readiness, apply any policy, compare dates, compute intervals or convert units. Report only what the text states.

General rules
- Every quote must be copied exactly, character for character, from the text of the document named by doc_ref. Copy one contiguous span. Do not paraphrase, shorten, join separate passages or add ellipses.
- Use doc_ref values exactly as given (e.g. "D3").
- If something is not stated, leave it out (or use null where allowed). Never infer or guess values.
- "ASA physical status" is an anesthesia score, not aspirin.

vital_readings: every blood pressure and body temperature value written in any document.
- value_text: the number exactly as written, e.g. "157/117" for blood pressure or "38.6" for temperature.
- unit: the unit written with the value: "C" or "F" for temperature, "mmHg" for blood pressure, "NOT_STATED" if none is written.
- time_text: the clock time (HH:MM) the text gives for when this specific reading was taken, including times given for rechecks or addenda; null if none.
- measured_date_text: only if the text says this reading was taken on a different date than the document date, copy that date as written; otherwise null.
- context: MEASURED_AT_THIS_ENCOUNTER for readings taken by clinical staff at the documented encounter; PATIENT_REPORTED_OR_HOME for home logs or patient-reported values; HISTORICAL_REFERENCE for readings the note cites from another encounter.
- Do not report statements without a number ("afebrile", "see flowsheet", "vitals deferred", "no fevers").
- quote: the sentence or line containing the value.

medication_mentions: every mention of an anticoagulant or antiplatelet ("blood thinner") medication, by generic or brand name, in any document, including medication-list lines.
- drug_as_written: the drug name exactly as written.
- class_hint: ANTICOAGULANT, ANTIPLATELET or OTHER_OR_UNSURE.
- status, as stated by that sentence or line:
  CURRENTLY_TAKING: the patient is taking / maintained on it now, or it is listed as a current medication.
  STOPPED: it was discontinued, stopped, completed, replaced, or the patient is not currently on it.
  STATUS_UNCERTAIN: the text says it is unclear or unconfirmed whether the patient is still taking it.
  PERIOPERATIVE_INSTRUCTION: the sentence instructs or discusses holding, bridging, resuming or managing the drug around surgery rather than stating current use.
- quote: the sentence or list line containing the mention.

anticoagulation_plans: one entry per document per anticoagulant for which that document addresses perioperative management.
- hold_instruction_quote: the exact text instructing when to stop or hold the drug before surgery, including its timing; null if the document gives none.
- resume_instruction_quote: the exact text instructing when to resume or restart the drug after surgery, including its timing; null if the document gives none.
- Text saying management is deferred, to be determined, to be discussed, or will be addressed later is NOT an instruction: use null for that part.

consents: exactly one entry for every document whose document_class is SURGICAL_CONSENT.
- patient_signature_status: SIGNED_BY_PATIENT only if the document shows the patient (or legal representative) signed it; NOT_SIGNED if the signature is pending, blank, not completed, or the form was only printed, discussed or sent; CANNOT_DETERMINE otherwise. A physician attestation or physician signature is not a patient signature.
- signature_quote: the line showing the patient's signature or its absence; null if there is none.
- procedure_quote: the procedure name exactly as written on the consent; null if none is named.

history_and_physicals: exactly one entry for every document whose document_class is HISTORY_AND_PHYSICAL.
- completion_status: COMPLETED if the H&P is signed or otherwise finalized; INCOMPLETE if the text says it is a draft, unsigned, incomplete or pending; CANNOT_DETERMINE otherwise.
- quote: the line supporting that status (for example the electronic signature line); null if none.

procedure_date_mentions: only when requested_fallbacks.procedure_date is true; otherwise return an empty list. Statements about when the planned surgery is scheduled.
- status: SCHEDULED if a specific date is given; NOT_YET_SCHEDULED if the date is pending or to be determined.
- date_text: the date exactly as written when SCHEDULED, otherwise null.

procedure_risk_mentions: only when requested_fallbacks.procedure_risk is true; otherwise return an empty list. Statements that explicitly name the surgical procedure risk category assigned to the planned procedure (LOW, MODERATE or HIGH), or that say no procedure risk category has been assigned (NOT_ASSIGNED). Do not report bleeding risk, thromboembolic risk, cardiac risk, fall risk, ASA physical status or general descriptions such as "elevated surgical risk".
""".strip()

EXTRACTION_PROMPT_VERSION = hashlib.sha256(
    (EXTRACTION_SYSTEM_PROMPT + json.dumps(DocumentExtraction.model_json_schema(), sort_keys=True)).encode()
).hexdigest()[:12]

MAX_OUTPUT_TOKENS = 16_000

# Prompt caching: instructions + schema (~2.1k tokens, above the 1,024-token minimum)
# are byte-identical on every call, so they form a shared cacheable prefix; all
# case-specific content lives in the single user message after it. The key pins
# routing to that prefix and changes automatically whenever the prompt/schema does.
PROMPT_CACHE_KEY = f"preop-extraction-{EXTRACTION_PROMPT_VERSION}"

telemetry_logger = logging.getLogger("preop.telemetry")


@dataclass(frozen=True)
class ExtractionTelemetry:
    # Cost/cache monitoring only. Never copied into TriageOutput: tokens, latency and
    # response ids vary per call and would break byte-identical outputs.

    response_id: str | None
    model: str | None
    prompt_version: str
    prompt_cache_key: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    latency_ms: int
    status: str

    @property
    def cache_hit_ratio(self) -> float:
        return self.cached_input_tokens / self.input_tokens if self.input_tokens else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "cache_hit_ratio": round(self.cache_hit_ratio, 4)}


def _usage_int(obj: Any, *path: str) -> int:
    # usage / *_details can be missing or None depending on model and SDK version.

    for name in path:
        obj = getattr(obj, name, None)
        if obj is None:
            return 0
    return int(obj) if isinstance(obj, (int, float)) else 0


def telemetry_from_response(response: Any, *, latency_ms: int) -> ExtractionTelemetry:
    usage = getattr(response, "usage", None)
    return ExtractionTelemetry(
        response_id=getattr(response, "id", None),
        model=getattr(response, "model", None),
        prompt_version=EXTRACTION_PROMPT_VERSION,
        prompt_cache_key=PROMPT_CACHE_KEY,
        input_tokens=_usage_int(usage, "input_tokens"),
        cached_input_tokens=_usage_int(usage, "input_tokens_details", "cached_tokens"),
        output_tokens=_usage_int(usage, "output_tokens"),
        reasoning_tokens=_usage_int(usage, "output_tokens_details", "reasoning_tokens"),
        latency_ms=latency_ms,
        status=str(getattr(response, "status", "unknown")),
    )


class ExtractionError(RuntimeError):
    # Refusal, truncation or unparseable output: the case goes to a human, never a guess.
    pass


# =====================================================================
# Verified facts: the only inputs policy_engine.py sees. Every document-derived record
# carries an EvidenceQuote snapped to its source line.
# =====================================================================

SignatureStatus = Literal["SIGNED_BY_PATIENT", "NOT_SIGNED", "CANNOT_DETERMINE"]
MentionStatus = Literal["CURRENTLY_TAKING", "STOPPED", "STATUS_UNCERTAIN", "PERIOPERATIVE_INSTRUCTION"]


@dataclass(frozen=True)
class DocRecord:
    doc_id: str
    title: str | None
    date: date
    doc_class: DocumentClass
    class_reason: str
    text: str


@dataclass(frozen=True)
class LabRecord:
    lab_id: str
    lab_type: LabType
    date: date
    evidence: EvidenceQuote


@dataclass(frozen=True)
class VitalRecord:
    kind: Literal["BLOOD_PRESSURE", "TEMPERATURE"]
    source_type: Literal["vital", "document"]
    source_id: str
    date: date
    minutes: int | None
    position: tuple[int, int]
    systolic: int | None
    diastolic: int | None
    temperature_f: Decimal | None
    display: str
    evidence: EvidenceQuote
    # LOCAL: facility wall-clock (all note times; structured times converted with a
    # configured timezone). UTC_OFFSET / UNSPECIFIED cannot be ordered against LOCAL.
    time_basis: str = "LOCAL"


@dataclass(frozen=True)
class UnverifiedVital:
    kind: str  # BLOOD_PRESSURE | TEMPERATURE
    source_id: str
    doc_id: str | None  # document to cite for review; None for structured or unresolvable sources
    date: date | None  # None = cannot be placed in time, so it can never be superseded
    minutes: int | None
    claimed: str
    reason: str
    time_basis: str = "LOCAL"


@dataclass(frozen=True)
class MedListEntry:
    source_id: str
    name: str
    active: bool | None
    drugs: tuple[tuple[str, DrugClass], ...]
    evidence: EvidenceQuote


@dataclass(frozen=True)
class MedicationMention:
    doc_id: str
    date: date
    drug_as_written: str
    generic: str | None
    drug_class: DrugClass | None
    class_hint: str
    status: MentionStatus
    evidence: EvidenceQuote


@dataclass(frozen=True)
class PlanRecord:
    doc_id: str
    date: date
    generic: str
    hold: EvidenceQuote | None
    resume: EvidenceQuote | None


@dataclass(frozen=True)
class ConsentRecord:
    doc_id: str
    date: date
    title: str | None
    status: SignatureStatus
    signature: EvidenceQuote | None
    procedure_name: str | None
    procedure: EvidenceQuote | None
    notes: tuple[str, ...]


@dataclass(frozen=True)
class HistoryAndPhysicalRecord:
    doc_id: str
    date: date
    title: str | None
    status: Literal["COMPLETED", "INCOMPLETE", "CANNOT_DETERMINE"]
    evidence: EvidenceQuote | None


@dataclass(frozen=True)
class ProcedureDateMention:
    doc_id: str
    doc_date: date
    scheduled_for: date | None
    evidence: EvidenceQuote


@dataclass(frozen=True)
class ProcedureRiskMention:
    doc_id: str
    doc_date: date
    risk: ProcedureRisk | None
    evidence: EvidenceQuote


@dataclass
class CaseFacts:
    extraction_model: str
    facility_timezone: str | None
    review_time: str | None
    review_date: date | None
    procedure_type: str | None
    structured_procedure_date: str | None
    structured_procedure_risk: str | None
    needs_procedure_date_fallback: bool = False
    needs_procedure_risk_fallback: bool = False
    documents: list[DocRecord] = field(default_factory=list)
    labs: list[LabRecord] = field(default_factory=list)
    vitals: list[VitalRecord] = field(default_factory=list)
    unverified_vitals: list[UnverifiedVital] = field(default_factory=list)
    medication_list: list[MedListEntry] = field(default_factory=list)
    medication_mentions: list[MedicationMention] = field(default_factory=list)
    plans: list[PlanRecord] = field(default_factory=list)
    consents: list[ConsentRecord] = field(default_factory=list)
    history_and_physicals: list[HistoryAndPhysicalRecord] = field(default_factory=list)
    procedure_date_mentions: list[ProcedureDateMention] = field(default_factory=list)
    procedure_risk_mentions: list[ProcedureRiskMention] = field(default_factory=list)
    excluded: list[ExcludedEvidence] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    telemetry: "ExtractionTelemetry | None" = None  # not part of the decision output

    @property
    def timezone(self):
        return load_timezone(self.facility_timezone)

    def exclude(self, source_type: Any, source_id: str, reason: str) -> None:
        self.excluded.append(ExcludedEvidence(source_type=source_type, source_id=source_id, reason=reason))


# =====================================================================
# Structured-field preprocessing (no model). Anything dated after review time, not final,
# implausible or malformed is excluded here and logged to audit.excluded_evidence.
# =====================================================================


def _is_after_review(item_date: date, facts: CaseFacts) -> bool:
    return facts.review_date is not None and item_date > facts.review_date


def _preprocess_documents(submission: PatientSubmission, facts: CaseFacts) -> None:
    seen: set[str] = set()
    for index, doc in enumerate(submission.documents):
        doc_id = doc.doc_id or f"documents[{index}]"
        if doc_id in seen:
            facts.exclude("document", doc_id, "duplicate doc_id; only the first occurrence is used")
            continue
        seen.add(doc_id)
        doc_date = parse_iso_date(doc.date)
        if doc_date is None:
            facts.exclude("document", doc_id, f"missing or invalid document date {doc.date!r}")
            continue
        if _is_after_review(doc_date, facts):
            facts.exclude("document", doc_id, f"dated {doc_date} after review time {facts.review_time}")
            continue
        if not (doc.text or "").strip():
            facts.exclude("document", doc_id, "empty document text")
            continue
        doc_class, reason = classify_document_title(doc.type)
        facts.documents.append(
            DocRecord(doc_id=doc_id, title=doc.type, date=doc_date, doc_class=doc_class, class_reason=reason, text=doc.text or "")
        )
    facts.documents.sort(key=lambda d: (d.date, d.doc_id))


def _preprocess_labs(submission: PatientSubmission, facts: CaseFacts) -> None:
    for index, lab in enumerate(submission.labs):
        lab_id = lab.id or f"labs[{index}]"
        lab_type, _reason = normalize_lab(lab.code, lab.display)
        if lab_type is None:
            if "conflicts" in _reason:
                facts.exclude("lab", lab_id, _reason)
            continue
        status = (lab.status or "").strip().lower()
        lab_date, _, _ = to_facility_time(lab.effective_at, facts.timezone)
        if status not in ACCEPTED_LAB_STATUSES:
            facts.exclude("lab", lab_id, f"{lab_type} status {lab.status!r} is not final")
            continue
        if lab_date is None:
            facts.exclude("lab", lab_id, f"{lab_type} has missing or invalid effective_at {lab.effective_at!r}")
            continue
        if _is_after_review(lab_date, facts):
            facts.exclude("lab", lab_id, f"{lab_type} dated {lab_date} after review time {facts.review_time}")
            continue
        facts.labs.append(
            LabRecord(
                lab_id=lab_id,
                lab_type=lab_type,
                date=lab_date,
                evidence=EvidenceQuote(
                    source_type="lab",
                    source_id=lab_id,
                    date=lab.effective_at,
                    text=f"{lab.code} ({lab.display}) effective {lab.effective_at}, status {lab.status}, normalized as {lab_type}",
                    verbatim=False,
                ),
            )
        )


def _plausible_bp(systolic: int, diastolic: int) -> bool:
    return 40 <= systolic <= 300 and 20 <= diastolic <= 200 and systolic > diastolic


def _plausible_temperature_f(value: Decimal) -> bool:
    return Decimal("86") <= value <= Decimal("113")


def _preprocess_structured_vitals(submission: PatientSubmission, facts: CaseFacts) -> None:
    for index, vital in enumerate(submission.vitals):
        vital_id = f"vitals[{index}]"
        vital_type = (vital.type or "").strip().lower()
        vital_date, minutes, time_basis = to_facility_time(vital.date, facts.timezone)
        if vital_type not in {"blood_pressure", "temperature"}:
            continue
        kind = "BLOOD_PRESSURE" if vital_type == "blood_pressure" else "TEMPERATURE"
        if vital_date is None:
            _reject_vital(facts, kind, vital_id, None, None, None, str(vital.date),
                          f"missing or invalid date {vital.date!r}", source_type="vital")
            continue
        if _is_after_review(vital_date, facts):
            facts.exclude("vital", vital_id, f"dated {vital_date} after review time {facts.review_time}")
            continue
        if vital_type == "blood_pressure":
            systolic = getattr(vital, "systolic", None)
            diastolic = getattr(vital, "diastolic", None)
            # Clinical BP is recorded in whole mmHg; fractional, missing or implausible values
            # are treated as unverified rather than truncated or silently dropped.
            if (systolic is None or diastolic is None or float(systolic) != int(systolic) or float(diastolic) != int(diastolic)
                    or not _plausible_bp(int(systolic), int(diastolic))):
                _reject_vital(facts, kind, vital_id, None, vital_date, minutes, f"{systolic}/{diastolic}",
                              f"blood pressure {systolic}/{diastolic} missing, fractional or implausible", source_type="vital",
                              time_basis=time_basis)
                continue
            display = f"BP {int(systolic)}/{int(diastolic)} mmHg"
            facts.vitals.append(
                VitalRecord(
                    kind="BLOOD_PRESSURE", source_type="vital", source_id=vital_id, date=vital_date, minutes=minutes,
                    position=(0, 0), systolic=int(systolic), diastolic=int(diastolic), temperature_f=None, display=display, time_basis=time_basis,
                    evidence=EvidenceQuote(source_type="vital", source_id=vital_id, date=vital.date,
                                           text=f"{display} recorded {vital.date} ({vital.source})", verbatim=False),
                )
            )
        else:
            value_f = getattr(vital, "value_f", None)
            if value_f is None:
                _reject_vital(facts, kind, vital_id, None, vital_date, minutes, "None",
                              "temperature value missing", source_type="vital", time_basis=time_basis)
                continue
            temperature_f = Decimal(str(value_f))
            if not _plausible_temperature_f(temperature_f):
                _reject_vital(facts, kind, vital_id, None, vital_date, minutes, f"{value_f} F",
                              f"temperature {value_f} F implausible", source_type="vital", time_basis=time_basis)
                continue
            display = f"Temp {temperature_f} F"
            facts.vitals.append(
                VitalRecord(
                    kind="TEMPERATURE", source_type="vital", source_id=vital_id, date=vital_date, minutes=minutes,
                    position=(0, 0), systolic=None, diastolic=None, temperature_f=temperature_f, display=display, time_basis=time_basis,
                    evidence=EvidenceQuote(source_type="vital", source_id=vital_id, date=vital.date,
                                           text=f"{display} recorded {vital.date} ({vital.source})", verbatim=False),
                )
            )


def _preprocess_medication_list(submission: PatientSubmission, facts: CaseFacts) -> None:
    for index, medication in enumerate(submission.medications):
        if not medication.name:
            continue
        drugs = tuple(classify_drug(medication.name))
        if not drugs:
            continue
        source_id = f"medications[{index}]"
        facts.medication_list.append(
            MedListEntry(
                source_id=source_id,
                name=medication.name,
                active=medication.active,
                drugs=drugs,
                evidence=EvidenceQuote(
                    source_type="medication", source_id=source_id, date=None,
                    text=f"medication list: {medication.name!r}, active={medication.active}", verbatim=False,
                ),
            )
        )


def preprocess(submission: PatientSubmission, *, model: str, facility_timezone: str | None = None) -> CaseFacts:
    procedure = submission.procedure
    review_time = submission.metadata.submission_received_at if submission.metadata else None
    facts = CaseFacts(
        extraction_model=model,
        review_time=review_time,
        facility_timezone=facility_timezone,
        review_date=to_facility_time(review_time, load_timezone(facility_timezone))[0],
        procedure_type=procedure.procedure_type if procedure else None,
        structured_procedure_date=procedure.procedure_date if procedure else None,
        structured_procedure_risk=procedure.procedure_risk if procedure else None,
    )
    # Documents are only consulted for date/risk when the structured field is unusable.
    facts.needs_procedure_date_fallback = structured_procedure_date(facts.structured_procedure_date) is None
    facts.needs_procedure_risk_fallback = structured_procedure_risk(facts.structured_procedure_risk) is None
    if facts.review_date is None:
        facts.warnings.append(
            f"submission_received_at {review_time!r} missing or invalid; evidence dated after review time cannot be excluded"
        )
    _preprocess_documents(submission, facts)
    _preprocess_labs(submission, facts)
    _preprocess_structured_vitals(submission, facts)
    _preprocess_medication_list(submission, facts)
    return facts


# =====================================================================
# LLM call
# =====================================================================


def document_refs(documents: list[DocRecord]) -> dict[str, DocRecord]:
    # The model sees D1, D2, ... instead of UUIDs: it was observed dropping UUID segments
    # when copying ids back. verify_extraction maps refs back to real doc_ids.

    return {f"D{index}": doc for index, doc in enumerate(documents, 1)}


def build_extraction_input(facts: CaseFacts) -> str:
    # Documents only: no patient identifiers, no policy, no structured fields.
    # sort_keys=True fixes the byte layout: "documents" serializes before the small
    # case-specific "requested_fallbacks" flags, which therefore sit at the very end.
    payload = {
        "requested_fallbacks": {
            "procedure_date": facts.needs_procedure_date_fallback,
            "procedure_risk": facts.needs_procedure_risk_fallback,
        },
        "documents": [
            {
                "doc_ref": ref,
                "title": doc.title,
                "date": doc.date.isoformat(),
                "document_class": doc.doc_class,
                "text": doc.text,
            }
            for ref, doc in document_refs(facts.documents).items()
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def call_extractor(
    client: Any, *, model: str, facts: CaseFacts, on_telemetry: Callable[[ExtractionTelemetry], None] | None = None
) -> tuple[DocumentExtraction, ExtractionTelemetry | None]:
    # responses.parse turns DocumentExtraction into a strict JSON schema. gpt-5.6-luna
    # only accepts temperature with reasoning effort "none". store=False keeps PHI off
    # OpenAI's stored responses.

    if not facts.documents:
        return DocumentExtraction(
            vital_readings=[], medication_mentions=[], anticoagulation_plans=[], consents=[],
            history_and_physicals=[], procedure_date_mentions=[], procedure_risk_mentions=[],
        ), None

    started = time.perf_counter()
    response = client.responses.parse(
        model=model,
        instructions=EXTRACTION_SYSTEM_PROMPT,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": build_extraction_input(facts)}],
            }
        ],
        text_format=DocumentExtraction,
        temperature=0,
        reasoning={"effort": "none"},
        max_output_tokens=MAX_OUTPUT_TOKENS,
        store=False,
        prompt_cache_key=PROMPT_CACHE_KEY,
    )
    telemetry = telemetry_from_response(response, latency_ms=round((time.perf_counter() - started) * 1000))
    facts.telemetry = telemetry
    telemetry_logger.info("extraction_call %s", json.dumps(telemetry.as_dict(), sort_keys=True))
    if on_telemetry is not None:
        on_telemetry(telemetry)  # before status checks: failed calls are billed too

    status = getattr(response, "status", "completed")
    if status != "completed":
        raise ExtractionError(f"extraction response status {status!r}: {getattr(response, 'incomplete_details', None)}")
    parsed = getattr(response, "output_parsed", None)
    if parsed is None:
        raise ExtractionError("extraction returned no parsed output (refusal or empty response)")
    if not isinstance(parsed, DocumentExtraction):
        parsed = DocumentExtraction.model_validate(parsed)
    return parsed, telemetry


# =====================================================================
# Verification. Each extracted fact must (1) point at a known document, (2) quote text that
# exists there, and (3) contain its own value inside that quote. Failures go to
# audit.excluded_evidence and never reach policy_engine.py.
# =====================================================================


def _source_id(docs: dict[str, DocRecord], ref: str) -> str:
    doc = docs.get(ref)
    return doc.doc_id if doc else f"unresolved-ref:{ref}"


def _doc_quote(doc: DocRecord, quote: str | None) -> tuple[int, EvidenceQuote] | None:
    if not quote:
        return None
    located = locate_quote(doc.text, quote)
    if located is None:
        return None
    line_index, source_text = located
    return line_index, EvidenceQuote(
        source_type="document", source_id=doc.doc_id, date=doc.date.isoformat(), text=source_text, verbatim=True
    )


def _contains(haystack: str, needle: str) -> bool:
    return normalize_for_match(needle) in normalize_for_match(haystack)


def _reject_vital(
    facts: CaseFacts, kind: str, ref: str, doc_id: str | None, when: date | None, minutes: int | None,
    claimed: str, reason: str, source_type: str = "document", time_basis: str = "LOCAL",
) -> None:
    # Fail closed: a reading the model reported but Python could not verify is logged AND
    # kept as "unverified", so Rule 4 sends the chart to review unless a newer verified
    # reading of the same type supersedes it (policy_engine._check_vital). Readings that are
    # legitimately out of scope (patient-reported, after review time) do not come here.
    facts.exclude(source_type, ref, reason)
    facts.unverified_vitals.append(
        UnverifiedVital(kind=kind, source_id=ref, doc_id=doc_id, date=when, minutes=minutes, claimed=claimed, reason=reason,
                        time_basis=time_basis)
    )


def _verify_vitals(extraction: DocumentExtraction, docs: dict[str, DocRecord], facts: CaseFacts) -> None:
    seen: set[tuple[str, str, str, int]] = set()
    for item in extraction.vital_readings:
        doc = docs.get(item.doc_ref)
        ref = f"{_source_id(docs, item.doc_ref)}:{item.measurement}:{item.value_text}"
        # Time of a reading that later fails verification; only trusted if written in the note.
        claimed_minutes = (
            parse_clock_time(item.time_text)
            if doc is not None and item.time_text and _contains(doc.text, item.time_text.strip())
            else None
        )
        if doc is None:
            _reject_vital(facts, item.measurement, ref, None, None, None, item.value_text,
                          "extracted vital references an unknown doc_ref")
            continue
        located = _doc_quote(doc, item.quote)
        if located is None:
            _reject_vital(facts, item.measurement, ref, doc.doc_id, doc.date, claimed_minutes, item.value_text,
                          "extracted vital quote not found verbatim in document")
            continue
        line_index, evidence = located
        if item.context != "MEASURED_AT_THIS_ENCOUNTER":
            facts.exclude("document", ref, f"vital not measured at this encounter ({item.context})")
            continue

        reading_date = doc.date
        if item.measured_date_text:
            stated = find_dates(item.measured_date_text)
            if len(stated) != 1 or not _contains(doc.text, item.measured_date_text):
                _reject_vital(facts, item.measurement, ref, doc.doc_id, doc.date, claimed_minutes, item.value_text,
                              f"measured date {item.measured_date_text!r} not verifiable")
                continue
            reading_date = stated[0]
            if _is_after_review(reading_date, facts):
                facts.exclude("document", ref, f"reading dated {reading_date} after review time")
                continue

        minutes = parse_clock_time(item.time_text)
        if item.time_text and (minutes is None or not _contains(doc.text, item.time_text.strip())):
            facts.warnings.append(f"{ref}: time {item.time_text!r} not verifiable; reading kept without a time")
            minutes = None

        # (line, column) of the value in its note: the tie-breaker most_recent_reading uses
        # for untimed rechecks/addenda written later in the same document.
        offset = normalize_for_match(evidence.text).find(normalize_for_match(item.value_text))
        position = (line_index, max(offset, 0))

        if item.measurement == "BLOOD_PRESSURE":
            match = re.fullmatch(r"\s*(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mmhg)?\s*", item.value_text, re.IGNORECASE)
            if not match:
                _reject_vital(facts, item.measurement, ref, doc.doc_id, reading_date, minutes, item.value_text,
                              "blood pressure value is not in systolic/diastolic form")
                continue
            systolic, diastolic = int(match.group(1)), int(match.group(2))
            in_quote = any(
                (int(s), int(d)) == (systolic, diastolic)
                for s, d in re.findall(r"(\d{2,3})\s*/\s*(\d{2,3})", evidence.text)
            )
            if not in_quote:
                _reject_vital(facts, item.measurement, ref, doc.doc_id, reading_date, minutes, item.value_text,
                              "blood pressure value not present in quoted text")
                continue
            if not _plausible_bp(systolic, diastolic):
                _reject_vital(facts, item.measurement, ref, doc.doc_id, reading_date, minutes, item.value_text,
                              f"blood pressure {systolic}/{diastolic} implausible")
                continue
            key = (doc.doc_id, "BP", f"{systolic}/{diastolic}", line_index)
            if key in seen:
                continue
            seen.add(key)
            facts.vitals.append(
                VitalRecord(
                    kind="BLOOD_PRESSURE", source_type="document", source_id=doc.doc_id, date=reading_date,
                    minutes=minutes, position=position, systolic=systolic, diastolic=diastolic, temperature_f=None,
                    display=f"BP {systolic}/{diastolic} mmHg", evidence=evidence,
                )
            )
        else:
            value_match = re.fullmatch(r"\s*(\d{2,3}(?:\.\d{1,2})?)\s*", item.value_text)
            if not value_match:
                _reject_vital(facts, item.measurement, ref, doc.doc_id, reading_date, minutes, item.value_text,
                              "temperature value is not numeric")
                continue
            raw_value = value_match.group(1)
            if not re.search(rf"(?<![\d.]){re.escape(raw_value)}(?![\d])", evidence.text):
                _reject_vital(facts, item.measurement, ref, doc.doc_id, reading_date, minutes, item.value_text,
                              "temperature value not present in quoted text")
                continue
            # Unit priority: the unit written next to the number in the source, then the
            # model's unit, then physiological range (30-45 => C, 86-113 => F). Conversion
            # uses Decimal so 38.0 C is exactly 100.4 F (not a fever) with no float drift.
            # fold_degree_signs maps ℃/℉/º/corrupted forms to °; also accepts "deg", "degrees", "Celsius", "Fahrenheit".
            unit_in_text = re.search(
                rf"(?<![\d.]){re.escape(raw_value)}\s*(?:°|º|deg(?:rees?)?\.?)?\s*(celsius|fahrenheit|c|f)\b",
                fold_degree_signs(evidence.text),
                re.IGNORECASE,
            )
            value = Decimal(raw_value)
            if unit_in_text:
                unit = unit_in_text.group(1)[0].upper()
                if item.unit in {"C", "F"} and item.unit != unit:
                    facts.warnings.append(f"{ref}: extractor unit {item.unit} overridden by unit written in text ({unit})")
            elif item.unit in {"C", "F"}:
                unit = item.unit
            elif Decimal("30") <= value <= Decimal("45"):
                unit = "C"
                facts.warnings.append(f"{ref}: no unit written; {value} treated as Celsius by physiological range")
            elif Decimal("86") <= value <= Decimal("113"):
                unit = "F"
                facts.warnings.append(f"{ref}: no unit written; {value} treated as Fahrenheit by physiological range")
            else:
                _reject_vital(facts, item.measurement, ref, doc.doc_id, reading_date, minutes, item.value_text,
                              f"temperature {value} has no unit and is implausible")
                continue
            temperature_f = value if unit == "F" else (value * Decimal(9) / Decimal(5) + Decimal(32))
            if not _plausible_temperature_f(temperature_f):
                _reject_vital(facts, item.measurement, ref, doc.doc_id, reading_date, minutes, item.value_text,
                              f"temperature {value} {unit} implausible")
                continue
            key = (doc.doc_id, "TEMP", f"{value}{unit}", line_index)
            if key in seen:
                continue
            seen.add(key)
            display = f"Temp {value} {unit}" if unit == "F" else f"Temp {value} C (= {temperature_f.quantize(Decimal('0.01'))} F)"
            facts.vitals.append(
                VitalRecord(
                    kind="TEMPERATURE", source_type="document", source_id=doc.doc_id, date=reading_date,
                    minutes=minutes, position=position, systolic=None, diastolic=None, temperature_f=temperature_f,
                    display=display, evidence=evidence,
                )
            )


def _is_antiplatelet_only(name: str) -> bool:
    drugs = classify_drug(name)
    return bool(drugs) and all(drug_class == "ANTIPLATELET" for _, drug_class in drugs)


def _verify_medication_mentions(extraction: DocumentExtraction, docs: dict[str, DocRecord], facts: CaseFacts) -> None:
    seen: set[tuple[str, str, str, str]] = set()
    for item in extraction.medication_mentions:
        # Antiplatelets are outside Rule 3 and unused downstream; dropping them before
        # verification keeps optional over-reporting by the model out of the audit log.
        if _is_antiplatelet_only(item.drug_as_written):
            continue
        doc = docs.get(item.doc_ref)
        ref = f"{_source_id(docs, item.doc_ref)}:{item.drug_as_written}"
        if doc is None:
            facts.exclude("document", ref, "medication mention references an unknown doc_ref")
            continue
        located = _doc_quote(doc, item.quote)
        if located is None or not _contains(located[1].text, item.drug_as_written):
            facts.exclude("document", ref, "medication mention quote/drug name not found verbatim in document")
            continue
        _, evidence = located
        drugs = classify_drug(item.drug_as_written)
        anticoagulants = [generic for generic, drug_class in drugs if drug_class == "ANTICOAGULANT"]
        if anticoagulants:
            entries = [(generic, "ANTICOAGULANT") for generic in anticoagulants]
        elif drugs:
            entries = [(drugs[0][0], drugs[0][1])]
        else:
            entries = [(None, None)]
        for generic, drug_class in entries:
            key = (doc.doc_id, generic or item.drug_as_written.casefold(), item.status, evidence.text)
            if key in seen:
                continue
            seen.add(key)
            facts.medication_mentions.append(
                MedicationMention(
                    doc_id=doc.doc_id, date=doc.date, drug_as_written=item.drug_as_written, generic=generic,
                    drug_class=drug_class, class_hint=item.class_hint, status=item.status, evidence=evidence,
                )
            )


def _verify_instruction(doc: DocRecord, quote: str | None, kind: str, ref: str, facts: CaseFacts) -> EvidenceQuote | None:
    if not quote:
        return None
    located = _doc_quote(doc, quote)
    if located is None:
        facts.exclude("document", ref, f"{kind} instruction quote not found verbatim in document")
        return None
    if not TIMING_PATTERN.search(quote):
        facts.exclude("document", ref, f"{kind} instruction has no concrete timing: {quote!r}")
        return None
    return located[1]


def _verify_plans(extraction: DocumentExtraction, docs: dict[str, DocRecord], facts: CaseFacts) -> None:
    # Merge per (document, drug): the model sometimes splits one plan into a hold-only and
    # a resume-only entry. Entries with neither instruction are dropped below; the
    # deferral text itself is still cited by policy_engine.drug_mention_lines.
    merged: dict[tuple[str, str], dict[str, EvidenceQuote | None]] = {}
    for item in extraction.anticoagulation_plans:
        doc = docs.get(item.doc_ref)
        ref = f"{_source_id(docs, item.doc_ref)}:{item.drug_as_written}"
        if doc is None:
            facts.exclude("document", ref, "anticoagulation plan references an unknown doc_ref")
            continue
        if not _contains(doc.text, item.drug_as_written):
            facts.exclude("document", ref, "anticoagulation plan drug name not found in document")
            continue
        generics = anticoagulant_generics(item.drug_as_written)
        if not generics and _is_antiplatelet_only(item.drug_as_written):
            continue  # recognized antiplatelet: outside Rule 3, not an extraction error
        if not generics:
            facts.exclude("document", ref, f"plan drug {item.drug_as_written!r} is not a recognized anticoagulant")
            continue
        hold = _verify_instruction(doc, item.hold_instruction_quote, "hold", ref, facts)
        resume = _verify_instruction(doc, item.resume_instruction_quote, "resume", ref, facts)
        for generic in generics:
            slot = merged.setdefault((doc.doc_id, generic), {"hold": None, "resume": None})
            slot["hold"] = slot["hold"] or hold
            slot["resume"] = slot["resume"] or resume
    for (doc_id, generic), slot in sorted(merged.items()):
        if slot["hold"] is None and slot["resume"] is None:
            continue  # no verifiable instruction; the document's drug lines are still cited by Rule 3
        facts.plans.append(
            PlanRecord(doc_id=doc_id, date=docs[doc_id].date, generic=generic, hold=slot["hold"], resume=slot["resume"])
        )


def _verify_consents(extraction: DocumentExtraction, docs: dict[str, DocRecord], facts: CaseFacts) -> None:
    by_doc: dict[str, list[ExtractedConsent]] = {}
    for item in extraction.consents:
        doc = docs.get(item.doc_ref)
        if doc is None or doc.doc_class != "SURGICAL_CONSENT":
            facts.exclude("document", _source_id(docs, item.doc_ref), "consent finding for a document not classified as a consent")
            continue
        by_doc.setdefault(doc.doc_id, []).append(item)

    for doc in (d for d in facts.documents if d.doc_class == "SURGICAL_CONSENT"):
        items = by_doc.get(doc.doc_id, [])
        notes: list[str] = []
        status: SignatureStatus = "CANNOT_DETERMINE"
        signature: EvidenceQuote | None = None
        procedure: EvidenceQuote | None = None
        procedure_name: str | None = None
        if not items:
            notes.append("extractor returned no finding for this consent document")
        elif len({item.patient_signature_status for item in items}) > 1:
            notes.append("extractor returned conflicting signature findings")
        else:
            item = items[0]
            status = item.patient_signature_status
            located = _doc_quote(doc, item.signature_quote)
            signature = located[1] if located else None
            # A "signed" claim must survive two text checks: the signature line exists
            # verbatim, and it does not read as pending/blank/not completed.
            if status == "SIGNED_BY_PATIENT":
                if signature is None:
                    status = "CANNOT_DETERMINE"
                    notes.append("signature quote not found verbatim")
                elif UNSIGNED_CONSENT_PATTERN.search(signature.text) or not re.search(r"sign|/s/", signature.text, re.IGNORECASE):
                    status = "CANNOT_DETERMINE"
                    notes.append("signature line does not unambiguously show a patient signature")
            if not item.procedure_quote:
                # Deterministic fallback when the model omits it: the form's own "Procedure:" line.
                labelled = re.search(r"^\s*(?:planned\s+)?procedure\s*:\s*(\S.*)$", doc.text, re.IGNORECASE | re.MULTILINE)
                if labelled:
                    item = item.model_copy(update={"procedure_quote": labelled.group(0).strip()})
                    notes.append("procedure taken from the form's 'Procedure:' line")
            if item.procedure_quote:
                located_procedure = _doc_quote(doc, item.procedure_quote)
                if located_procedure:
                    procedure = located_procedure[1]
                    procedure_name = re.sub(r"^\s*(planned\s+)?procedure\s*:\s*", "", item.procedure_quote, flags=re.IGNORECASE).strip()
                else:
                    notes.append("procedure quote not found verbatim")
        facts.consents.append(
            ConsentRecord(
                doc_id=doc.doc_id, date=doc.date, title=doc.title, status=status, signature=signature,
                procedure_name=procedure_name, procedure=procedure, notes=tuple(notes),
            )
        )


def _verify_history_and_physicals(extraction: DocumentExtraction, docs: dict[str, DocRecord], facts: CaseFacts) -> None:
    by_doc = {docs[item.doc_ref].doc_id: item for item in extraction.history_and_physicals if item.doc_ref in docs}
    for doc in (d for d in facts.documents if d.doc_class == "HISTORY_AND_PHYSICAL"):
        item = by_doc.get(doc.doc_id)
        status: Literal["COMPLETED", "INCOMPLETE", "CANNOT_DETERMINE"] = "CANNOT_DETERMINE"
        evidence = None
        if item is not None:
            located = _doc_quote(doc, item.quote)
            evidence = located[1] if located else None
            status = item.completion_status
            if status == "INCOMPLETE" and evidence is None:
                facts.warnings.append(f"{doc.doc_id}: H&P reported incomplete without a verifiable quote; not treated as incomplete")
                status = "CANNOT_DETERMINE"
        facts.history_and_physicals.append(
            HistoryAndPhysicalRecord(doc_id=doc.doc_id, date=doc.date, title=doc.title, status=status, evidence=evidence)
        )


def _verify_procedure_mentions(extraction: DocumentExtraction, docs: dict[str, DocRecord], facts: CaseFacts) -> None:
    # Structured fields take priority: document mentions are only used (or logged) when
    # the corresponding structured field was unusable and the fallback was requested.
    date_mentions = extraction.procedure_date_mentions if facts.needs_procedure_date_fallback else []
    risk_mentions = extraction.procedure_risk_mentions if facts.needs_procedure_risk_fallback else []
    for item in date_mentions:
        doc = docs.get(item.doc_ref)
        located = _doc_quote(doc, item.quote) if doc else None
        if doc is None or located is None:
            facts.exclude("document", _source_id(docs, item.doc_ref), "procedure date mention quote not found verbatim")
            continue
        scheduled_for = None
        if item.status == "SCHEDULED":
            dates = find_dates(item.date_text or "")
            if len(dates) != 1 or not _contains(located[1].text, item.date_text or ""):
                facts.exclude("document", _source_id(docs, item.doc_ref), f"scheduled date {item.date_text!r} not verifiable in quote")
                continue
            scheduled_for = dates[0]
        facts.procedure_date_mentions.append(
            ProcedureDateMention(doc_id=doc.doc_id, doc_date=doc.date, scheduled_for=scheduled_for, evidence=located[1])
        )

    for item in risk_mentions:
        doc = docs.get(item.doc_ref)
        located = _doc_quote(doc, item.quote) if doc else None
        if doc is None or located is None:
            facts.exclude("document", _source_id(docs, item.doc_ref), "procedure risk mention quote not found verbatim")
            continue
        text = located[1].text
        if "risk" not in text.lower() or PROCEDURE_RISK_EXCLUDE_PATTERN.search(text):
            facts.exclude("document", _source_id(docs, item.doc_ref), f"risk mention is not a procedure risk category: {text!r}")
            continue
        risk: ProcedureRisk | None = None
        if item.risk_category != "NOT_ASSIGNED":
            if not re.search(rf"\b{item.risk_category}\b", text, re.IGNORECASE):
                facts.exclude("document", _source_id(docs, item.doc_ref), f"risk category {item.risk_category} not written in quote")
                continue
            risk = item.risk_category
        facts.procedure_risk_mentions.append(
            ProcedureRiskMention(doc_id=doc.doc_id, doc_date=doc.date, risk=risk, evidence=located[1])
        )


def verify_extraction(extraction: DocumentExtraction, facts: CaseFacts) -> CaseFacts:
    # The model sees short refs; exact doc_ids are also accepted. Refs take precedence.
    docs = {doc.doc_id: doc for doc in facts.documents} | document_refs(facts.documents)
    _verify_vitals(extraction, docs, facts)
    _verify_medication_mentions(extraction, docs, facts)
    _verify_plans(extraction, docs, facts)
    _verify_consents(extraction, docs, facts)
    _verify_history_and_physicals(extraction, docs, facts)
    _verify_procedure_mentions(extraction, docs, facts)
    return facts


def extract_case_facts(
    submission: PatientSubmission,
    *,
    model: str,
    client: Any,
    facility_timezone: str | None = None,
    on_telemetry: Callable[[ExtractionTelemetry], None] | None = None,
) -> CaseFacts:
    facts = preprocess(submission, model=model, facility_timezone=facility_timezone)
    extraction, _ = call_extractor(client, model=model, facts=facts, on_telemetry=on_telemetry)
    return verify_extraction(extraction, facts)


__all__ = [
    "EXTRACTION_PROMPT_VERSION",
    "PROMPT_CACHE_KEY",
    "ExtractionTelemetry",
    "EXTRACTION_SYSTEM_PROMPT",
    "CaseFacts",
    "DocumentExtraction",
    "ExtractionError",
    "extract_case_facts",
    "preprocess",
    "verify_extraction",
]
